"""Remote storage backends for assets.

Backends are deliberately tiny so they're easy to swap (R2, S3, B2, local).
"""
import hashlib
import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class NotFound(Exception):
    """The requested key does not exist in storage."""


class Storage(ABC):

    @abstractmethod
    def get_bytes(
            self, key: str, if_none_match: Optional[str] = None
    ) -> Tuple[Optional[bytes], Optional[str]]:
        """Fetch an object.

        :returns: (data, etag). If ``if_none_match`` equals the current etag
            data is None, meaning "unchanged" (HTTP 304).
        :raises NotFound: if the key doesn't exist.
        """

    @abstractmethod
    def put_bytes(self, key: str, data: bytes):
        """Write an object, replacing any existing one."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """True if the key exists."""


class LocalStorage(Storage):
    """Stores objects in a local folder. Used for tests and offline dev."""

    def __init__(self, root: str):
        self.root = root

    def _path(self, key: str) -> str:
        return os.path.join(self.root, *key.split('/'))

    def get_bytes(self, key, if_none_match=None):
        path = self._path(key)
        if not os.path.isfile(path):
            raise NotFound(key)
        with open(path, 'rb') as f:
            data = f.read()
        etag = hashlib.md5(data).hexdigest()
        if if_none_match is not None and if_none_match == etag:
            return None, etag
        return data, etag

    def put_bytes(self, key, data):
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            f.write(data)
        os.replace(tmp, path)

    def exists(self, key):
        return os.path.isfile(self._path(key))

    def __repr__(self):
        return f'LocalStorage({self.root!r})'


class R2Storage(Storage):
    """Cloudflare R2 via its S3-compatible API. Requires boto3."""

    def __init__(self, account_id: str, bucket: str,
                 access_key_id: str, secret_access_key: str):
        import boto3  # lazy: only needed when R2 is actually configured
        from botocore.config import Config

        self.bucket = bucket
        self.account_id = account_id
        self.client = boto3.client(
            's3',
            endpoint_url=f'https://{account_id}.r2.cloudflarestorage.com',
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name='auto',
            config=Config(
                retries={'max_attempts': 3, 'mode': 'standard'},
                connect_timeout=5, read_timeout=30,
            ),
        )

    @staticmethod
    def _code(error) -> str:
        return str(error.response.get('Error', {}).get('Code', ''))

    def get_bytes(self, key, if_none_match=None):
        from botocore.exceptions import ClientError
        kwargs = {'Bucket': self.bucket, 'Key': key}
        if if_none_match:
            kwargs['IfNoneMatch'] = if_none_match
        try:
            response = self.client.get_object(**kwargs)
        except ClientError as e:
            code = self._code(e)
            if code in ('304', 'NotModified'):
                return None, if_none_match
            if code in ('404', 'NoSuchKey', 'NotFound'):
                raise NotFound(key) from e
            raise
        return response['Body'].read(), response.get('ETag')

    def put_bytes(self, key, data):
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def exists(self, key):
        from botocore.exceptions import ClientError
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if self._code(e) in ('404', 'NoSuchKey', 'NotFound'):
                return False
            raise

    def __repr__(self):
        return f'R2Storage(bucket={self.bucket!r})'
