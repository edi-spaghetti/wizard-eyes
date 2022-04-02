

class Timeout(object):

    def __repr__(self):
        return f'Timeout<{self.time_left:.3f}>'

    def __init__(self, client, offset):
        self.client = client
        self.created_at = self.client.time
        self.offset = self.created_at + offset

    @property
    def time_left(self):
        return self.offset - self.client.time
