import time


class Timeout(object):

    def __init__(self, offset):
        self.created_at = time.time()
        self.offset = self.created_at + offset