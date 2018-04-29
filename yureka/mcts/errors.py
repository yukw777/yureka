import attr


@attr.s
class MCTSError(Exception):
    node = attr.ib()
    message = attr.ib()
