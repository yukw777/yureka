import time
import pdb
import sys
import os


class ForkablePdb(pdb.Pdb):

    _original_stdin_fd = sys.stdin.fileno()
    _original_stdin = None

    def __init__(self):
        pdb.Pdb.__init__(self, nosigint=True)

    def _cmdloop(self):
        current_stdin = sys.stdin
        try:
            if not self._original_stdin:
                self._original_stdin = os.fdopen(self._original_stdin_fd)
            sys.stdin = self._original_stdin
            self.cmdloop()
        finally:
            sys.stdin = current_stdin


def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        print('%r  %2.2f ms' %
              (method.__name__, (te - ts) * 1000))
        return result
    return timed


def print_flush(*args, **kwargs):
    print(*args, flush=True, **kwargs)
