import time


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
