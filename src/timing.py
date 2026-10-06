import time
from contextlib import contextmanager


@contextmanager
def timed(timings, name):
    # usage:  with timed(timings, "rerank"):  ...code to measure...
    # stores the elapsed seconds in timings[name], even if the block raises an error
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = round(time.perf_counter() - start, 3)
