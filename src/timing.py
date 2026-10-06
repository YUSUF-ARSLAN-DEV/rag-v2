import logging
import sys
import time
from contextlib import contextmanager

# A named logger for timing lines. A StreamHandler on stdout means Docker captures the lines,
# so they show up in `docker compose logs api`. propagate=False stops uvicorn's root logger
# from printing them a second time.
logger = logging.getLogger("rag.timing")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


@contextmanager
def timed(timings, name):
    # usage:  with timed(timings, "rerank"):  ...code to measure...
    # stores the elapsed seconds in timings[name], even if the block raises an error
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = round(time.perf_counter() - start, 3)


def log_timings(label, timings):
    logger.info("%s %s", label, timings)
