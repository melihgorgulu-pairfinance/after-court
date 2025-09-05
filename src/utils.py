from src.app_logger import logging


def time_decorator(func):
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Execution time: {end_time - start_time:.2f} seconds")
        return result
    return wrapper

