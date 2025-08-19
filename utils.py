
from typing import Union
from pandas import Timestamp
import pandas as pd


def get_object_key(doc_file_name: str, created_at: Union[Timestamp, str], doc_type: str) -> str:
    if type(created_at) is str:
        created_at = pd.to_datetime(created_at)

    day = created_at.day
    month = created_at.month
    year = created_at.year
    if day < 10:
        day = f"0{day}"
    if month < 10:
        month = f"0{month}"
        
    s3_file_key = f"{day}.{month}.{year}/{doc_file_name}"
    
    return s3_file_key


# time decorator to measure execution time
def time_decorator(func):
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"\033[94mExecution time: {end_time - start_time:.2f} seconds\033[0m")
        return result
    return wrapper