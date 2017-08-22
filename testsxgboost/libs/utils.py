import os
import multiprocessing


def get_number_processors():
    try:
        num = os.cpu_count()
    except:
        num = multiprocessing.cpu_count()
    return num  

