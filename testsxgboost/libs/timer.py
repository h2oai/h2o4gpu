#code based on https://github.com/miguelgfierro/codebase/ 

from timeit import default_timer

class Timer(object):
    """Timer class.
    Examples:
        >>> big_num = 100000
        >>> t = Timer()
        >>> t.start()
        >>> for i in range(big_num):
        >>>     r = 1
        >>> t.stop()
        >>> print(t.interval)
        0.0946876304844
        >>> with Timer() as t:
        >>>     for i in range(big_num):
        >>>         r = 1
        >>> print(t.interval)
        0.0766928562442
        >>> try:
        >>>     with Timer() as t:
        >>>         for i in range(big_num):
        >>>             r = 1
        >>>             raise(Exception("Get out!"))
        >>> finally:
        >>>     print(t.interval)
        0.0757778924471

    """
    def __init__(self):
        self._timer = default_timer
    
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        """Start the timer."""
        self.start = self._timer()

    def stop(self):
        """Stop the timer. Calculate the interval in seconds."""
        self.end = self._timer()
        self.interval = self.end - self.start

