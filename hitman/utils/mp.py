import multiprocessing
from multiprocessing.queues import Queue


class MPCounter(object):

    def __init__(self, initial_val=0):
        self.current_counter = multiprocessing.Value('i', initial_val)

    def increment(self, n=1):
        """ Increment the current counter by n (default = 1) """
        with self.current_counter.get_lock():
            self.current_counter.value += n

    def decrement(self, n=1):
        """ Decrement the current counter by n (default = 1) """
        self.increment(-n)

    @property
    def current_value(self):
        """ The current value of the counter """
        return self.current_counter.value


class MPQueue(Queue):

    def __init__(self, maxsize=0, *args, ctx=multiprocessing.get_context()):
        super().__init__(maxsize=maxsize, *args, ctx=ctx)
        self.q_size = MPCounter(0)

    def put(self, *args, **kwargs):
        super().put(*args, **kwargs)
        self.q_size.increment(1)

    def get(self, *args, **kwargs):
        val = super().get(*args, **kwargs)
        self.q_size.decrement(1)
        return val

    def qsize(self):
        """ Overcomes the current problems of queue in Mac os X (See multiprocessing.Queue doc) """
        return self.q_size.current_value

    def empty(self):
        """ Overcomes the current problems of queue in Mac os X (See multiprocessing.Queue doc) """
        return self.q_size.current_value == 0

    def full(self):
        """ Overcomes the current problems of queue in Mac os X (See multiprocessing.Queue doc) """
        return self.q_size.current_value == self._maxsize
