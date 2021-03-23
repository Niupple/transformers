from queue import Queue

def wrapper(func, qin: Queue, qout: Queue, *args, sentinal=None, **kwargs):
    while True:
        x = qin.get()
        if x is sentinal:
            break
        ret = func(x, *args, **kwargs)
        qout.put(ret)
