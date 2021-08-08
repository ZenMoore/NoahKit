import multiprocessing
from multiprocessing import Queue
from apply import apply
import time, os, random
from multiprocessing.managers import BaseManager
import affinity

'''
learning material 1 creating: https://blog.csdn.net/zhaozhi406/article/details/8137670
learning material 2 pool : https://blog.csdn.net/brucewong0516/article/details/85788202?spm=1001.2014.3001.5501
learning material 3 queue/pipe : https://blog.csdn.net/brucewong0516/article/details/85796073?spm=1001.2014.3001.5501
learning material 4 lock/rlock : https://blog.csdn.net/brucewong0516/article/details/85798414?spm=1001.2014.3001.5501
learning material 5 distribution : https://www.liaoxuefeng.com/wiki/1016959663602400/1017631559645600
learning material 6 cpu assignment : https://www.cnblogs.com/domestique/p/7718510.html
api : https://docs.python.org/3/library/
'''

# general functions
def fun1(n):
    print(n+1)
    return n + 1

def fun2(n):
    print(n+2)
    return n + 2


class Callable1(object):
    def __init__(self,func, args):
        self.func = func
        self.args = args

    def __call__(self):
        apply(self.func, self.args)


class SubProcess1(multiprocessing.Process):
    def __init__(self, name):
        multiprocessing.Process.__init__(self, name=name)

    def run(self):
        print('this is subprocess')


# queue functions
def _write(q,urls):
    print('Process(%s) is writing...' % os.getpid())
    for url in urls:
        q.put(url)  # put one message in queue
        print('Put %s to queue...' % url)
        time.sleep(random.random())


def _read(q):
    print('Process(%s) is reading...' % os.getpid())
    while True:
        url = q.get(True)  # get one message in queue
        print('Get %s from queue.' % url)


# pipe function
def send(pipe):
    pipe.send(['spam'] + [42, 'egg'])   # send
    # pipe.recv()  # we can also receive message using the same pipe end
    pipe.close()


# distribution class
class QueueManager(BaseManager):  # on both server and client
    pass



if __name__ == '__main__':
    # must in main

    'create process'
    # similar to threading.Thread
    # the function such as setDaemon is also similar to Thread, please see my_threading for more details
    process1 = multiprocessing.Process(target=fun1, args=(0, ))
    process1.start()
    # process1.join()

    process2 = multiprocessing.Process(target=Callable1(fun1, (0,)))
    process2.start()
    # process2.join()

    process3 = SubProcess1('process3')
    process3.start()
    process3.join()

    'process pool'
    # we use pool when there are too many p
    pool = multiprocessing.Pool(processes=4)
    for i in range(500):  # there are 500 tasks for example
        pool.apply_async(fun1, args=[i,])  # don't sleep main process, but the main process is too fast, we have not time to do subprocess
        pool.apply(fun2, args=[0,])  # sleep main process, we use this
    pool.close()
    pool.join()
    lists = [i for i in range(500)]
    pool.map(fun1, lists)  # the same thing as above
    pool.map_async(fun1, lists)  # the same thing as above
    pool.close()
    pool.join()
    # imap returns the iterator, we also have function terminate()

    'queue'
    # queue is for message communication between processes (many)
    q = Queue()
    _writer1 = multiprocessing.Process(target=_write, args=(q, ['url_1', 'url_2', 'url_3']))
    _writer2 = multiprocessing.Process(target=_write, args=(q, ['url_4', 'url_5', 'url_6']))
    _reader = multiprocessing.Process(target=_read, args=(q,))
    _writer1.start()
    _writer2.start()
    _reader.start()
    _writer1.join()
    _writer2.join()
    _reader.terminate()

    'pipe'
    # pipe is for message communication between only two processes
    # we can set the two ends of the pipe on two necessary locations, emp: function and main process
    (con1, con2) = multiprocessing.Pipe()
    sender = multiprocessing.Process(target=send, args=(con1,))
    sender.start()
    print("con2 got: %s" % con2.recv())  # con2 can also send
    con2.close()

    'lock & rlock'
    # the same thing as threading, please see my_threading

    'event & semaphore & condition'
    # also the same for : semaphore, event, condition

    'distribution'
    # above is for one machine multi-unit : CPU0+CPU1+GPU0 on my own laptop for example
    # but we can also use multi-machine : distributed multiprocessing
    # using manager

    url = ''  # on server: ''; on client: 'url_of_server'

    task_queue = Queue()  # only on server
    result_queue = Queue()  # only on server

    # register two queues on internet, callable correlates queue object
    QueueManager.register('get_task_queue', callable=lambda: task_queue)
    QueueManager.register('get_result_queue', callable=lambda: result_queue)

    # bind port 5000, set auth code
    manager = QueueManager(address=(url, 5000), authkey=b'abc')
    # manager.connect()  # on server: needn't; on client: connect first

    # start queue
    manager.start()

    # get queue object on internet
    task = manager.get_task_queue()
    result = manager.get_result_queue()

    for i in range(10):
        n = random.randint(0, 10000)
        print('Put task %d...' % n)
        task.put(n)

    print('Try get results...')
    for i in range(10):
        r = result.get(timeout=10)
        print('Result: %s' % r)

    manager.shutdown()
    print('master exit.')

    'unit assignment'
    # we want to assign a specific cpu for specific process
    # affinity can only assign cpu, not gpu
    p = multiprocessing.Process(target=time.sleep, args=(1000, ))
    p.start()
    pid = p.pid
    value = 1  # cpu1
    affinity._get_handle_for_pid(pid, ro=True)  # get process by pid
    affinity.get_process_affinity_mask(pid)  # get affinity mask by pid ,return long, for example, '2l' means long on cpu2
    affinity.set_process_affinity_mask(pid, value)  # bind process pid to cpu-value
