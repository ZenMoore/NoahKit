import threading
from apply import apply
import time

'''
learning material 1 creating : https://blog.csdn.net/zhaozhi406/article/details/8137670
learning material 2 daemon : https://blog.csdn.net/brucewong0516/article/details/81028716
learning material 3 lock/rlock : https://blog.csdn.net/brucewong0516/article/details/81050939
learning material 4 condition : https://blog.csdn.net/brucewong0516/article/details/84587522?spm=1001.2014.3001.5501
learning material 5 event : https://blog.csdn.net/brucewong0516/article/details/84588804?spm=1001.2014.3001.5501
learning material 6 timer : https://blog.csdn.net/brucewong0516/article/details/84589616?spm=1001.2014.3001.5501
learning material 7 local : https://blog.csdn.net/brucewong0516/article/details/84589806?spm=1001.2014.3001.5501
api : https://docs.python.org/3/library/
'''

'create thread'
# pass in a function
def fun1(n:int):
    return n+1

t1 = threading.Thread(target=fun1, args=[0,])
t1.start()
t1.join()  # main thread will sleep until t1 finishes

# pass in a callable
def fun2(n:int):
    return n+2

class Callable2(object):
    def __init__(self, func, args):
        self.fun = func
        self.args = args

    def __call__(self):
        apply(self.fun, self.args)

t2 = threading.Thread(target=Callable2(fun2, (0, )))
t2.start()
t2.join()



# inherit a thread
class SubThread(threading.Thread):
    def __init__(self, name):
        # super(SubThread, self).__init__(self, name=name)  # not right
        threading.Thread.__init__(self, name=name)

    def run(self):
        print('this is subthread.')

th3 = SubThread('th3')
th3.start()
th3.join()

'limitation'
# limitation : global interpreter lock : only one thread can use the interpreter at any time
# computing-dense thread : occupation of cpu
# but io-dense thread : when input-output, it will unleash the interpreter.
# so we should consider about the type of task.

'daemon'
th3.setDaemon(True)  # close when main thread finishes
th3.setDaemon(False)  # continue to finish when thread finishes

'''
overall, there are four strategies for protection of shared variables:
lock & rlock, condition, event, timer 
'''

'lock & rlock'
# mutex : for shared variable
# request lock -> enter into lock pool and wait -> get lock -> unleash lock
num = 0
lock = threading.RLock()
def fun3():
    lock.acquire()
    global num
    print(num+1)
    lock.release()

for i in range(10):
    t = threading.Thread(target=fun3)
    t.start()

# lock : only once, if more, dead lock
# rlock : not only once, but #acquire = #release, we can think that rlock has a pool with counter.

'condition'
# here we solve the producer-consumer problem
# condition has a lock pool and a waiting pool
# acquire -> get condition -> wait -> release lock and blocked and enter into waiting pool
# notify -> let one thread in waiting pool acquire a lock
# we can also use notifyAll
count = 500
con = threading.Condition()

class Producer(threading.Thread):
    # 生产者函数
    def run(self):
        global count
        while True:
            if con.acquire():
                # produce when count <= 1000
                if count > 1000:
                    con.wait()
                else:
                    count = count + 100
                    msg = self.name + ' produce 100, count=' + str(count)
                    print(msg)  # choose a thread in waiting pool, notify it to acquire a lock
                    con.notify()
                con.release()
                time.sleep(1)


class Consumer(threading.Thread):
    # 消费者函数
    def run(self):
        global count
        while True:
            # consume when count >= 100
            if con.acquire():
                if count < 100:
                    con.wait()

                else:
                    count = count - 5
                    msg = self.name + ' consume 5, count=' + str(count)
                    print(msg)
                    con.notify()  # choose a thread in waiting pool, notify it to acquire a lock
                con.release()
                time.sleep(1)


def test():
    for i in range(2):  # two producers
        p = Producer()
        p.start()
    for i in range(5):  # five consumers
        c = Consumer()
        c.start()


if __name__ == '__main__':
    test()


'event'
# a simplified condition, but have no lock
# event has an inner state False as default,
# wait->waiting, util another thread set(his inner state)=True, event notifies waiting threads to run
# clear->reset into False
# isSet() check
import threading
import time
event = threading.Event()
def func():
    # wait for event, sleeping
    print('%s wait for event...' % threading.currentThread().getName())
    event.wait()
    # receive event, get into running
    print('%s recv event.' % threading.currentThread().getName())
t1 = threading.Thread(target=func)
t2 = threading.Thread(target=func)
t1.start()
t2.start()

time.sleep(2)

print('MainThread set event.')
event.set()

'timer'
def fun4():
    print('hello, world')

if __name__=='__main__':
    t = threading.Timer(5.0, fun4)  # similar to threading.Thread, but call the fun4 after a specific period
    t.start()  # start, print hello world if there is not cancel behind
    t.cancel()  # but cancel cancels the running thread, so we cannot print hello world


'local'
# maintain a thread-local related things
# the local variable of one thread is not accessible for other threads.
localManager = threading.local()
lock = threading.RLock()

class MyThead(threading.Thread):
    def __init__(self, threadName, name):
        super(MyThead, self).__init__(name=threadName)
        self.__name = name

    def run(self):
        global localManager
        localManager.ThreadName = self.name
        localManager.Name = self.__name
        MyThead.ThreadPoc()

    @staticmethod
    def ThreadPoc():
        lock.acquire()
        try:
            print('Thread={id}'.format(id=localManager.ThreadName))
            print('Name={name}'.format(name=localManager.Name))
        finally:
            lock.release()

if __name__ == '__main__':
    bb = {'Name': 'bb'}
    aa = {'Name': 'aa'}
    xx = (aa, bb)
    threads = [MyThead(threadName='id_{0}'.format(i), name=xx[i]['Name']) for i in range(len(xx))]
    for i in range(len(threads)):
        threads[i].start()
    for i in range(len(threads)):
        threads[i].join()