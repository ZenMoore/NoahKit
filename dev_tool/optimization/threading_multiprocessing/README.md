# Threading : multi-thread
## Principle of multi-thread
![](https://img-blog.csdn.net/20180713112621182?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JydWNld29uZzA1MTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
- sleeping : `join()`, `sleep()`
- waiting : `wait()`
- locked
- running : original, `notify()`

# Multiprocessing : multi-process
## Terminology
**multi-thread** : multi tasks on one unit on one machine\
**multi-process** : multi tasks on many unit on one machine or many machines\
**distribution** : multi tasks on many machines\
**concurrence** : pseudo-simultaneous through CPU-call algorithm\
**parallel** : real-simultaneous through multi-CPU or multi-machine\
**high concurrence** : concurrence with huge amount of tasks

## Unit assignment
```commandline
pip install affinity
```
```text
_get_handle_for_pid(pid, ro=True)  # get process by pid 
get_process_affinity_mask(pid)  # get affinity mask by pid ,return long, for example, '2l' means long on cpu2 
set_process_affinity_mask(pid, value)  # bind process pid to cpu-value 
```
`affinity can only assign cpu, not gpu`
 
