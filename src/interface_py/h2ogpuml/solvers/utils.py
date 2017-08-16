import sys

def devicecount(n_gpus=0):

    deviceCount, total_mem = getgpuinfo()

    if n_gpus < 0:
        if deviceCount >= 0:
            n_gpus = deviceCount
        else:
            print("Cannot automatically set n_gpus to all GPUs %d %d, trying n_gpus=1" % (n_gpus, deviceCount))
            n_gpus = 1

    if n_gpus > deviceCount:
        n_gpus = deviceCount

    return (n_gpus, deviceCount)

# get GPU info, but do in sub-process to avoid mixing parent-child cuda contexts
# https://stackoverflow.com/questions/22950047/cuda-initialization-error-after-fork
def getgpuinfo():
    from concurrent.futures import ProcessPoolExecutor
    res=None
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(getgpuinfo_subprocess)
        res=future.result()
    return res

def getgpuinfo_subprocess():
    total_gpus=0
    total_mem=0
    try:
        import py3nvml.py3nvml
        py3nvml.py3nvml.nvmlInit()
        total_gpus = py3nvml.py3nvml.nvmlDeviceGetCount()

        total_mem = \
            min([py3nvml.py3nvml.nvmlDeviceGetMemoryInfo(py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(i)).total for i in
                 range(total_gpus)])
    except Exception as e:
        print("No GPU, setting total_gpus=0")
        print(e)
        sys.stdout.flush()
        pass
    return (total_gpus, total_mem)