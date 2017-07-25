from py3nvml.py3nvml import *
import sys

def devicecount(n_gpus=0):
    verbose = 1

    try:
        nvmlInit()
        deviceCount = nvmlDeviceGetCount()
        if verbose == 1:
            for i in range(deviceCount):
                handle = nvmlDeviceGetHandleByIndex(i)
                print("Device {}: {}".format(i, nvmlDeviceGetName(handle)))
            print("Driver Version:", nvmlSystemGetDriverVersion())
            try:
                import subprocess
                maxNGPUS = int(subprocess.check_output("nvidia-smi -L | wc -l", shell=True))
                print("\nNumber of GPUS:", maxNGPUS)
                subprocess.check_output("lscpu", shell=True)
            except:
                pass

    except Exception as e:
        print("No GPU, setting deviceCount=0")
        # print(e)
        sys.stdout.flush()
        deviceCount = 0
        pass

    if n_gpus < 0:
        if deviceCount >= 0:
            n_gpus = deviceCount
        else:
            print("Cannot automatically set n_gpus to all GPUs %d %d, trying n_gpus=1" % (n_gpus, deviceCount))
            n_gpus = 1
    return(n_gpus, deviceCount)