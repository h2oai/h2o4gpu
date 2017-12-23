# For GPU case, must modify /usr/local/cuda/include/host_config.h to add && __ICC != 1700 to #error about unsupported ICC configuration
$(warning Using CPU file)

ICCFILE := $(shell command -v icpc 2> /dev/null)

ifdef ICCFILE
USEICC=1
else
USEICC=0
endif

ifdef MKLROOT
USEMKL=1
else
USEMKL=0
endif


#local settings
USEDEBUG=0
USENVTX=0
USENCCL=0
