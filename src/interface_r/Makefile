default: all

all:
	MAKE="make -j" R CMD INSTALL --build h2oaiglm
	echo "Default using gpulib.  Edit interface_r/src/config2.mk and choose TARGET as cpulib or gpulib."
