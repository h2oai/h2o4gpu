default: all

all:
	mkdir -p ~/lib/
	MAKE="make -j" R CMD INSTALL --build h2oaiglm || MAKE="make -j" R CMD INSTALL -l ~/lib/ --build h2oaiglm
	echo "Default using gpulib.  Edit interface_r/src/config2.mk and choose TARGET as cpulib or gpulib."
