default: all

all:
	mkdir -p ~/lib/
	MAKE="make -j" R CMD INSTALL --build h2ogpuml || MAKE="make -j" R CMD INSTALL -l ~/lib/ --build h2ogpuml
	echo "Default using gpulib.  Edit interface_r/src/config2.mk and choose TARGET as cpulib or gpulib."

clean:
	rm -rf *.tar.gz
