default: all

all:
	echo "Default using gpulib.  Edit interface_r/src/config2.mk and choose TARGET as cpulib or gpulib."

install: all
	mkdir -p ~/lib/
	MAKE="make -j" R CMD INSTALL --build h2ogpuml || MAKE="make -j" R CMD INSTALL -l ~/lib/ --build h2ogpuml

clean:
	rm -rf *.tar.gz
