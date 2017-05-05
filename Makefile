default: all

all:
	$(MAKE) -j all -C src/
	$(MAKE) -j all -C examples/cpp/
	$(MAKE) -j all -C src/interface_c
	$(MAKE) -j all -C src/interface_py
	$(MAKE) -j all -C src/interface_r
