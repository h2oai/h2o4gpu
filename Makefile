default: all

all:
	$(MAKE) -j all -C src/
	$(MAKE) -j all -C examples/cpp/
	$(MAKE) -j all -C src/interface_c
	$(MAKE) -j all -C src/interface_py
	$(MAKE) -j all -C src/interface_r

allclean:
	$(MAKE) -j clean -C src/
	$(MAKE) -j all -C src/
	$(MAKE) -j clean -C examples/cpp/
	$(MAKE) -j all -C examples/cpp/
	$(MAKE) -j clean -C src/interface_c
	$(MAKE) -j all -C src/interface_c
	$(MAKE) -j clean -C src/interface_py
	$(MAKE) -j all -C src/interface_py
	$(MAKE) -j clean -C src/interface_r
	$(MAKE) -j all -C src/interface_r

clean:
	$(MAKE) -j clean -C src/
	$(MAKE) -j clean -C examples/cpp/
	$(MAKE) -j clean -C src/interface_c
	$(MAKE) -j clean -C src/interface_py
	$(MAKE) -j clean -C src/interface_r
