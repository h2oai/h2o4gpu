location = $(CURDIR)/$(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))
WHERE_ART_THOU := $(location)
$(info ** -> $(WHERE_ART_THOU))
$(info ** ------------------------------------------------------------------ **)

DAAL_LIBS := -ldaal_core -ldaal_sequential -lpthread -lm

DAAL_INCLUDE := -I$HOME/daal/include

DAAL_HDR = \
	cpu/daal/include/debug.hpp \
	cpu/daal/include/iinput.hpp
	
DAAL_DEF = DDAAL_DEF
