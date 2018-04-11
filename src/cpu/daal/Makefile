location = $(CURDIR)/$(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))
WHERE_ART_THOU := $(location)
$(info ** -> $(WHERE_ART_THOU))
$(info ** ------------------------------------------------------------------ **)

DAAL := $(shell ldconfig -p | grep daal_core)

ifndef DAAL
$(warning ** NO DAAL found. running without DAAL library.)
else
$(info ** DAAL librar installed on your system.)
endif

# detect Operating System
CFLAGS += -c
OSFLAG 				:=
ifeq ($(OS),Windows_NT)
	OSFLAG += -D WIN32
	ifeq ($(PROCESSOR_ARCHITECTURE),AMD64)
		OSFLAG += -D AMD64
	endif
	ifeq ($(PROCESSOR_ARCHITECTURE),x86)
		OSFLAG += -D IA32
	endif
	TARGET = libh2o4gpu_daal.dll
	CFLAGS += -O2 -std=c++11 -stdlib=libc++
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		OSFLAG += -D LINUX
		ifndef CC
			export CC = gcc
		endif
		ifndef CXX
			export CXX = g++
		endif
	endif
	ifeq ($(UNAME_S),Darwin)
		OSFLAG += -D OSX
		ifndef CC
			export CC = $(if $(shell which clang), clang, gcc)
		endif
		ifndef CXX
			export CXX = $(if $(shell which clang++), clang++, g++)
		endif
	endif
		UNAME_P := $(shell uname -p)
	ifeq ($(UNAME_P),x86_64)
		OSFLAG += -D AMD64
	endif
		ifneq ($(filter %86,$(UNAME_P)),)
	OSFLAG += -D IA32
		endif
	ifneq ($(filter arm%,$(UNAME_P)),)
		OSFLAG += -D ARM
	endif
	TARGET = libh2o4gpu_daal.so
	CFLAGS += -O2 -std=gnu++11 -fPIC
endif


PROJECT_ROOT = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

SRC_DIR := $(PROJECT_ROOT)
OBJ_DIR := $(PROJECT_DIR)obj
$(shell mkdir -p $(OBJ_DIR))
SRC_FILES := $(wildcard $(SRC_DIR)*.cpp)
OBJ_FILES := $(patsubst $(SRC_DIR)%.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))

DAALINCLUDE := -I$(HOME)/daal/include
DAALLIB := -ldaal_core -ldaal_sequential -lpthread -lm

ifdef DAAL
all: $(TARGET)

$(TARGET): $(OBJ_FILES)
	@echo "--- Linking Daal ---"
	$(CXX) -shared $(DAALLIB) -o $@ $^
	
$(OBJ_DIR)/%.o:	$(SRC_DIR)%.cpp
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(CPPFLAGS) $(DAALINCLUDE) -o $@ $< $(OSFLAGS)

clean:
	@echo "--- Cleaning Daal ---"
	rm -rf $(TARGET) $(OBJ_FILES) $(OBJ_DIR)
	
endif



