EXE = test.exe

ALL_SRC	= matVec.cu

TEMP_OBJS = $(ALL_SRC:.c=.o)
ALL_OBJS = $(TEMP_OBJS:.cu=.o)

TARGET_ARCH ?= $(HOST_ARCH)
TARGET_SIZE := 64

# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda-10.1

HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS     :=
LDFLAGS     :=


# Gencode arguments
SMS := 70

GENCODE_FLAGS += -gencode arch=compute_$(SMS),code=compute_$(SMS)

# Target rules
all: build

build: matVec

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif

matVec.o:matVec.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

matVec: matVec.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	$(EXEC) ./matVec

clean:
	rm -f $(EXE) $(ALL_OBJ)

