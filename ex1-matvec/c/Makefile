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
#NVCCFLAGS   := -m${TARGET_SIZE} 
CCFLAGS     := --resource-usage -Xptxas -v --maxrregcount 64
LDFLAGS     := --resource-usage -Xnvlink -v


# Gencode arguments
SMS := 70

GENCODE_FLAGS += -gencode arch=compute_$(SMS),code=sm_$(SMS)

# Target rules
all: build

build: $(EXE)

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif

matVec.o:matVec.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(EXE): matVec.o
	$(EXEC) $(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	$(EXEC) $(EXE)

clean:
	rm -f $(EXE) $(ALL_OBJ) *.o

