EXE = test.exe

ALL_SRC	= main.F90 matvec.F90 nvtx_mod.F90

TEMP_OBJS = $(ALL_SRC:.c=.o)
ALL_OBJS = $(TEMP_OBJS:.F90=.o)



FC = mpif90
CUDAFLAGS = -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,lineinfo,cuda10.1,unroll -fast -Mfprelaxed -L/usr/local/cuda/lib64 -lnvToolsExt
CUDAFLAGS+= -Minfo=inline,accel -acc
F_FLAGS=-gopt -O3 -mp -Minfo
LINK_FLAGS = -ta=tesla:pinned
C_FLAGS = -O2 	
F_FLAGS+=$(CUDAFLAGS)


CC = mpicc



.SUFFIXES:
.SUFFIXES: .o .F90 .c .mod

all: $(ALL_OBJS) 
	$(FC) $(F_FLAGS) $(LINK_FLAGS) $(ALL_OBJS) -o $(EXE)

.F90.o:
	$(FC) $(F_FLAGS) -c  $<  -o $@
.c.o:
	$(CC) $(C_FLAGS) -c  $<  -o $@

clean:
	rm -f *.o *.mod $(EXE)


#=============================================================================

##... Dependencies rules

main.o: \
	matvec.o \
	nvtx_mod.o

matvec.o: \
	matvec.F90
