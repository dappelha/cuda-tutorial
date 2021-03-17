// System includes
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>


/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
void matvec_serial(double *A, double *x, double *y, int n) {
    printf("hello in subroutine \n");

    for(int j=0;j<n;++j){
      y[j] = 0;
      for(int i=0;i<n;++i){
	y[j] = y[j] + A[i + n*j] * x[i];
      }
    }

}



/**
 * Program main
 */
int main(int argc, char **argv) {
    printf("[Matrix Vector Multiply Example CUDA] - Starting...\n");

    int n;

    if(argc < 2){
      n = 20*1024;
    }    
    else{
      n = atoi(argv[1]);
    }
    // Allocate memory:
    double *A, *x, *y;
    A = (double*) malloc(n*n*sizeof(*A));
    x = (double*) malloc(n*n*sizeof(*x));
    y = (double*) malloc(n*n*sizeof(*y));


    for(int j=0;j<n;++j){
      x[j] = 1;
      for(int i=0;i<n;++i){
	A[i + n*j]=1;
      }
    }

    matvec_serial(A, x, y, n);

    printf("y[2] = %f \n", y[2]);

}

