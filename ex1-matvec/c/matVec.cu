// System includes
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>


void matvec_serial(double *A, double *x, double *y, int n) {
    printf("hello in subroutine \n");

    for(int j=0;j<n;++j){
      y[j] = 0;
      for(int i=0;i<n;++i){
	y[j] = y[j] + A[i + n*j] * x[i];
      }
    }

}




__global__ void matvec_cudav1(double *A, double *x, double *y, int n) {

    for(int j=threadIdx.x+blockIdx.x*blockDim.x;j<n;j+=blockDim.x*gridDim.x){
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
    x = (double*) malloc(n*sizeof(*x));
    y = (double*) malloc(n*sizeof(*y));


    for(int j=0;j<n;++j){
      x[j] = 1;
      for(int i=0;i<n;++i){
    	A[i + n*j]=1;
      }
    }

    matvec_serial(A, x, y, n);

    printf("y[2] = %f \n", y[2]);

    //allocate device arrays
    double *d_A, *d_x, *d_y;
    cudaMalloc( (void**) &d_A, n*n*sizeof(*A));
    cudaMalloc( (void**) &d_x, n*sizeof(*x));
    cudaMalloc( (void**) &d_y, n*sizeof(*y));

    cudaMemcpy(d_A, A, n*n*sizeof(*A), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, n*sizeof(*x), cudaMemcpyHostToDevice);

    int xthreads = 32;
    dim3 blocks(n/xthreads,1,1);
    dim3 threads(xthreads,1,1);
    int shmem = 0;

    matvec_cudav1<<<blocks,threads,shmem,0 >>>(d_A, d_x, d_y, n);    

    cudaMemcpy(y, d_y, n*sizeof(*y), cudaMemcpyDeviceToHost);

    printf("y[2] = %f \n", y[2]);




}

