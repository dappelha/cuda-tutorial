// System includes
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>
extern const int rrmax=32;

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


__global__ void matvec_cudav2(double *A, double *x, double *y, int n) {
  
  //int const rrmax=32;
  extern __shared__ double s[];
  double *A_s= &s[0];
  double *x_s= &A_s[blockDim.x*rrmax];
  double *y_s= &x_s[blockDim.x];
  
  //__shared__ double A_s[blockDim.x*rrmax];
  //__shared__ double x_s[blockDim.x];
  //__shared__ double y_s[rrmax];

    // this code assumes enough blocks are launched to cover the space.
    // TODO fix up a outer loop over rows to handle less blocks.

    for(int q=threadIdx.x; q<n; q+= blockDim.x){
      // load a section of x into shared memory:
      x_s[threadIdx.x] = x[q];
      for( int rr=0; rr<rrmax; rr++ ){
	// load a partial row of matrix A into shared memory tile
	A_s[threadIdx.x + rr*blockDim.x] = A[q + rr*n + blockIdx.x*rrmax*n];
      }

      __syncthreads();
      // use the partials that you loaded to have each thread
      // compute a partial dot product of that row A with partial column x

      for(int rr=threadIdx.x;rr<rrmax;rr+=blockDim.x){
	y_s[rr] = 0;
	for(int i=0;i<blockDim.x;++i){
	  y_s[rr] = y_s[rr] + A_s[i + rrmax*rr] * x_s[i];
	}
	// accumulate shared memory result into global y
	y[rr + blockIdx.x*rrmax] = y[rr + blockIdx.x*rrmax] + y_s[rr];
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

    int xthreads = 512;
    dim3 blocks(n/xthreads,1,1);
    dim3 threads(xthreads,1,1);
    int shmem = 0;

    matvec_cudav1<<<blocks,threads,shmem,0 >>>(d_A, d_x, d_y, n);    

    cudaMemcpy(y, d_y, n*sizeof(*y), cudaMemcpyDeviceToHost);

    printf("y[2] = %f \n", y[2]);

    // Call v2 with shared memory for coalesced reads.

    xthreads = 32;
    //#define rrmax=32
    blocks = dim3(n/xthreads,1,1);
    threads= dim3(xthreads,1,1);
    shmem = (xthreads*rrmax + rrmax + xthreads)*8;

    matvec_cudav2<<<blocks,threads,shmem,0 >>>(d_A, d_x, d_y, n);    

    cudaMemcpy(y, d_y, n*sizeof(*y), cudaMemcpyDeviceToHost);

    printf("y[2] = %f \n", y[2]);



}

