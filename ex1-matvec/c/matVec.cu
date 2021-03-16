// System includes
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>


/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
void matvec_serial() {
    printf("[hello in subroutine \n");
}



/**
 * Program main
 */
int main(int argc, char **argv) {
    printf("[Matrix Vector Multiply Example CUDA] - Starting...\n");

    matvec_serial();
}

