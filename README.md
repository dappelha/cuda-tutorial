# cuda-tutorial
Example of progressive steps of CUDA parallelization and optimization of a kernel.

Example 1: Matrix Vector multiply. Expect to be latency bound at first and once memory accesses and parallelisim are fixed should be bandwidth bound.
  - Serial version
  - Add cuda threads and blocks 
  - Use shared memory to fix non-coalesced access of matrix and input vector.
  - Threading inner "dot product" loop requires some coordination among threads, either with register shuffles or atomic shared memory updates.
