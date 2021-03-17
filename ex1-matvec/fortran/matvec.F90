module matvec
  use cudafor
  integer, parameter :: mINT=8

contains
  
  subroutine matvec_serial(A,x,y,n)
    implicit none
    integer(mINT), intent(in) :: n
    real(kind=8), intent(in) :: A(n*n)
    real(kind=8), intent(in) :: x(n)
    real(kind=8), intent(out) :: y(n)

    ! local variables
    integer :: i, j
    
    ! Serial version of matrix vector multiplication. 
    ! y = A*x
    do j = 1 , n ! rows
       y(j) = 0.0d+0
       do i = 1, n ! columns
          y(j) = y(j) + A(i + (j-1)*n) * x(i)
       enddo
    enddo

  end subroutine matvec_serial


  subroutine matvec_cudav1(A,x,y,n)
    implicit none
    integer(mINT), intent(in) :: n
    real(kind=8), intent(in), device :: A(n*n)
    real(kind=8), intent(in), device :: x(n)
    real(kind=8), intent(out), device :: y(n)
    type(dim3) :: blocks, threads
    integer :: shmem ! shared memory in bytes
    
    blocks = dim3(2*80,1,1)
    threads = dim3(512,1,1)
    shmem = 0

    call matvec_kernelv1<<<blocks,threads,shmem,0 >>>(A,x,y,n)
  end subroutine matvec_cudav1
    
  attributes(global) subroutine matvec_kernelv1(A,x,y,n)
    implicit none
    integer(mINT), value, intent(in) :: n
    real(kind=8), intent(in), device :: A(n*n)
    real(kind=8), intent(in), device :: x(n)
    real(kind=8), intent(out), device :: y(n)
    
    ! local variables
    integer :: i, j
    
    ! simple parallel version of matrix vector multiplication. 
    ! y = A*x
    do j = threadIdx%x + blockDim%x*(blockIdx%x-1), n, gridDim%x*blockDim%x ! rows
       y(j) = 0.0d+0
       do i = 1, n ! columns
          y(j) = y(j) + A(i + (j-1)*n) * x(i)
       enddo
    enddo    

  end subroutine matvec_kernelv1



  subroutine matvec_cudav2(A,x,y,n)
    implicit none
    integer(mINT), intent(in) :: n
    real(kind=8), intent(in), device :: A(n*n)
    real(kind=8), intent(in), device :: x(n)
    real(kind=8), intent(out), device :: y(n)
    type(dim3) :: blocks, threads
    integer :: shmem ! shared memory in bytes
    integer :: xthreads

    xthreads = 32    

    blocks = dim3(n/xthreads,1,1) ! TODO fix perfect division assumption
    threads = dim3(xthreads,1,1)
    shmem = (xthreads*32 + xthreads + 32)*8
    print *, "v2 shmem : ", shmem
    call matvec_kernelv2<<<blocks,threads,shmem,0 >>>(A,x,y,n)
  end subroutine matvec_cudav2




  attributes(global) subroutine matvec_kernelv2(A,x,y,n)
    implicit none
    integer(mINT), value, intent(in) :: n
    real(kind=8), intent(in), device :: A(n*n)
    real(kind=8), intent(in), device :: x(n)
    real(kind=8), intent(out), device :: y(n)
    
    ! shared memory (local to blocks)
    integer, parameter :: rrmax = 32
    real(kind=8), shared :: A_s(blockDim%x,rrmax)
    real(kind=8), shared :: x_s(blockDim%x)
    real(kind=8), shared :: y_s(rrmax)
    

    ! local variables
    integer :: i, j
    integer :: rr, q


    do q = threadIdx%x, n, blockDim%x
       x_s(threadIdx%x) = x(q)
       do rr=1, rrmax
          A_s(threadIdx%x, rr) = A(q + (rr-1)*n + n*rrmax*(blockIdx%x-1))
       enddo

       call syncthreads() ! shared memory fence

       do rr=threadIdx%x, rrmax, blockDim%x
          y_s(rr) = 0.0d+0
          do i = 1, blockDim%x ! columns
             y_s(rr) = y_s(rr) + A_s(i,rr) * x_s(i)
          enddo          
          call syncthreads()
          ! accumulate the shared memory y_s to global y
          y(rr + rrmax*(blockIdx%x-1)) = y( rr + rrmax*(blockIdx%x-1)  ) + y_s(rr)
       enddo
    enddo
       
  end subroutine matvec_kernelv2






end module matvec
