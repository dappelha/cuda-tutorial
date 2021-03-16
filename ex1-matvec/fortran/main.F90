program main
  use matvec
  use nvtx_mod
  implicit none
  integer :: n
  real(kind=8), allocatable :: A(:), x(:), y(:)

  n = 10
  allocate( A(n*n), x(n), y(n) )


  print *, "hello from main"

  call matvec_serial(A,x,y,n)

  write(*,'(F8.4)') y(:)

  return


end program main
