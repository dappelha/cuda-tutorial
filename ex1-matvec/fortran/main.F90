program main
  use matvec
  use nvtx_mod
  implicit none

  integer(mINT) :: n
  real(kind=8), allocatable, pinned :: A(:), x(:), y(:)
  real(kind=8), allocatable, device :: A_d(:), x_d(:), y_d(:)
  integer :: istat
  CHARACTER(len=32) :: arg
  
  ! get n from as a runtime arguement
  
  call get_command_argument(1, VALUE=arg, STATUS=istat)
  if (istat==0) then 
     read(arg,*), n
  else
     n = 20*1024
  endif

  print *, "n = ", n

  allocate( A(n*n), x(n), y(n) )

  A(:) = 1.0d+0
  x(:) = 1.0d+0

  print *, "hello from main"

  call nvtxStartRange("serial", color=1)
  call matvec_serial(A,x,y,n)
  call nvtxEndRange()

  ! if A and x are 1, y is n.
  !write(*,'(F8.4)') y(:)
  
  ! Make device versions of arrays and populate with same data as x and A
  !  allocate( A_d(n), x_d(n), y_d(n) )
  allocate( A_d, SOURCE=A)
  allocate( x_d, SOURCE=x)
  allocate( y_d, MOLD=y)
  

  ! Do niave v1 with blocks/threads over loop over rows.
  call matvec_cudaV1(A_d,x_d,y_d,n)

  y_d = y
  write(*,'(F14.4)') y(2)

  ! Do shared memory batched loading of A and x.
  call matvec_cudaV2(A_d,x_d,y_d,n)

  y_d = y
  write(*,'(F14.4)') y(2)


  return


end program main
