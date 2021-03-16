module matvec


contains
  
  subroutine matvec_serial(A,x,y,n)
    integer, intent(in) :: n
    real(kind=8), intent(in) :: A(n*n)
    real(kind=8), intent(in) :: x(n)
    real(kind=8), intent(out) :: y(n)
    
    ! Serial version of matrix vector multiplication. 
    ! y = A*x
    do j = 1 , n ! rows
       do i = 1, n ! columns
          y(j) = A(i + (j-1)*n) * x(j)
       enddo
    enddo

  end subroutine matvec_serial


end module matvec
