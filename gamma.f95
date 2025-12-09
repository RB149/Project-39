Program gamma
  implicit none
  !define variables
  !parameter(s)
  integer, parameter  :: dp=selected_real_kind(15,300)

  !defing var for gamma = [ Esurf - N_surf*E_bulk/atom ] /[2*A]
  !E_bulk/atom = Bulk_Energy = Optimised bulk energy
  !Esurf = Optimised surface energy
  !N_surf = # of atoms in surface
  !A = area = length^2
  real (kind=dp) :: Esurf, N_surf, Bulk_Energy, A, gamma_
  real (kind=dp) :: Jconversion = 16.021766

  !requesting variables
  print*, "Welcome to the gamma solver program!"
  print*, "please input the varibles for the equation (Optimised surface energy, # of &
  &atoms in surface, Optimised bulk energy, length) below. in listed order with spaces between them" 
  !reading space delimited data
  read(*,*) Esurf, N_surf, Bulk_Energy, A

  !maths
  A=A**2
  gamma_ = ( Esurf - N_surf*Bulk_Energy ) /(2.0_dp*A)
  print*, "Gamma (eV/Angstrom^2):", gamma_
  gamma_ = gamma_*Jconversion
  print*, "Gamma (J/m^2):", gamma_
End program