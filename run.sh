# Generate input matrix
g++ ./gen1.cc -o gen1
g++ ./gen2.cc -o gen2 -lblas

./gen1 5 >input5
./gen2 5 >my5

# Householder Serial and Parallel
g++ ./householder_serial.cc -o householder_serial -O0 -lblas
./householder_serial -file input5

mpic++ ./householder_parallel.cc -o householder_parallel -lblas
mpiexec -n 4 ./householder_parallel -file input5

# Givens Serial and Parallel
g++ ./givens_serial.cc -o givens_serial -O0 -lblas
./givens_serial -file input5

mpic++ ./givens_parallel.cc -o givens_parallel -lblas
mpiexec -n 4 ./givens_parallel -file input5

# QR Iteration to get eigenvalues
mpic++ ./eig_givens.cc -o eig_givens -lblas
mpiexec -n 4 ./eig_givens -file my5 -silent

g++ ./eig_givens_hessenberg.cc -o eig_givens_hessenberg -O0 -lblas
./eig_givens_hessenberg -file my5 -silent

g++ ./eig_givens_hessenberg_shift.cc -o eig_givens_hessenberg_shift -O0 -lblas
./eig_givens_hessenberg_shift -file my5 -silent

#-silent is an option