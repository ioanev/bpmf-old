ROOT=../../../c++

include ../Makefile.common
export ARGO_INSTALL_DIRECTORY=${HOME}/argodsm

CXX=mpicxx
CXXFLAGS+=-DBPMF_ARGO_COMM -DBPMF_OMP_SCHED -DBPMF_PROFILING
CXXFLAGS+=-L${ARGO_INSTALL_DIRECTORY}/build/lib -I${ARGO_INSTALL_DIRECTORY}/src

# for openmp
CXXFLAGS+=-fopenmp

# for argo
LDFLAGS+=-largo -largobackend-mpi

# plus argo
LDFLAGS+=-lnuma -lrt

