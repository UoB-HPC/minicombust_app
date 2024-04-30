export LD_LIBRARY_PATH=/lustre/fsw/coreai_devtech_all/hwaugh/repos/AMGX/install/lib/:$LD_LIBRARY_PATH
make CUDA_INSTALL_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/cuda/ AMGX_INSTALL_PATH=/lustre/fsw/coreai_devtech_all/hwaugh/repos//AMGX/install/ MPI_INSTALL_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/  clean gpu
#make CUDA_INSTALL_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/cuda/ AMGX_INSTALL_PATH=/lustre/fsw/coreai_devtech_all/hwaugh/repos//AMGX/install/ MPI_INSTALL_PATH=/lustre/fsw/coreai_devtech_all/hwaugh/repos/mpi/ clean gpu
