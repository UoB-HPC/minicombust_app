export LD_LIBRARY_PATH=/lustre/fsw/coreai_devtech_all/hwaugh/repos/AMGX_vector_upload/install/lib/:$LD_LIBRARY_PATH
make CUDA_INSTALL_PATH=/usr/local/cuda/ AMGX_INSTALL_PATH=/lustre/fsw/coreai_devtech_all/hwaugh/repos//AMGX_vector_upload/install/ MPI_INSTALL_PATH=/usr/local/openmpi/ clean gpu
#make CUDA_INSTALL_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/cuda/ AMGX_INSTALL_PATH=/lustre/fsw/coreai_devtech_all/hwaugh/repos//AMGX/install/ MPI_INSTALL_PATH=/lustre/fsw/coreai_devtech_all/hwaugh/repos/mpi/ clean gpu
