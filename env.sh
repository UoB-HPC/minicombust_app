#module swap perftools-base/21.05.0 papi
#emodule load intel/compiler/64/2020/19.1.3
#module swap craype-broadwell craype-x86-cascadelake
module swap PrgEnv-cray/6.0.9 PrgEnv-gnu
module use /lustre/projects/bristol/modules-arm-phase2/modulefiles/
module swap gcc/9.3.0 gcc/12.1.0

export GCC_VERSION=12.1.0
export GNU_VERSION=12.1.0

#source /cm/shared/apps/intel/compilers_and_libraries/2020.4.304/mpi/intel64/bin/mpivars.sh
