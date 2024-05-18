#include "utils/utils.hpp"
#include "amgx_c.h"
#include "cuda_runtime.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

namespace minicombust::flow 
{
    class MemoryTracker 
    {
        private:

        public:
            unordered_map<void*, std::pair<string, size_t>> host_pointers;
            unordered_map<void*, std::pair<string, size_t>> device_pointers;
            vector<void*> host_vector;
            vector<void*> device_vector;

            MPI_Config *mpi_config;

            MemoryTracker(MPI_Config *mpi_config) : mpi_config(mpi_config)
            { }

            void *allocate_host(const std::string& name, size_t size)
            {
                void *ptr = (void*)malloc(size);

                host_pointers[ptr] = std::make_pair(name, size);
                host_vector.push_back(ptr);

                return ptr;
            }

            void *allocate_host(const std::string& name, size_t size, void *ptr_id)
            {
                if (host_pointers.contains(ptr_id))
                {
                    void *ptr = (void*)malloc(size);
                    
                    host_pointers[ptr_id].second = host_pointers[ptr_id].second + size;
                    return ptr;
                }
                else
                {
                    void *ptr = (void*)malloc(size);

                    host_pointers[ptr_id] = std::make_pair(name, size);
                    host_vector.push_back(ptr_id);

                    return ptr;
                }
            }

            void *allocate_cuda_host(const std::string& name, void **ptr, size_t size)
            {
                gpuErrchk(cudaMallocHost(ptr, size));

                host_pointers[*ptr] = std::make_pair(name, size);
                host_vector.push_back(*ptr);

                return ptr;
            }

            void *allocate_cuda_host(const std::string& name, void **ptr, size_t size, void *ptr_id)
            {
                if (host_pointers.contains(ptr_id))
                {
                    gpuErrchk(cudaMallocHost(ptr, size));
                    
                    host_pointers[ptr_id].second = host_pointers[ptr_id].second + size;
                    return ptr;
                }
                else
                {
                    gpuErrchk(cudaMallocHost(ptr, size));

                    host_pointers[ptr_id] = std::make_pair(name, size);
                    host_vector.push_back(ptr_id);

                    return ptr;
                }
            }

            // If ptr key given, accumulate array sizes. Note: resize functionality will not work.

            void *allocate_device(const std::string& name, void **ptr, size_t size)
            {
                // printf("Allocating %s with size %f\n", name.c_str(), ((float)size)/1.e9);

				gpuErrchk(cudaMalloc(ptr, size));

                device_pointers[*ptr] = std::make_pair(name, size);
                device_vector.push_back(*ptr);

                return ptr;
            }

            void *allocate_device(const std::string& name, void **ptr, size_t size, void *ptr_id)
            {
                // printf("Allocating %s with size %f\n", name.c_str(), ((float)size)/1.e9);
                if (device_pointers.contains(ptr_id))
                {
                    gpuErrchk(cudaMalloc(ptr, size));
                    
                    device_pointers[ptr_id].second = device_pointers[ptr_id].second + size;
                    return ptr;
                }
                else
                {
                    gpuErrchk(cudaMalloc(ptr, size));

                    device_pointers[ptr_id] = std::make_pair(name, size);
                    device_vector.push_back(ptr_id);

                    return ptr;
                }
            }

            void print_usage()
            {
                size_t local_size = 0;
                size_t global_size = 0;

                if (mpi_config->particle_flow_rank == 0)
                    printf("Flow Host Memory Usage:\n");

                for (void *ptr : host_vector)
                {
                    const auto val_pair = host_pointers[ptr];

                    size_t world_size;
                    MPI_Allreduce(&val_pair.second, &world_size, 1, MPI_UINT64_T, MPI_SUM, mpi_config->particle_flow_world);
                    if (mpi_config->particle_flow_rank == 0)
                        printf("\t%40s        (TOTAL %8.2f GB) (AVG %8.2f GB) \n", val_pair.first.c_str(), (float) world_size / 1.e9, (float) world_size / (1.e9 * (float)mpi_config->particle_flow_world_size));
                    
                    local_size  += val_pair.second;
                    global_size += world_size;
                }
                if (mpi_config->particle_flow_rank == 0)
                    printf("\t%40s        (TOTAL %8.2f GB) (AVG %8.2f GB) \n", "FLOW TOTAL HOST USAGE", (float) global_size / 1.e9, (float) global_size / (1.e9 * (float)mpi_config->particle_flow_world_size)  );
            

                local_size = 0;
                global_size = 0;

                if (mpi_config->particle_flow_rank == 0)
                    printf("Flow Device Memory Usage:\n");

                for (void *ptr : device_vector)
                {
                    const auto val_pair = device_pointers[ptr];

                    size_t world_size;
                    MPI_Allreduce(&val_pair.second, &world_size, 1, MPI_UINT64_T, MPI_SUM, mpi_config->particle_flow_world);
                    if (mpi_config->particle_flow_rank == 0)
                        printf("\t%40s (TOTAL %8.2f GB) (AVG %8.2f GB) \n", val_pair.first.c_str(), (float) world_size / 1.e9, (float) world_size / (1.e9 * (float)mpi_config->particle_flow_world_size));
                    
                    local_size  += val_pair.second;
                    global_size += world_size;
                }
                if (mpi_config->particle_flow_rank == 0)
                    printf("\t%40s (TOTAL %8.2f GB) (AVG %8.2f GB) \n\n", "FLOW TOTAL DEVICE USAGE",  (float) global_size / 1.e9, (float) global_size / (1.e9 * (float)mpi_config->particle_flow_world_size)  );
            
            
            }


    };
}