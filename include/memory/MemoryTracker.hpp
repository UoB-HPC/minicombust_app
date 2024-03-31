#pragma once

#include "utils/utils.hpp"

#include <vector>

namespace minicombust::memory 
{
    class MemoryTracker 
    {
        private:


            int mpi_rank;
            int mpi_root;
            int mpi_size;
            int mpi_world;

            string tracking_name;

            vector<uint64_t> array_sizes;
            vector<string>   array_names;


        public: 

            MemoryTracker ( int mpi_rank, int mpi_root, int mpi_size, MPI_Comm mpi_world, string tracking_name ) : mpi_rank(mpi_rank), mpi_root(mpi_root), mpi_size(mpi_size), mpi_world(mpi_world), tracking_name(tracking_name)
            {

            }

            void track_field ( string array_name, uint64_t array_size )
            {
                uint64_t total_size = array_size;
                
                if (mpi_root == mpi_rank)
                {
                    MPI_Reduce(MPI_IN_PLACE, &total_size, 1, MPI_UINT64_T, MPI_SUM, mpi_root, mpi_world);

                    array_sizes.push_back(total_size);
                    array_names.push_back(array_name);
                }
                else
                {
                    MPI_Reduce(&total_size, nullptr, 1, MPI_UINT64_T, MPI_SUM, mpi_root, mpi_world);
                }
            }

            void print_memory_requirents (  )
            {
                if (mpi_root == mpi_rank)
                {
                    uint64_t total_size = 0;
                    cout << tracking_name << " storage requirements (" << mpi_size << " processes) :" << endl;
                    for (uint64_t i = 0; i < array_sizes.size(); i++)
                    {
                        total_size += array_sizes[i];
                        printf("\t%-50s size (TOTAL %12.2f MB) (AVG %8.2f MB) \n", array_names[i].c_str(), (float) array_sizes[i] / 1000000.0, (float) array_sizes[i] / (1000000.0 * mpi_size));
                    }

                    printf("\t%-50s size (TOTAL %12.2f MB) (AVG %8.2f MB) \n\n", tracking_name.c_str(), (float)total_size / 1000000.0,  (float)total_size / (1000000.0 * mpi_size));
                }
            }
    };;
}