#pragma once

#ifdef PAPI

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <ctime>
#include <papi.h>

#include "utils/utils.hpp"

using namespace minicombust::utils; 

namespace minicombust::performance 
{
    using namespace std;

    template<class T>
    class PerformanceLogger 
    {
        private:

            int event_set, num_events, *events;

            int128_t *temp_count_store;


            string papi_config_file = "PAPI.CONFIG";
        
        public:

            int128_t *position_kernel_event_counts;
            int128_t *spray_kernel_event_counts;
            int128_t *interpolation_kernel_event_counts;
            int128_t *particle_interpolation_event_counts;
            int128_t *emit_event_counts;

            double position_ticks = 0.f;
            double interpolation_ticks = 0.f;
            double particle_interpolation_ticks = 0.f;
            double spray_ticks = 0.f;
            double emit_ticks = 0.f;
            clock_t output; 
            
            vector<string> event_names;

            inline void print_counters()
            {
                ofstream myfile;
                myfile.open("out/performance.csv");
                myfile << "kernel,time";
                for (int e = 0; e < num_events; e++)  myfile << "," << event_names[e];
                myfile << endl;


                myfile << "interpolate_nodal_data," << interpolation_ticks /  CLOCKS_PER_SEC;
                for (int e = 0; e < num_events; e++)    myfile << "," << interpolation_kernel_event_counts[e];
                myfile << endl;

                myfile << "particle_interpolation_data," << particle_interpolation_ticks /  CLOCKS_PER_SEC;
                for (int e = 0; e < num_events; e++)    myfile << "," << particle_interpolation_event_counts[e];
                myfile << endl;


                myfile << "solve_spray_equations,"  << spray_ticks /  CLOCKS_PER_SEC;
                for (int e = 0; e < num_events; e++)    myfile << "," << spray_kernel_event_counts[e];
                myfile << endl;

                myfile << "update_particle_positions," << position_ticks /  CLOCKS_PER_SEC;
                for (int e = 0; e < num_events; e++)    myfile << "," << position_kernel_event_counts[e];
                myfile << endl;

                myfile << "emitted_particles," << emit_ticks /  CLOCKS_PER_SEC;
                for (int e = 0; e < num_events; e++)    myfile << "," << emit_event_counts[e];
                myfile << endl;
                myfile.close();
            }


            inline void my_papi_start()
            {
                if (event_set != PAPI_NULL)
                {
                    for (int e = 0; e < num_events; e++) 
                    {
                        temp_count_store[e] = 0;
                    }
                    if (PAPI_start(event_set) != PAPI_OK) 
                    {
                        printf("ERROR: Failed to start PAPI\n");
                        exit(EXIT_FAILURE);
                    }
                }
                output = clock(); 

            }

            inline void my_papi_stop(int128_t *kernel_event_counts, double *ticks)
            {
                if (event_set != PAPI_NULL) 
                {
                    if (PAPI_stop(event_set, temp_count_store) != PAPI_OK) 
                    {
                        printf("ERROR: Failed to stop PAPI\n");
                        exit(EXIT_FAILURE);
                    }
                    for (int e = 0; e < num_events; e++) 
                    {
                        kernel_event_counts[e] += temp_count_store[e];
                    }
                }
                *ticks += double(clock() - output);
            }

            inline void init_papi()
            {
                int ret;

                if ((ret=PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT) {
                    printf("ERROR: PAPI_library_init() failed: '%s'.\n", PAPI_strerror(ret));
                    exit(EXIT_FAILURE);
                }

                int num_ctrs = 0;
                int num_comps = PAPI_num_components();
                for (int c=0; c<num_comps; c++) 
                {
                    num_ctrs += PAPI_num_cmp_hwctrs(c);
                }
                if (num_ctrs < 2) 
                {
                    printf("ERROR: No hardware counters here, or PAPI not supported (num_ctrs=%d)\n", num_ctrs);
                    exit(-1);
                }
            }

            inline void load_papi_events()
            {
                int ret;

                event_set = PAPI_NULL;
                ret = PAPI_create_eventset(&event_set);
                if (ret != PAPI_OK || event_set == PAPI_NULL)
                {
                    printf("ERROR: PAPI_create_eventset() failed: '%s'.\n", PAPI_strerror(ret));
                    exit(EXIT_FAILURE);
                }

                char event_name[512];
                std::string line;
                std::ifstream file_if(papi_config_file);
                if(!file_if.is_open())
                {
                    printf("WARNING: Failed to open PAPI config: '%s'\n", papi_config_file.c_str());
                    event_set = PAPI_NULL;
                } 
                else 
                { 
                    while(std::getline(file_if, line))
                    {
                        if (line.c_str()[0] == '#' || strcmp(line.c_str(), "") == 0) 
                        {
                            continue;
                        }

                        strcpy(event_name, line.c_str());

                        ret = PAPI_OK - 1;
                        if (ret != PAPI_OK) 
                        {
                            int code = -1;
                            ret = PAPI_event_name_to_code(event_name, &code);
                            event_names.push_back(string(event_name));
                            cout << event_names.back() << " " << std::hex << code << std::dec << endl;
                            if (ret != PAPI_OK)
                            {
                                printf("Could not convert string '%s' to PAPI event, error = %s\n", event_name, PAPI_strerror(ret));
                            } 
                            else 
                            {
                                if (PAPI_query_event(code) != PAPI_OK) 
                                {
                                    printf("PAPI event %s not present\n", event_name);
                                }
                                else
                                {
                                    ret = PAPI_add_event(event_set, code);
                                    if (ret != PAPI_OK) 
                                    {
                                        printf("ERROR: Failed to add event %d to event set: '%s'.\n", code, PAPI_strerror(ret));
                                        if (event_set == PAPI_NULL)
                                        {
                                            printf("ERROR: ... and event_set=PAPI_NULL\n");
                                        }
                                        exit(EXIT_FAILURE);
                                    }
                                }
                            }
                        }
                    }
                }


                if (file_if.bad()) 
                {
                    printf("ERROR: Failed to read papi conf file: %s\n", papi_config_file.c_str());
                    exit(EXIT_FAILURE);
                }

                num_events = PAPI_num_events(event_set);
                cout << "Monitoring " << num_events << " PAPI events.." << endl;
                if (num_events == 0) 
                {
                    event_set = PAPI_NULL;
                }

                events = (int*)malloc(sizeof(int)*num_events);
                int* temp_thread_events = (int*)malloc(sizeof(int)*num_events);
                if (PAPI_list_events(event_set, temp_thread_events, &num_events) != PAPI_OK) 
                {
                    printf("ERROR: PAPI_list_events() failed\n");
                }

                for (int i = 0; i < num_events; i++) 
                {
                    events[i] = temp_thread_events[i];
                }
                free(temp_thread_events);

                spray_kernel_event_counts = (int128_t*)malloc(sizeof(int128_t)*num_events);
                for (int i = 0; i < num_events; i++) 
                {
                    spray_kernel_event_counts[i] = 0;
                }

                position_kernel_event_counts = (int128_t*)malloc(sizeof(int128_t)*num_events);
                for (int i = 0; i < num_events; i++) 
                {
                    position_kernel_event_counts[i] = 0;
                }

                interpolation_kernel_event_counts = (int128_t*)malloc(sizeof(int128_t)*num_events);
                for (int i = 0; i < num_events; i++) 
                {
                    interpolation_kernel_event_counts[i] = 0;
                }

                particle_interpolation_event_counts = (int128_t*)malloc(sizeof(int128_t)*num_events);
                for (int i = 0; i < num_events; i++) 
                {
                    particle_interpolation_event_counts[i] = 0;
                }

                emit_event_counts = (int128_t*)malloc(sizeof(int128_t)*num_events);
                for (int i = 0; i < num_events; i++) 
                {
                    emit_event_counts[i] = 0;
                }


                temp_count_store = (int128_t*)malloc(sizeof(int128_t)*num_events);
                for (int e = 0; e < num_events; e++)
                {
                    temp_count_store[e] = 0;
                }

                if (event_set != PAPI_NULL) 
                {
                    if (PAPI_start(event_set) != PAPI_OK) 
                    {
                        printf("ERROR: Failed to start PAPI\n");
                        exit(EXIT_FAILURE);
                    }
                    if (PAPI_stop(event_set, temp_count_store) != PAPI_OK) 
                    {
                        printf("ERROR: Failed to stop PAPI\n" );
                        exit(EXIT_FAILURE);
                    }
                    for (int e=0; e<num_events; e++)
                    {
                        temp_count_store[e] = 0;
                    }
                }
            }   


    }; // class PerformanceLogger

}   // namespace minicombust::performance 

#endif
