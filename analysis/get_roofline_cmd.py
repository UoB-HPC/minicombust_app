#!/usr/bin/python

import sys

options = ["CASCADE_LAKE", "TX2"]

if (len(sys.argv) < 3):
    print("No argument supplied. Run './get_roofline_cmd.py ARCH CONFIG RANKS' or './get_roofline_cmd TIMES_ONLY RANKS'")
    print("ARCH options are: " + str(options))
    exit()

command = ""
point_string = ""
counters         = []
kernel_names     = []
times            = []
kernel_counters  = {}

for proc in range(int(sys.argv[-1])):
    perf_file = open("out/performance_rank" + str(proc) +".csv", 'r')

    read_header = False
    for line in perf_file:
        if (not read_header):
            counters          = line.split(",")[1:]
            counters[-1]      = counters[-1][:-1]
            read_header       = True
            continue

        values                       = line.split(",")[1:]
        kernel_name                  = line.split(",")[0]
        if (proc == 0):
            kernel_names.append(kernel_name)
            kernel_counters[kernel_name] = {}
            for i in range(len(values)):
                kernel_counters[kernel_name][counters[i]] = [float(values[i])]
        else:
            for i in range(len(values)):
                kernel_counters[kernel_name][counters[i]].append(float(values[i]))

print("%40s %15s %15s %15s %15s" % ("kernel", "real_time", "total_time_%", "performance", "mem_bandwidth"))

total_time = max(kernel_counters["minicombust"]["time"])

if (len(counters) != 1):
    if (sys.argv[1] == "CASCADE_LAKE"):
        for kernel in kernel_names:
                
                kernel_vals = kernel_counters[kernel]

                kernel_vals["time"]         = max(kernel_vals["time"])
                kernel_vals["PAPI_DP_OPS"]  = sum(kernel_vals["PAPI_DP_OPS"])
                kernel_vals["PAPI_LST_INS"] = sum(kernel_vals["PAPI_LST_INS"])

                cacheline=8
                performance = kernel_vals["PAPI_DP_OPS"] / kernel_vals["time"]
                performance /= 1000000000
                OI = kernel_vals["PAPI_DP_OPS"] / (cacheline*kernel_vals["PAPI_LST_INS"])
                point_string += " --point " + str(OI) + "x" + str(performance) + " --pointname " + kernel

                mem_bandwidth  = cacheline*kernel_vals["PAPI_LST_INS"] / kernel_vals["time"]
                mem_bandwidth /= 1000000000

                times.append(kernel_vals["time"])

                print("%40s %15.2f %15.2f %15.2f %15.2f " % (kernel, kernel_vals["time"], (kernel_vals["time"] / total_time) * 100., performance, mem_bandwidth))


        print("\n\npython roofline.py procs/cascade-lake-6230-"+sys.argv[2]+".yaml --cacheaware " + point_string)
    elif (sys.argv[1] == "TX2"):
        for kernel in kernel_names:
                    
                    kernel_vals = kernel_counters[kernel]

                    kernel_vals["time"]        = max(kernel_vals["time"])
                    kernel_vals["PAPI_FP_INS"] = sum(kernel_vals["PAPI_FP_INS"])
                    kernel_vals["PAPI_LD_INS"] = sum(kernel_vals["PAPI_LD_INS"])
                    kernel_vals["PAPI_SR_INS"] = sum(kernel_vals["PAPI_SR_INS"])

                    cacheline=8
                    performance = kernel_vals["PAPI_FP_INS"] / kernel_vals["time"]
                    performance /= 1000000000
                    OI = kernel_vals["PAPI_FP_INS"] / (cacheline*(kernel_vals["PAPI_LD_INS"] + kernel_vals["PAPI_SR_INS"]))
                    point_string += " --point " + str(OI) + "x" + str(performance) + " --pointname " + kernel

                    mem_bandwidth  = cacheline*(kernel_vals["PAPI_LD_INS"] + kernel_vals["PAPI_SR_INS"]) / kernel_vals["time"]
                    mem_bandwidth /= 1000000000

                    times.append(kernel_vals["time"])

                    print("%40s %15.2f %15.2f %15.2f %15.2f " % (kernel, kernel_vals["time"], (kernel_vals["time"] / total_time) * 100., performance, mem_bandwidth))

        print("\n\npython roofline.py procs/thunderx2-isambard-"+sys.argv[2]+".yaml --cacheaware " + point_string)
else:
    for kernel in kernel_names:
            kernel_vals = kernel_counters[kernel]
            
            times.append(kernel_vals["time"])
            print(kernel + ": time " + str(kernel_counters[kernel]["time"]))
