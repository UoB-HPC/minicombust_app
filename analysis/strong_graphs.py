#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

# Declare kernels
kernels=["particle_interpolation_data", "solve_spray_equations", "update_particle_positions", "emitted_particles", "updated_flow_field", "minicombust"]

# Get kernel dataframes
strong_dfs  = {}
for k in kernels:
    strong_dfs[k] = pd.read_csv("results/tx2-strong-%dcells_modifier-%dparticles-%s.log" % (int(sys.argv[2]), int(sys.argv[1]), k), delim_whitespace=True)
    strong_dfs[k].set_index("cores", inplace=True)

strong_dfs["updated_flow_field"] = strong_dfs["minicombust"] - (strong_dfs["particle_interpolation_data"] + strong_dfs["solve_spray_equations"] + strong_dfs["update_particle_positions"] + strong_dfs["emitted_particles"])

print(strong_dfs["updated_flow_field"])
print(strong_dfs["minicombust"])

for k in kernels:
    print(k)
    df_copy = strong_dfs[k].sort_index()
    first = df_copy.head(1).copy()
    
    temp_df = pd.DataFrame(first.to_numpy() / df_copy.to_numpy(), index=df_copy.index)
    temp_df.rename({0: 'speedup', 1: 'total_time', 2: 'performance', 3: 'mem_bandwidth'}, axis=1, inplace=True)
    print(temp_df)

    temp_df['Parallel Efficiency'] = 100 * temp_df['speedup'] / (temp_df.index.to_series() / temp_df.first_valid_index() )
    print(temp_df)

    ax  = temp_df['Parallel Efficiency'].iloc[0:int(sys.argv[3])].plot.line(title="MiniCombust Parallel Efficiency : %d PPI, %d cells" % (int(sys.argv[1]), 2*int(sys.argv[2])**3) , style='.-')
    plt.savefig("results/efficiency-%s.pdf" % (str(k)), bbox_inches='tight')
    plt.clf()

    
# Reorganise dataframes into fields
# plot kernels against each other, for each field
fields = strong_dfs["minicombust"].columns
strong_field_df = []
for field in fields:
    print("Creating " + field + " graph")

    strong_field_df.append(strong_dfs["minicombust"].copy())
    for k in kernels:
        strong_field_df[-1].insert(1, k, strong_dfs[k][field])
    strong_field_df[-1].drop(columns=fields, inplace=True)
    strong_field_df[-1].sort_index(inplace=True)
    
    # strong_dfs[k] = strong_dfs[k].divide(strong_dfs[k].head())


    ax  = strong_field_df[-1].iloc[0:int(sys.argv[3])].plot.line(title="Strong scaling : %d PPI, %d cells" % (int(sys.argv[1]), 2*int(sys.argv[2])**3) , style='.-')

    perfect_start = strong_dfs["minicombust"].first_valid_index()
    perfect_end   = strong_dfs["minicombust"].last_valid_index()
    perfect_cores = np.linspace(perfect_start, perfect_end, 200)
    ax.plot(perfect_cores, [strong_dfs["minicombust"]["real_time"].iat[0] / (x/perfect_start) for x in perfect_cores], 'k--', label="Perfect scaling", alpha=0.5)
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.set_ylabel(field)
    plt.suptitle("%dparticles-%dcells_modifier" % (int(sys.argv[1]), int(sys.argv[2])), y=1.05, fontsize=18)
    plt.savefig(("results/tx2-strong-%dparticles-%dcells_modifier-%s.pdf") % (int(sys.argv[1]), int(sys.argv[2]), str(field)), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
