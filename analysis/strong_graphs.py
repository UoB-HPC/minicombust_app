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
kernels=["interpolate_nodal_data", "particle_interpolation_data", "solve_spray_equations", "update_particle_positions", "emitted_particles", "updated_flow_field", "minicombust"]

# Get kernel dataframes
strong_dfs  = {}
for k in kernels:
    strong_dfs[k] = pd.read_csv("results/tx2-strong-%dcells_modifier-%dparticles-%s.log" % (int(sys.argv[2]), int(sys.argv[1]), k), delim_whitespace=True)
    strong_dfs[k].set_index("cores", inplace=True)


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
    
    ax  = strong_field_df[-1].plot.line(title="strong scaling: " + field, style='.-')
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.set_ylabel(field)
    plt.suptitle("%dparticles-%dcells_modifier" % (int(sys.argv[1]), int(sys.argv[2])), y=1.05, fontsize=18)
    plt.savefig(("results/tx2-strong-%dparticles-%dcells_modifier-%s.pdf") % (int(sys.argv[1]), int(sys.argv[2]), str(field)), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
