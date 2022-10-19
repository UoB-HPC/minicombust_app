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
weak_particles_dfs  = {}
for k in kernels:
    weak_particles_dfs[k] = pd.read_csv("results/tx2-weak_mesh-%dnodes-%dppn-%dparticles-%s.log" % (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), k), delim_whitespace=True)
    weak_particles_dfs[k].set_index("cells", inplace=True)


# Reorganise dataframes into fields
# plot kernels against each other, for each field
fields = weak_particles_dfs["minicombust"].columns
weak_particles_field_df = []
for field in fields:
    print("Creating " + field + " graph")

    weak_particles_field_df.append(weak_particles_dfs["minicombust"].copy())
    for k in kernels:
        weak_particles_field_df[-1].insert(1, k, weak_particles_dfs[k][field])
    weak_particles_field_df[-1].drop(columns=fields, inplace=True)
    
    ax  = weak_particles_field_df[-1].plot.line(title="weak mesh scaling: " + field, style='.-')
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.set_ylabel(field)
    plt.suptitle("%dnodes-%dppn-%dparticles" % (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])), y=1.05, fontsize=18)
    plt.savefig(("results/tx2-weak_mesh-%dnodes-%dppn-%dparticles-%s.pdf") % (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), str(field)), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
