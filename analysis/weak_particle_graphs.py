#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

kernels=["interpolate_nodal_data", "particle_interpolation_data", "solve_spray_equations", "update_particle_positions", "emitted_particles", "updated_flow_field", "minicombust"]

weak_particles_dfs  = {}
for k in kernels:
    weak_particles_dfs[k] = pd.read_csv("out/tx2-weak_particles-%dnodes-%dppn-%dcells_modifier-%s.log" % (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), k), delim_whitespace=True)
    weak_particles_dfs[k].set_index("particles", inplace=True)

fields = weak_particles_dfs["minicombust"].columns
weak_particles_field_df = []
for field in fields:
    print("Creating " + field + " graph")
    weak_particles_field_df.append(weak_particles_dfs["minicombust"].copy())
    for k in kernels:
        weak_particles_field_df[-1].insert(1, k, weak_particles_dfs[k][field])
    weak_particles_field_df[-1].drop(columns=fields, inplace=True)
    
    print(weak_particles_field_df[-1])

    ax = weak_particles_field_df[-1].plot.line(title="weak particle scaling " + field)
    plt.savefig(("results/tx2-weak_particles-%dnodes-%dppn-%dcells_modifier-%s.pdf") % (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), str(field)))
    plt.clf()
