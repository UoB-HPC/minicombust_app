# MiniCOMBUST Kernel Performance

## Kernels
- Interpolate nodal data
- Particle interpolation data
- Solve spray equations
- Update particle positions
- Emitted particles
- Update flow field 

### Emitted particles 
Code location: `ParticleDistribution.hpp : emit_particles (..)`
Operational intensity: 0.15 (FLOPS per byte)
Fraction of runtime: <1% (Oct 2022)

Kernel responsible for creating N new particles per iteration. Stores new particles at the end of a reserved vector.Each particle finds cell in mesh, from random start position - estimate from last particle.

### Update flow field 
Code location: `ParticleSolver.inl && FlowSolver.inl : update_flow_field ()`
Operational intensity : 0.00 (FLOPS per byte) 
Fraction of runtime: 50% - 80% (Oct 2022)

Kernel responsible for getting updated flow fields from flow solver.

Algorithm:
1. (FLOW && PARTICLE) MPI_Gather:  Gather the size of each rank's cell array to flow rank.
2. (FLOW && PARTICLE) MPI_Gatherv: Gather the cells array from each particle rank.
3. (FLOW && PARTICLE) MPI_Gatherv: Gather the cells particle fields data from each particle rank.
4. (FLOW) Get the neighbours of each cell sent from the particle ranks.
5. (FLOW && PARTICLE) MPI_Bcast:   Send the size of neighbours to all ranks.
6. (FLOW && PARTICLE) MPI_Bcast:   Send the neighbour cells to all ranks.
7. (FLOW && PARTICLE) MPI_Bcast:   Send the neighbour cells to all ranks.
8. (FLOW && PARTICLE) MPI_Bcast:   Broadcast neighbour flow terms to all ranks.


### Interpolate nodal data
Code location: `ParticleSolver.inl : interpolate_nodal_data ()`
Operational intensity : 0.03 (FLOPS per byte)
Fraction of runtime: 10% - 40% (Oct 2022)

Kernel interpolates field values from the cell centres to the cell nodes. 

Algorithm:
1. Iterate over each cell in the neighbours set:
    - Iterate over each node in the cell:
        - Add flow gradient to each node flow value.
        - Increment node centre count.
        - Mark node.
2. Iterate over the marked nodes:
    - If at boundary, add boundary flow values.
    - Divide node flow values by cell centre count.

### Particle interpolation data
Code location: `ParticleSolver.inl : solve_spray_equations ()` (first half)
Operational intensity: 0.2 (FLOPS per byte)
Fraction of runtime: <5% (Oct 2022)

Kernel interpolates nodal field values to particle position field values.

Algorithm:
1. Iterate over particles vector:
    - Iterate over each node in the particle's cell:
        - Weight node field value by distance and get weighted average.

### Solve spray equations
Code location: `Particle.hpp : solve_spray ()`
Operational intensity: 0.16 (FLOPS per byte)
Fraction of runtime:  <5% (Oct 2022)

Kernel solves drag, evaporation and breakup equations for each particle.

### Update particle positions
Code location: `ParticleSolver.inl : update_particle_positions ()` 
Operational intensity: 0.15 (FLOPS per byte)
Fraction of runtime:  <5% (Oct 2022)

Kernel iterates over particles, and updates the cell based on the position from `solve_spray()`. If the particle hasn't decayed, increment the bounding cell's flow field.



