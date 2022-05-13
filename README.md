# miniCOMBUST

## Dependencies
- Catch++ Unit Testing (Header-only)

## Build

Tested with GCC, Cray and Intel compilers, Intel is most tested compiler at the moment.

Without PAPI:
```
make clean all
```

With PAPI:
```
PAPI=1 make clean all
```

## Run 
```
./bin/minicombust # emits 10 particles per timestep by default
```

```
./bin/minicombust NUM_PARTICLES_PER_TIMESTEP
```

## Run tests 
```
./bin/minicombust_tests
```

## Output

Output vtk files for the mesh and particles are written to `out/`

## Get roofline CMD (PAPI Build Required)
```
python analysis/get_roofline_cmd.py CASCADE_LAKE 1-core out/performance.csv
```

## Future Features
- Primary breakup
- C API for particle side
- Tetrahedral mesh
- Benchmarks
- MPI Implementationi
- YAML config file

## References
- (Parallel load-balancing for combustion with spray for large-scale simulation)[https://www.sciencedirect.com/science/article/pii/S0021999121000826?via%3Dihub#br0160]
- (Particle breakup)
  [https://d1wqtxts1xzle7.cloudfront.net/30787458/GetTRDoc-libre.pdf?1392132662=&response-content-disposition=inline%3B+filename%3DAn_experimental_and_computational_invest.pdf&Expires=1649411271&Signature=UjaWeJ468xnrPetPPHzfeTmawxW-0i7yUb~39pAPXPzfpQ4AD2fdyijRPQ39KNj3UkIcxx4MoWysHGcaebDvLQTKvqye89ibtGhwkOxXAiZyfJZ1H2nPZGIWOdCqe6X15D4KGOGLYglb0o0SeuccQsh6p~BNFNh1WAiEoqsYOf6aQhc2rl0hNO8s5lqYBodlGFjVFEaiNqkZu8t3U3AQ0TXu~kk10TmB1asHH69oyR5K9cTNcnarHlKAcnzd6BbIMJJFqE2nPYHTDHtOrELC-eRAhWqbJxyFNKfAMVUEldIn9cV5GH~x7YYfKWzsBcI2tMwm1CuMsu9GHHptueA6jw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA#page=166]
- (Particle evaporation)[https://arc.aiaa.org/doi/10.2514/3.8264]
- (Particle breakup)[https://www.sciencedirect.com/science/article/pii/S0301932203001113?via=ihub]
- (Particle breakup parceling)[https://www.sciencedirect.com/science/article/pii/S1540748908002678#bb0085]
