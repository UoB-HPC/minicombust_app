# miniCOMBUST

## Dependencies
- Catch++ Unit Testing (Header-only)


## Build

```
make clean all
```

## Run 
```
./bin/minicombust
```

## Run tests 
```
./bin/minicombust_tests
```

## Unsupported Feature List
- Visualise particle fields (temperature, mass, evaporated gas)
- Spray Breakup (spawn new particles, primary and secondary breakup)
- C API for particle side
- Tetrahedral mesh
- Benchmarks
- Serial Optimised Implementation 
- MPI Implementation

## References
- (Parallel load-balancing for combustion with spray for large-scale simulation)[https://www.sciencedirect.com/science/article/pii/S0021999121000826?via%3Dihub#br0160]
- (Particle breakup)
  [https://d1wqtxts1xzle7.cloudfront.net/30787458/GetTRDoc-libre.pdf?1392132662=&response-content-disposition=inline%3B+filename%3DAn_experimental_and_computational_invest.pdf&Expires=1649411271&Signature=UjaWeJ468xnrPetPPHzfeTmawxW-0i7yUb~39pAPXPzfpQ4AD2fdyijRPQ39KNj3UkIcxx4MoWysHGcaebDvLQTKvqye89ibtGhwkOxXAiZyfJZ1H2nPZGIWOdCqe6X15D4KGOGLYglb0o0SeuccQsh6p~BNFNh1WAiEoqsYOf6aQhc2rl0hNO8s5lqYBodlGFjVFEaiNqkZu8t3U3AQ0TXu~kk10TmB1asHH69oyR5K9cTNcnarHlKAcnzd6BbIMJJFqE2nPYHTDHtOrELC-eRAhWqbJxyFNKfAMVUEldIn9cV5GH~x7YYfKWzsBcI2tMwm1CuMsu9GHHptueA6jw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA#page=166]
- (Particle evaporation)[https://arc.aiaa.org/doi/10.2514/3.8264]
- (Particle breakup)[https://www.sciencedirect.com/science/article/pii/S0301932203001113?via=ihub]
- (Particle breakup parceling)[https://www.sciencedirect.com/science/article/pii/S1540748908002678#bb0085]
