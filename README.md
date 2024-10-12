# miniCOMBUST

## Dependencies
- Catch++ Unit Testing (Header-only)
- PETSc:
  - Set `PETSC_INSTALL_PATH` to the locations of the PETSc folder containing the `include` and `lib` folders.
- AmgX

## MiniCombust Job Generation Script

### Overview

This bash script, `.gen_job.sh`, generates and manages jobs for the MiniCombust miniapp. It supports SLURM job generation, container-based execution, and various configuration options for running the application.

### Usage

```bash
./.gen_job.sh [--build] [--nodes <nnodes>] [--mpi_procs <mpi_procs_per_node>]  [--gpus <ngpus>] 
              [--container <path>] [--results_name <name>] [--job_template <file>] 
              [--walltime <time>] [--enroot] [--nsys] [--ncu] [--prof_ranks <ranks>] 
              [--interactive_run] [--interactive]

```

### Key Features
 - Configurable node, GPU, and MPI process settings
 - Container support (Enroot)
 - Interactive and batch job modes
 - Profiling options (Nsight Systems and NVIDIA Compute Profiler)
 - Customizable job templates

### Common Options
 - `--build`: build minicombust instead of running
 - `--nodes <number>`: Set number of nodes
 - `--mpi_procs <number>`: Set MPI processes per node
 - `--gpus <number>`: Set number of GPUs
 - `--container <img/path>`: Specify container img or path (sqshfs)
 - `--results_name <name>`: Set results directory name, this is appended to `results/CURRENT_DATE`
 - `--job_template <file>`: Specify job template file
 - `--walltime <time>`: Set job walltime
 - `--enroot`: Enable Enroot containerization, default is srun.
 - `--nsys`: Enable Nsight Systems profiling
 - `--ncu`: Enable NVIDIA Compute Profiler
 - `--prof_ranks <ranks>`: Set ranks to profile
 - `--interactive`: Run interactively within the container
 - `--jump_into_container`: Jump into container, don't run minicombust.

Default options are set in `defaults.sh`

### Examples

Build with enroot:
```bash
./gen_job.sh --interactive --enroot --build --results_name build_enroot/ --
```

Run with slurm interactively:
```bash
./gen_job.sh --results_name build_enroot/ --
```

Create batch slurm batch job:
```bash
./gen_job.sh --nodes 2 --mpi_procs 112 --gpus 8 --job_template job_templates/eos.job --results_name 2node_112proc_8gpu --
```

### Output

Results are stored in `results/CURRENT_DATE/RESULTS_NAME`.

The script generates:
A log file with configuration details
A job file based on the specified template
Both are saved in the designated results directory.

### Job Submission
For non-interactive runs, the script automatically submits the generated job to SLURM using sbatch.


## References
- (Parallel load-balancing for combustion with spray for large-scale simulation)[https://www.sciencedirect.com/science/article/pii/S0021999121000826?via%3Dihub#br0160]
- (Particle breakup)
  [https://d1wqtxts1xzle7.cloudfront.net/30787458/GetTRDoc-libre.pdf?1392132662=&response-content-disposition=inline%3B+filename%3DAn_experimental_and_computational_invest.pdf&Expires=1649411271&Signature=UjaWeJ468xnrPetPPHzfeTmawxW-0i7yUb~39pAPXPzfpQ4AD2fdyijRPQ39KNj3UkIcxx4MoWysHGcaebDvLQTKvqye89ibtGhwkOxXAiZyfJZ1H2nPZGIWOdCqe6X15D4KGOGLYglb0o0SeuccQsh6p~BNFNh1WAiEoqsYOf6aQhc2rl0hNO8s5lqYBodlGFjVFEaiNqkZu8t3U3AQ0TXu~kk10TmB1asHH69oyR5K9cTNcnarHlKAcnzd6BbIMJJFqE2nPYHTDHtOrELC-eRAhWqbJxyFNKfAMVUEldIn9cV5GH~x7YYfKWzsBcI2tMwm1CuMsu9GHHptueA6jw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA#page=166]
- (Particle evaporation)[https://arc.aiaa.org/doi/10.2514/3.8264]
- (Particle breakup)[https://www.sciencedirect.com/science/article/pii/S0301932203001113?via=ihub]
- (Particle breakup parceling)[https://www.sciencedirect.com/science/article/pii/S1540748908002678#bb0085]
