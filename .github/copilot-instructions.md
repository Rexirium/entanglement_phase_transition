# Workspace Instructions: Entanglement Phase Transition Simulator

This Julia project simulates entanglement dynamics and phase transitions in random-unitary/measurement circuits using Matrix Product States (MPS). Below are key patterns and conventions for working effectively on this codebase.

---

## Project Structure & Modules

### Core Simulation (`src/`)
- **`time_evolution.jl`**: MPS time evolution under random two-site unitaries and measurements. Handles TEBD updates and bond dimension management.
- **`entanglement.jl`**: Entropy calculations (Renyi, von Neumann) via Schmidt decomposition. Core function: compute entropy at each cut.
- **`simulation.jl`**: Simulation driver; orchestrates time evolution + entropy measurement for single samples or ensembles.
- **`observers.jl`**: Observer pattern implementation for tracking observables during evolution (e.g., entanglement growth).
- **`correlation.jl`**: Correlation analysis utilities (if needed for phase boundary characterization).

### Data Generation (`data_generation/`)
- Scripts use **distributed computing** via `SlurmClusterManager` (HPC parallelization).
- Typical workflow: define parameter grids (probability `p`, parameter `η`, system size `L`) → run multi-sample calculations → average results → store in HDF5 with metadata.
- **Pattern**: `entrcorr_averager.jl` and `entropy_generator.jl` generate raw data; `*_averager.jl` scripts aggregate across samples.
- **Input**: Command-line argument for number of samples (e.g., `julia script.jl 100`).
- **Output**: HDF5 files in `data/` with structure: datasets named by parameters, attributes storing metadata (ps, ηs, Ls, sample count).

### Visualization & Analysis (`phase_separate/`, `twoparams/`, `visualization/`, `time_evolve/`)
- **Phase diagrams**: `phase_diagram.jl`, `phase_plot_large.jl` read HDF5 data, compute scaling indices via linear regression (`linear_regress.jl`), plot with error bars.
- **Finite-size scaling**: `scaling_demo_*.jl` scripts extract critical exponents; use `Interpolations.jl` for smooth interpolation.
- **Plotting**: Uses `Plots.jl` (CairoMakie backend configured in Project.toml). Results saved as PNG/PDF.
- **Data flow**: HDF5 → read with `h5read()` → compute observables (finite-size scaling, correlations) → plot with error bars.

### Testing (`test/`)
- Ad-hoc test scripts (no Test.jl framework). Examples: `calc_test.jl`, `entropy_evolve_test.jl`.
- Pattern: write sample parameters → run simulation → manually verify results are physically reasonable.
- **No CI/CD** — tests are developer-run during development.

### Data Storage (`data/`)
- All results stored as **HDF5** files (binary, efficient for large arrays).
- Naming convention: `<observable>_<descriptor>_<system_sizes>.h5` (e.g., `entropy_data_L4_2_18_21x21.h5`).
- Structure: HDF5 datasets keyed by parameter values; attributes store metadata.

---

## Execution Model & Common Tasks

### Running Simulations
```julia
# Data generation (distributed, HPC typically):
# julia data_generation/entropy_generator.jl 100

# Single-sample tests or quick verification:
# julia test/calc_test.jl

# Visualization (reads existing HDF5 data):
# julia phase_separate/phase_plot_large.jl
```

### Environment Setup
- Run `Pkg.instantiate()` in Julia REPL to restore dependencies from `Project.toml`.
- Key dependencies: **ITensors.jl**, **ITensorMPS.jl**, **HDF5.jl**, **CairoMakie.jl**, **Interpolations.jl**.
- **MKL** is configured for BLAS/threading optimization (see Project.toml).

### Parameter Conventions
- **`p`** or **`prob`**: Measurement probability (0 ≤ p ≤ 1). Controls entanglement phase transition.
- **`η`** or **`eta`**: Measurement strength or correlator parameter. Related to phase boundary location.
- **`L`** or **`sys_size`**: System size (number of qubits/sites). Typically 4–21 in this project.
- **`N_samples`**: Number of independent trajectories/realizations averaged per parameter point.

### HDF5 Data Layout
- Files read/written with functions like `h5read(filename, "/dataset/path")` and `h5write()`.
- Typical structure: group per observable (e.g., `/entropy`, `/correlations`) → datasets with keys like `"p_0.1_eta_0.5"` → value is array of measurements.
- **Always store metadata**: parameter arrays, system sizes, averaging count as HDF5 attributes.

---

## Code Style & Conventions

### Julia Idioms
- **Type stability** is important for performance; avoid type instability in hot loops.
- **Dispatch pattern** common: different implementations for different data types (e.g., MPS state vs. density matrix).
- **In-place operations** (`mul!`, `ldiv!`, etc.) used in tight loops for memory efficiency.
- **Comments**: Use `#` for inline; docstrings for public functions (triple-quoted `"""..."""`).

### Naming
- Functions: `snake_case` (e.g., `compute_entropy`, `evolve_mps`).
- Constants: UPPER_CASE (e.g., `MAX_BOND_DIM`, `CUTOFF`).
- Modules: CamelCase (e.g., `TimeEvolution`, `Entanglement`).

### File Structure
- One main concept per file (e.g., `entanglement.jl` for entropy functions).
- Utility functions grouped in `*.jl` files; no deep nesting of directories.
- Top-level scripts (e.g., `phase_plot.jl`) are self-contained and include all setup via `include()` or explicit imports.

---

## Common Development Tasks

### Adding a New Observable
1. Define computation function in `src/simulation.jl` or a new module (e.g., `src/observables.jl`).
2. Update `observers.jl` if using the observer pattern.
3. Extend data generation script (e.g., `data_generation/entropy_generator.jl`) to compute and store the new observable.
4. Create a corresponding visualization script in `phase_separate/` or `twoparams/`.
5. Test with a small parameter grid before scaling to full ensemble.

### Debugging Numerics
- Use `calc_test.jl` as a template: set small system size (L=4), few samples (N=1), and manually verify intermediate results.
- Print intermediate arrays with `println()` or use the debugger (`Infiltrator.jl` if available).
- Check for **NaN/Inf** values and bond dimension growth (`maxlinkdim(ψ)`).

### Profiling Performance
- Use `@time` macro to measure individual runs.
- For detailed profiling, use `@profiler` or `Profile.jl`.
- Focus on hot loops: MPS contractions, entropy calculations, and distributed communication.
- Leverage **MKL** threading via `BLAS.set_num_threads()` for multi-threaded BLAS operations.

### Managing Large Datasets
- Use HDF5 datasets with chunking (`chunk=(1000,)` in `h5write()`) for efficient I/O.
- Store only summary statistics (mean, std) for memory efficiency; store all samples only if needed.
- Use `h5ls()` to inspect HDF5 file structure before loading large files.

---

## Common Pitfalls

1. **Bond dimension explosion**: If MPS bond dimension grows unbounded, entropy will diverge. Check `maxlinkdim(ψ)` regularly and set cutoff appropriately.
2. **Mismatched parameter grids**: Ensure data generation and visualization scripts use the same parameter ranges; inconsistent grids lead to plotting errors.
3. **HDF5 metadata loss**: Always store parameters as attributes or datasets alongside data. Don't rely on filenames alone.
4. **Floating-point precision**: Entropy values near critical point can be sensitive to numerical errors. Use `Float64` consistently.
5. **Distributed communication overhead**: For small systems or very fast computations, distributed overhead may dominate. Test serial vs. parallel scaling.

---

## Useful Resources

- **ITensors.jl documentation**: [https://itensor.org/](https://itensor.org/) — MPS and tensor network operations.
- **HDF5.jl documentation**: [https://github.com/JuliaIO/HDF5.jl](https://github.com/JuliaIO/HDF5.jl) — file I/O.
- **CairoMakie.jl**: Backend for high-quality plots.
- **Julia Performance Tips**: [https://docs.julialang.org/en/v1/manual/performance-tips/](https://docs.julialang.org/en/v1/manual/performance-tips/) — type stability, avoiding globals, and in-place operations.

---

## Quick Start for New Tasks

1. **Understanding a file**: Start by reading the docstring of key functions (if present), then trace through examples in `test/`.
2. **Running experiments**: Start with a small test case (few samples, small system size) to verify correctness before scaling up.
3. **Modifying data generation**: Edit parameter grids in the script → test with serial execution first → then scale to HPC.
4. **Debugging**: Use REPL with `include()` to load modules interactively, then step through calculations manually.
