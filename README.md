# Entanglement Phase Transition Simulator

A Julia project for simulating entanglement phase transitions in quantum spin chains using Matrix Product States (MPS) and random unitary circuits.

## Overview

This project implements numerical simulations to study quantum entanglement dynamics in non-Hermitian systems and under measurements. Key features include:

- Time evolution of MPS under random unitary gates and non-Hermitian operations
- Entropy calculations (von Neumann, Renyi) for MPS bipartitions
- Scaling analysis of entanglement entropy
- Data collapse analysis for phase transitions
- Visualization tools for entropy evolution and distributions

## Dependencies

- `Julia 1.x`
- `ITensors.jl`: For MPS operations
- `ITensorMPS.jl`: For MPS-specific functionality
- `HDF5.jl`: For data storage
- `Plots.jl`: For visualization
- `LaTeXStrings.jl`: For plot labels
- `Statistics.jl`: For statistical analysis

## Key Components

- [`time_evolution.jl`](time_evolution.jl): Core evolution routines for MPS
- [`entanglement_entropies.jl`](entanglement_entropies.jl): Entropy calculation functions
- [`mytebd.jl`](mytebd.jl): Time-Evolving Block Decimation implementation
- [`entropy_calc.jl`](entropy_calc.jl): Statistical sampling of entropy measures
- [`time_evolve_plot.jl`](time_evolve_plot.jl): Visualization of time evolution results

## Usage

1. Ensure Julia and required packages are installed
2. Run simulations:
   ```julia
   julia> include("time_evolve_calc.jl")  # Generate evolution data
   julia> include("time_evolve_plot.jl")  # Plot results
   ```

3. For scaling analysis:
   ```julia
   julia> include("entropy_scale.jl")
   julia> include("entropy_plot.jl")
   ```

## Data Storage
Simulation results are stored in HDF5 files:
- `time_evolve_data.h5`: Time evolution data
- `entropy_scale_data.h5`: Scaling analysis data
- `critical_params.h5`: Phase transition parameters

## Examples
See `time_evolve_plot.jl` for example usage and parameter settings.

## License
MIT License: 