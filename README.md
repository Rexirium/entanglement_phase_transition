# Entanglement Phase Transition Simulator

A Julia project for simulating entanglement dynamics and phase transitions in random-unitary/measurement circuits using Matrix Product States (MPS). The project supports data generation, finite-size scaling, and visualization of results.

---

## Features

- **Time Evolution**: Simulate MPS dynamics under random two-site unitaries and weak measurements.
- **Entropy Calculations**: Compute Renyi and von Neumann entropies for single samples or ensembles.
- **Finite-Size Scaling**: Perform scaling analysis to extract critical exponents.
- **Phase Diagrams**: Generate and visualize phase diagrams for entanglement transitions.
- **Data Storage**: Store results in HDF5 format for efficient analysis and plotting.

---

## Project Structure

- **Core Code**:
  - [`src/time_evolution.jl`](src/time_evolution.jl): Implements MPS time evolution.
  - [`src/entropy_calc.jl`](src/entropy_calc.jl): Entropy calculation routines.
  - [`mytebd.jl`](mytebd.jl): TEBD utilities and benchmarks.

- **One-Parameter Experiments**:
  - [`oneparam/entropy_data.jl`](oneparam/entropy_data.jl): Data generation.
  - [`oneparam/entropy_plot.jl`](oneparam/entropy_plot.jl): Plotting results.
  - [`oneparam/finite_size_scaling.jl`](oneparam/finite_size_scaling.jl): Scaling analysis.

- **Two-Parameter Experiments**:
  - [`twoparams/data_generator.jl`](twoparams/data_generator.jl): Data generation.
  - [`twoparams/phase_diagram.jl`](twoparams/phase_diagram.jl): Phase diagram generation.
  - [`twoparams/scaling_demo_prob.jl`](twoparams/scaling_demo_prob.jl): Scaling analysis.

- **Time Evolution**:
  - [`time_evolve/time_evolve_calc.jl`](time_evolve/time_evolve_calc.jl): Time evolution driver.
  - [`time_evolve/time_evolve_plot.jl`](time_evolve/time_evolve_plot.jl): Plotting time evolution results.

- **Testing**:
  - [`test/calc_test.jl`](test/calc_test.jl): Unit tests for entropy calculations.

- **Data**:
  - HDF5 files stored in the [`data/`](data/) directory.

---

## Requirements

- **Julia**: Version 1.x
- **Dependencies**:
  - ITensors.jl, ITensorMPS.jl, HDF5.jl, Plots.jl, LaTeXStrings.jl
  - Interpolations.jl, FiniteSizeScaling.jl
- **Recommended**: MKL for optimized performance.

---

## Quick Start

1. **Install Dependencies**:
   ```sh
   julia -e 'using Pkg; Pkg.instantiate()'
   ```

2. **Run Experiments**:
   - One-parameter experiment:
     ```julia
     include("oneparam/entropy_data.jl")
     ```
   - Two-parameter experiment:
     ```julia
     include("twoparams/data_generator.jl")
     ```
   - Time evolution:
     ```julia
     include("time_evolve/time_evolve_calc.jl")
     ```

3. **Analyze Results**:
   - Plot entropy data:
     ```julia
     include("oneparam/entropy_plot.jl")
     ```
   - Perform finite-size scaling:
     ```julia
     include("oneparam/finite_size_scaling.jl")
     ```
   - Generate phase diagrams:
     ```julia
     include("twoparams/phase_diagram.jl")
     ```

---

## Data Layout

- **Directory**: All generated data is stored in the [`data/`](data/) directory.
- **Examples**:
  - [`data/time_evolve_data.h5`](data/time_evolve_data.h5)
  - [`data/oneparam_L8_2_18_21x21.h5`](data/oneparam_L8_2_18_21x21.h5)
  - [`data/critical_params_11x11.h5`](data/critical_params_11x11.h5)

---

## Example: Finite-Size Scaling

The scaling variable is defined as:

$$
x = L^{1/\nu} (p - p_c)
$$

Quantities are plotted as $S_1(p) - S_1(p_c)$ vs. $x$. See:
- [`oneparam/finite_size_scaling.jl`](oneparam/finite_size_scaling.jl)
- [`twoparams/scaling_demo_prob.jl`](twoparams/scaling_demo_prob.jl)

---

## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for details.

---

## References

- **Core Evolution**: [`src/time_evolution.jl`](src/time_evolution.jl)
- **Entropy Calculations**: [`src/entropy_calc.jl`](src/entropy_calc.jl)
- **TEBD Utilities**: [`mytebd.jl`](mytebd.jl)
- **Data Generation**:
  - One-parameter: [`oneparam/entropy_data.jl`](oneparam/entropy_data.jl)
  - Two-parameter: [`twoparams/data_generator.jl`](twoparams/data_generator.jl)
- **Plotting**:
  - Entropy: [`oneparam/entropy_plot.jl`](oneparam/entropy_plot.jl)
  - Time Evolution: [`time_evolve/time_evolve_plot.jl`](time_evolve/time_evolve_plot.jl)

