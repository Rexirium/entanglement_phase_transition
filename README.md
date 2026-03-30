
# Entanglement Phase Transition Simulator

This Julia project simulates entanglement dynamics and phase transitions in random-unitary/measurement circuits using Matrix Product States (MPS). It supports data generation, finite-size scaling, correlation analysis, and visualization of results for quantum many-body systems.

---

## Features

- **Time Evolution**: Simulate MPS dynamics under random two-site unitaries and weak measurements.
- **Entropy Calculations**: Compute Renyi and von Neumann entropies for single samples or ensembles.
- **Finite-Size Scaling**: Perform scaling analysis to extract critical exponents.
- **Phase Diagrams**: Generate and visualize phase diagrams for entanglement transitions.
- **Data Storage**: Store results in HDF5 format for efficient analysis and plotting.

---


## Project Structure

- **src/**: Core simulation and analysis code
  - `RandomUnitary.jl`: Main module, exports all core functions and types
  - `time_evolution.jl`, `simulation.jl`, `entanglement.jl`, `correlation.jl`, `operators.jl`, `observers.jl`, `time_correlation.jl`: Algorithms for time evolution, entropy, correlations, operators, and observers

- **data_generation/**: Scripts for generating entanglement and correlation data
  - `entrcorr_averager.jl`, `entrcorr_averager2.jl`, `entrcorr_generator.jl`, `entrcorr_timecorr2.jl`, `entropy_generator.jl`

- **datacollapse/**: Data collapse and phase diagram analysis
  - `data_collapse.jl`, `phase_diagram.jl`

- **phase_separate/**: Phase diagram and phase separation plotting
  - `phase_plot.jl`, `phase_plot_large.jl`, `phase_plot_large2.jl`

- **correlation_analysis/**: Time correlation analysis
  - `timecorr.jl`

- **time_evolve/**: Time evolution drivers and plotting
  - `time_evolve_calc.jl`, `time_evolve_plot.jl`

- **visualization/**: Visualization utilities
  - `linesearch.jl`

- **test/**: Unit and integration tests
  - `calc_test.jl`, `average_test.jl`, `check_randomness.jl`, `entropy_evolve_test.jl`, `timecorr_test.jl`

- **data/**: HDF5 files and generated data

---


## Requirements

- **Julia**: Version 1.8 or later
- **Dependencies** (see Project.toml):
  - ITensors.jl, ITensorMPS.jl, HDF5.jl, Plots.jl, LaTeXStrings.jl
  - Interpolations.jl, FiniteSizeScaling.jl
- **Recommended**: MKL.jl for optimized performance

---


## Quick Start

1. **Install dependencies**:
   ```sh
   julia --project -e 'using Pkg; Pkg.instantiate()'
   ```

2. **Generate data** (examples):
   ```julia
   include("data_generation/entropy_generator.jl")
   include("data_generation/entrcorr_generator.jl")
   ```

3. **Analyze and plot**:
   ```julia
   include("datacollapse/data_collapse.jl")
   include("datacollapse/phase_diagram.jl")
   include("phase_separate/phase_plot.jl")
   include("time_evolve/time_evolve_plot.jl")
   ```

---

## API Usage Examples

### Basic Setup and MPS Creation

```julia
using ITensors, ITensorMPS
include("src/RandomUnitary.jl")
using .RandomUnitary

# Create MPS for a 10-qubit system
L = 10
ss = siteinds("S=1/2", L)
psi = MPS(ComplexF64, ss, "Up")  # Initialize in |Up⟩ state
```

### Entropy and Entanglement Measures

```julia
# Compute Renyi entropy at bond position b
L = 10
b = L ÷ 2  # Bond between sites L÷2 and L÷2+1
S1 = ent_entropy(psi, b, 1)  # Von Neumann entropy (n=1)
S2 = ent_entropy(psi, b, 2)  # 2nd order Renyi entropy (n=2)
S0 = zeroth_entropy(psi, b)  # Zeroth order entropy

# Compute entanglement negativity and concurrence
neg = negativity(psi, b)
conc = concurrence(psi, b)

# Compute mutual information between regions
As = [1, 2]  # Sites in region A
Bs = [L-1, L]  # Sites in region B
MI = mutual_information(psi, As, Bs, 1)
```

### Correlation Functions

```julia
# Compute correlation between two specific sites
corr_ij = correlation(psi, "Sz", "Sz", 2, 8)  # ⟨Sz_2 Sz_8⟩

# Compute correlation vector at all distances (symmetric w.r.t center)
corr_vec = correlation_vec(psi, "Sx", "Sx")  # Returns vector of length L
```

### Time Evolution with Non-Hermitian Disentangler

```julia
# Define disentangler parameters: probability p and strength eta
p = 0.5  # Probability of applying non-Hermitian operator
η = 0.5  # Strength parameter
dent = NHDisentangler{Float64}(p, η)

# Evolve MPS for ttotal time steps
ttotal = 4 * L
psi = MPS(ComplexF64, ss, "Up")
truncerr = timeevolve!(psi, ttotal, dent; cutoff=1e-14, maxdim=4096)

# Re-initialize and evolve with observer to track entropy
psi = MPS(ComplexF64, ss, "Up")
obs = EntropyObserver{Float64}(b; n=1)  # Track entropy at bond b
timeevolve!(psi, ttotal, dent, obs; cutoff=1e-14, maxdim=4096)

# Access tracked observables
entropies = obs.entropies  # Vector of entropies over time
truncerrs = obs.truncerrs  # Truncation errors at each step
```

### Tracking Correlation Functions During Evolution

```julia
# Create observer to track both entropy and correlations
obs = EntrCorrObserver{Float64}(b, L; n=1, op="Sz")
psi = MPS(ComplexF64, ss, "Up")
timeevolve!(psi, ttotal, dent, obs; cutoff=1e-14, maxdim=4096)

# Access results
entropies = obs.entrs  # Time-dependent entropies
correlations = obs.corrs  # Time-dependent correlation vectors
```

### Ensemble Calculations

```julia
# Compute ensemble average over N samples
N = 100
res = EntropyResults{Float64}(b; n=1, nsamp=N)

# Single-threaded calculation
calculation_mean(L, ttotal, dent, res; cutoff=1e-14)

# Multi-threaded calculation (recommended for large ensembles)
res_mt = EntropyResults{Float64}(b; n=1, nsamp=N)
calculation_mean_multi(L, ttotal, dent, res_mt; cutoff=1e-14)

# Access ensemble results
entr_mean = mean(res.entropies)
entr_std = std(res.entropies)
```

### Ensemble Calculations with Correlations

```julia
# Track both entropy and correlations in ensemble
res_corr = EntrCorrResults{Float64}(b, L; n=1, op="Sz", nsamp=N)
calculation_mean_multi(L, ttotal, dent, res_corr; cutoff=1e-14)

# Access results
entr_mean = mean(res_corr.entropies)
corr_mean = mean(res_corr.corrs, dims=2)  # Average correlations
```

### Time Correlation Functions

```julia
# Compute ⟨ops1_i(t) ops2_j(0)⟩ for varying times
tstart = 2*L  # Thermalization time
ttotal = 4*L  # Evolution time
ops = ("Sz", 3, "Sz", L-2)  # ⟨Sz_3(t) Sz_{L-2}(0)⟩

psi = MPS(ComplexF64, ss, "Up")
timecorrs, truncerr = timecorrelation!(psi, ttotal, tstart, dent, ops; cutoff=1e-14)

# timecorrs contains correlation values for times tstart+1 to ttotal
```

### Post-Evolution Analysis

```julia
# After evolution, analyze final state properties
S_final = ent_entropy(psi, b, 1)
corr_final = correlation_vec(psi, "Sz", "Sz")
neg_final = negativity(psi, b)

println("Final Von Neumann entropy: $S_final")
println("Final max correlation: $(maximum(abs.(corr_final)))")
println("Final negativity: $neg_final")
```

---

## Data Layout

- All generated data is stored in the [`data/`](data/) directory.
- Example files:
  - `critical_params_11x11.h5`, `time_evolve_data.h5`, `entr_corr_data_L4_2_18_21x21.h5`, etc.

---


## Example: Finite-Size Scaling

The scaling variable is defined as:

$$
x = L^{1/\nu} (p - p_c)
$$

Quantities are typically plotted as $S_1(p) - S_1(p_c)$ vs. $x$.
See the scripts in `datacollapse/` for examples.

---


## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for details.

---


## References

- **Core simulation**: `src/`
- **Data generation**: `data_generation/`
- **Data analysis and collapse**: `datacollapse/`
- **Phase diagram plotting**: `phase_separate/`
- **Correlation analysis**: `correlation_analysis/`
- **Time evolution**: `time_evolve/`
- **Visualization utilities**: `visualization/`
- **Tests**: `test/`

