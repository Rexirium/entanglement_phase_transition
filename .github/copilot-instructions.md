# Copilot Instructions for entanglement_phase_transition

## Project Overview
This Julia project simulates entanglement phase transitions in quantum spin chains using Matrix Product States (MPS) and random unitary circuits. The codebase is organized around time evolution, entropy calculation, and plotting routines.

## Key Components
- `time_evolution.jl`: Core logic for evolving MPS under random unitary and non-Hermitian operations. Defines `mps_evolve` and `entropy_evolve`.
- `entropies.jl`: Implements entropy calculations (von Neumann, Renyi, zeroth order) for MPS bipartitions.
- `mytebd.jl`: Contains TEBD (Time-Evolving Block Decimation) routines and gate construction for spin chains.
- `time_evolve_plot.jl`: Example script for running entropy evolution and plotting results.

## Data Flow
- Simulations start from an initial MPS (`psi0`), evolved over time using `entropy_evolve` (see `time_evolution.jl`).
- Entropy is computed at each step using functions from `entropies.jl`.
- Results are aggregated and can be visualized (see `time_evolve_plot.jl`).

## Developer Workflows
- **Run a simulation:** Edit and execute `time_evolve_plot.jl` or use the `let` block in `time_evolution.jl` for quick tests.
- **Dependencies:** Requires Julia, ITensors.jl, ITensorMPS.jl, and Plots.jl. Install via Julia's package manager.
- **No automated build/test scripts** are present; run scripts directly in Julia REPL or via `julia <script>.jl`.

## Project Conventions
- All MPS operations use ITensors.jl idioms (e.g., `siteinds`, `MPS`, `apply`, `op`).
- Entropy functions expect MPS and bipartition index as arguments.
- Randomness is used for circuit generation; results may vary between runs.
- Parameters (system size, time, probabilities) are set at the top of scripts for easy experimentation.

## Examples
- See `time_evolve_plot.jl` for parameter sweep and plotting usage.
- See the `let` block in `time_evolution.jl` for a minimal working example.

## Tips for AI Agents
- When adding new simulations, follow the pattern in `time_evolve_plot.jl`.
- For new entropy measures, extend `entropies.jl` and update `entropy_evolve` if needed.
- Keep all code Julia 1.x compatible and prefer ITensors idioms for tensor operations.

## External Integration
- No external data or APIs; all computation is self-contained.
- Plots are generated using Plots.jl and can be saved or displayed interactively.

---
For questions, see code comments or refer to ITensors.jl documentation for advanced usage.
