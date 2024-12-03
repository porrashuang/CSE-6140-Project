# CSE-6140-Project README

## Overview

This program solves the Traveling Salesperson Problem (TSP) using different algorithms. A wrapper script is used to monitor execution time externally, providing cleaner and more extendable implementation without embedding timeout handling directly into the program.

## Usage

If you are using non-macOS like Linux, please see the compile instruction. For MacOS users, you can run the program using the following command:

```bash
cd code
./run_program -inst <fileName> -alg [BF|Approx|LS] -time <cut_off_in_sec> [-seed <random_seed>]
```
### Parameters

- **`-inst <fileName>`**: Path to the input file containing the TSP instance.
- **`-alg [BF|Approx|LS]`**: Algorithm to use for solving:
  - `BF`: Brute Force
  - `Approx`: Approximation
  - `LS`: Local Search
- **`-time <cut_off_in_sec>`**: Execution cutoff time in seconds.
- **`-seed <random_seed>`** (optional): Random seed for reproducibility.

### Example

```bash
cd code
./run_program.sh -inst ../DATA/Atlanta.tsp -alg LS -time 60 -seed 42
```
## Build Instructions

To compile the program, use the following command:

```bash
cd code
g++ main.cpp -o tsp_solver
```
## Caveats

1. **Wrapper Script Behavior**:
   - The wrapper script will not terminate automatically if the `tsp_solver` program finishes earlier than the specified timeout.
   - In such cases, you can manually terminate the wrapper script using `Ctrl-C`.

2. **Runtime Discrepancy**:
   - The runtime displayed in the program output might differ from the runtime shown in the solution.
   - This discrepancy is due to the inclusion of file-saving time in the overall runtime evaluation.

## Notes

The wrapper script ensures that the timeout is handled externally, making the program cleaner and easier to extend for future enhancements.

