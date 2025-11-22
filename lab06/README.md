# Lab 06: Hopfield Networks for Error Correction and Combinatorial Optimization

## Overview

This lab implements three key applications of Hopfield networks:
1. **Error Correction**: Associative memory with Hebbian storage
2. **Eight-Rook Problem**: Constraint satisfaction problem solving
3. **Traveling Salesman Problem**: Combinatorial optimization for 10 cities

## Lab Report

Based on the report: **"Hopfield Networks for Error Correction and Combinatorial Optimization"** by Kush Sonawane, Ram Sharma, and Ronak Vaghela.

## Files

### Main Implementation Files

- **`hopfield_error_correction.py`** - Problem 1: Error correction and associative memory
  - Implements binary Hopfield network with Hebbian learning
  - Stores patterns and performs error correction through associative recall
  - Tests error correction capability at different noise levels
  - Analyzes storage capacity (≈ 0.138N theoretical limit)

- **`eight_rook_hopfield.py`** - Problem 2: Eight-Rook constraint satisfaction
  - Solves the classic Eight-Rook chess problem
  - Places 8 rooks on an 8×8 board with no attacks
  - Uses energy function with constraint penalties
  - Demonstrates constraint satisfaction through energy minimization

- **`tsp_hopfield_tank.py`** - Problem 3: Traveling Salesman Problem
  - Implements Hopfield/Tank network for 10-city TSP
  - 100 neurons (n²) with 9,900 directed weights
  - Encodes tour constraints and distance minimization
  - Demonstrates combinatorial optimization capabilities

- **`lab06_hopfield_demo.py`** - Comprehensive demonstration script
  - Runs all three implementations
  - Generates visualizations and reports
  - Creates comprehensive summary document

## Running the Code

### Run All Demonstrations

```bash
python lab06_hopfield_demo.py
```

This will execute all three problems and generate complete output.

### Run Individual Problems

```bash
# Problem 1: Error Correction
python hopfield_error_correction.py

# Problem 2: Eight-Rook
python eight_rook_hopfield.py

# Problem 3: TSP
python tsp_hopfield_tank.py
```

## Requirements

```bash
pip install numpy matplotlib
```

## Problem Descriptions

### Problem 1: Error-Correcting Capability

**Objective**: Analyze error-correcting capability under Hebbian storage

**Theory**:
- Network: N neurons with states s_i ∈ {-1, +1}
- Hebbian weights: w_ij = (1/N) Σ_μ ξ^μ_i ξ^μ_j
- Storage capacity: P_max ≈ 0.138N
- Error correction within basin of attraction

**Implementation**:
- 256 neurons (16×16 grid)
- 3 stored patterns (letters T, L, I)
- Tested at noise levels: 5%, 10%, 15%, 20%, 25%, 30%

**Key Results**:
- ✓ Perfect recovery at 5-10% noise
- ✓ >50% success rate up to 15-20% bit flips
- ✓ Fast convergence (<10 iterations)

### Problem 2: Eight-Rook Problem

**Objective**: Use Hopfield energy to solve constraint satisfaction

**Problem**: Place 8 rooks on 8×8 board, no two sharing row or column

**Encoding**:
- Neurons: x_ij ∈ {0,1} (rook at position i,j)
- 64 total neurons

**Constraints**:
- C1 (Row): Σ_j x_ij = 1 for all i
- C2 (Column): Σ_i x_ij = 1 for all j

**Energy Function**:
```
E = (A/2) Σ_i (Σ_j x_ij - 1)² + (B/2) Σ_j (Σ_i x_ij - 1)²
```

**Weight Design**:
- Same row: w = -A (discourages multiple rooks)
- Same column: w = -B (prevents collisions)
- Biases: θ_ij = -A - B

**Key Results**:
- ✓ 70-90% success rate
- ✓ Fast convergence (~50-100 iterations)
- ✓ Energy = 0 indicates valid solution

### Problem 3: TSP for 10 Cities

**Objective**: Encode TSP in Hopfield/Tank network, derive weight count

**Encoding**:
- Neurons: V_ip ∈ {0,1} (city i at position p)
- Total: n² = 100 neurons for n=10

**Constraints**:
- C1: Each city once: Σ_p V_ip = 1 for all i
- C2: One city per position: Σ_i V_ip = 1 for all p

**Energy Function**:
```
E = (A/2) Σ_i (Σ_p V_ip - 1)²           [constraint C1]
  + (B/2) Σ_p (Σ_i V_ip - 1)²           [constraint C2]
  + (D/2) Σ_p Σ_{i≠j} d_ij V_ip V_j,p±1  [tour length]
```

**Weight Matrix Size**:
- Directed weights: N(N-1) = 100 × 99 = **9,900**
- Unique undirected pairs: N(N-1)/2 = **4,950**
- Plus N = 100 threshold values

**Key Results**:
- ✓ Correctly sized weight matrix (9,900 weights)
- ✓ Valid tours found (10-40% success rate)
- ✓ Heuristic approximation (local minima challenges)

## Output Files

### Problem 1 Outputs
- `hopfield_stored_patterns.png` - Visualization of stored patterns
- `hopfield_error_correction_example.png` - Example recovery process
- `hopfield_performance_curves.png` - Success rate vs noise level
- `hopfield_error_correction_report.txt` - Detailed analysis

### Problem 2 Outputs
- `eight_rook_solutions.png` - Multiple valid configurations
- `eight_rook_convergence.png` - Convergence sequence
- `eight_rook_report.txt` - Detailed results

### Problem 3 Outputs
- `tsp_cities.png` - City locations
- `tsp_best_solution.png` - Best tour found
- `tsp_tour_length_distribution.png` - Quality distribution
- `tsp_10_cities_report.txt` - Detailed analysis

### Comprehensive Summary
- `lab06_comprehensive_report.txt` - Complete analysis of all three problems

## Theoretical Insights

### Energy Minimization Framework
Hopfield networks implement gradient descent on a Lyapunov energy function. Asynchronous updates guarantee monotonic energy decrease, ensuring convergence to local minima.

### Storage Capacity
For random patterns, theoretical capacity is P_max ≈ 0.138N. Beyond this, crosstalk noise induces spurious minima and retrieval errors.

### Weight Design Principles
1. Identify constraints and objectives
2. Formulate energy function with penalty terms
3. Expand E to derive pairwise weights and biases
4. Negative couplings discourage conflicting activations
5. Balance constraint penalties vs objective weights

### Scaling and Complexity
- **Problem 1**: O(N²) weights for N neurons
- **Problem 2**: O(n⁴) weights for n×n board
- **Problem 3**: O(n⁴) weights, exponential solution space

Practical limits: ~100-500 neurons for discrete binary Hopfield networks.

## Key Learnings

1. **Energy-based Learning**: Hopfield networks provide unified framework for memory and optimization
2. **Constraint Satisfaction**: Penalty terms naturally encode constraints in energy function
3. **Local Minima**: Main limitation - not guaranteed to find global optimum
4. **Parameter Tuning**: Critical balance between constraint penalties and objective weights
5. **Scalability**: O(n⁴) scaling limits practical problem size

## Comparative Analysis

| Aspect | Problem 1 | Problem 2 | Problem 3 |
|--------|-----------|-----------|-----------|
| Type | Associative Memory | Constraint Satisfaction | Combinatorial Optimization |
| Neurons | 256 | 64 | 100 |
| Weights | ~65K | ~4K | ~10K |
| Learning | Hebbian | Hand-crafted | Hand-crafted |
| Convergence | Fast (<10 iter) | Medium (~100) | Slow (1000+) |
| Success Rate | High (90%+) | High (70-90%) | Medium (10-40%) |

## References

1. J.J. Hopfield, "Neural networks and physical systems with emergent collective computational abilities," PNAS, vol. 79, no. 8, 1982.

2. J.J. Hopfield and D.W. Tank, "Neural computation of decisions in optimization problems," Biological Cybernetics, vol. 52, 1985.

3. D.J.C. MacKay, "Information Theory, Inference and Learning Algorithms," Cambridge University Press, 2003.

## Authors

- Kush Sonawane (202351137) - BTech CSE, IIIT Vadodara
- Ram Sharma (202351129) - BTech CSE, IIIT Vadodara
- Ronak Vaghela (202351152) - BTech CSE, IIIT Vadodara

## License

This code is for educational purposes as part of the Artificial Intelligence course at IIIT Vadodara.
