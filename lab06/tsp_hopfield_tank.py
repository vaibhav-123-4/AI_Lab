"""
Hopfield/Tank Network for Traveling Salesman Problem (TSP)
Lab 06 - Problem 3: TSP for 10 Cities

This module implements a Hopfield/Tank network to solve the 10-city TSP
by encoding tour constraints and distance minimization in the energy function.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import os


class TSPHopfieldTank:
    """
    Hopfield/Tank network for solving the Traveling Salesman Problem.
    
    Encoding: V_ip ∈ {0,1} indicates city i at tour position p
    
    Constraints:
        C1: Each city appears exactly once: Σ_p V_ip = 1 for all i
        C2: Each position has exactly one city: Σ_i V_ip = 1 for all p
    
    Energy function:
        E = (A/2) Σ_i (Σ_p V_ip - 1)²           [constraint C1]
          + (B/2) Σ_p (Σ_i V_ip - 1)²           [constraint C2]
          + (D/2) Σ_p Σ_{i≠j} d_ij V_ip (V_j,p+1 + V_j,p-1)  [tour length]
    
    For n=10 cities: N = 100 neurons, 9900 directed weights (4950 unique pairs)
    """
    
    def __init__(self, cities: np.ndarray, penalty_A: float = 500, 
                 penalty_B: float = 500, distance_weight: float = 1.0):
        """
        Initialize TSP Hopfield/Tank network.
        
        Args:
            cities: Array of city coordinates (n × 2)
            penalty_A: Constraint penalty for C1 (each city once)
            penalty_B: Constraint penalty for C2 (one city per position)
            distance_weight: Weight D for distance minimization term
        """
        self.n = len(cities)  # Number of cities
        self.N = self.n * self.n  # Total neurons (n²)
        self.cities = cities
        self.A = penalty_A
        self.B = penalty_B
        self.D = distance_weight
        
        # Compute distance matrix
        self.distances = self._compute_distance_matrix()
        
        # Build weight matrix and biases
        self.weights, self.theta = self._build_weights_and_biases()
        
        print(f"Initialized TSP Hopfield/Tank Network for {self.n} cities")
        print(f"  - Total neurons (n²): {self.N}")
        print(f"  - Directed weights: {self.N * (self.N - 1)} = {self.n}² × ({self.n}² - 1)")
        print(f"  - Unique undirected weights: {self.N * (self.N - 1) // 2}")
        print(f"  - Constraint penalties: A={self.A}, B={self.B}")
        print(f"  - Distance weight: D={self.D}")
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """Compute Euclidean distance matrix between all city pairs."""
        n = self.n
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist[i, j] = np.linalg.norm(self.cities[i] - self.cities[j])
        return dist
    
    def _index(self, i: int, p: int) -> int:
        """Convert (city, position) to neuron index."""
        return i * self.n + p
    
    def _neuron_to_city_pos(self, idx: int) -> Tuple[int, int]:
        """Convert neuron index to (city, position)."""
        return divmod(idx, self.n)
    
    def _build_weights_and_biases(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build weight matrix W and bias vector θ from energy function.
        
        Weight contributions:
        1. Constraint C1 (same city i, different positions): w = -A
        2. Constraint C2 (same position p, different cities): w = -B
        3. Distance term (adjacent positions): w = -D * d_ij
        
        Returns:
            Weight matrix W and threshold vector θ
        """
        W = np.zeros((self.N, self.N))
        theta = np.zeros(self.N)
        
        for idx1 in range(self.N):
            i1, p1 = self._neuron_to_city_pos(idx1)
            
            for idx2 in range(self.N):
                if idx1 == idx2:
                    continue
                
                i2, p2 = self._neuron_to_city_pos(idx2)
                
                # Constraint C1: same city, different positions
                if i1 == i2 and p1 != p2:
                    W[idx1, idx2] -= self.A
                
                # Constraint C2: same position, different cities
                if p1 == p2 and i1 != i2:
                    W[idx1, idx2] -= self.B
                
                # Distance term: adjacent positions in tour
                if i1 != i2:
                    # Check if p2 is adjacent to p1 (cyclically)
                    if (p2 == (p1 + 1) % self.n) or (p2 == (p1 - 1) % self.n):
                        W[idx1, idx2] -= self.D * self.distances[i1, i2]
            
            # Bias terms from expanding (Σ - 1)²
            theta[idx1] = -self.A - self.B
        
        return W, theta
    
    def energy(self, state: np.ndarray) -> float:
        """
        Compute total energy of current state.
        
        E = constraint_energy + distance_energy
        
        Args:
            state: Current tour matrix (flattened)
            
        Returns:
            Total energy
        """
        V = state.reshape(self.n, self.n)
        
        # Constraint C1: each city once
        city_sums = V.sum(axis=1)
        c1_penalty = self.A * 0.5 * np.sum((city_sums - 1) ** 2)
        
        # Constraint C2: one city per position
        pos_sums = V.sum(axis=0)
        c2_penalty = self.B * 0.5 * np.sum((pos_sums - 1) ** 2)
        
        # Distance term: sum over adjacent positions
        dist_energy = 0
        for p in range(self.n):
            p_next = (p + 1) % self.n
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        dist_energy += self.D * 0.5 * self.distances[i, j] * V[i, p] * V[j, p_next]
        
        return c1_penalty + c2_penalty + dist_energy
    
    def constraint_violations(self, state: np.ndarray) -> Tuple[int, int]:
        """
        Count constraint violations.
        
        Returns:
            (city_violations, position_violations)
        """
        V = state.reshape(self.n, self.n)
        
        city_sums = V.sum(axis=1)
        pos_sums = V.sum(axis=0)
        
        city_violations = np.sum(np.abs(city_sums - 1) > 0.01)
        pos_violations = np.sum(np.abs(pos_sums - 1) > 0.01)
        
        return city_violations, pos_violations
    
    def is_valid_tour(self, state: np.ndarray) -> bool:
        """Check if state represents a valid tour."""
        city_v, pos_v = self.constraint_violations(state)
        return city_v == 0 and pos_v == 0
    
    def extract_tour(self, state: np.ndarray) -> List[int]:
        """
        Extract tour sequence from state matrix.
        
        Returns:
            List of city indices in tour order
        """
        V = state.reshape(self.n, self.n)
        tour = []
        
        for p in range(self.n):
            # Find which city is at position p
            cities_at_p = np.where(V[:, p] > 0.5)[0]
            if len(cities_at_p) > 0:
                tour.append(cities_at_p[0])
            else:
                # Handle invalid state: pick most active
                tour.append(np.argmax(V[:, p]))
        
        return tour
    
    def tour_length(self, tour: List[int]) -> float:
        """Compute total length of a tour."""
        length = 0
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            length += self.distances[tour[i], tour[j]]
        return length
    
    def update_async(self, state: np.ndarray, max_iter: int = 1000,
                     verbose: bool = False) -> Tuple[np.ndarray, int, List[float]]:
        """
        Asynchronous update with energy minimization.
        
        Args:
            state: Initial configuration
            max_iter: Maximum iterations
            verbose: Print progress
            
        Returns:
            Final state, iterations, energy history
        """
        state = state.copy()
        energy_history = [self.energy(state)]
        
        for iteration in range(max_iter):
            converged = True
            
            # Random update order
            update_order = np.random.permutation(self.N)
            
            for idx in update_order:
                # Compute local field
                h = np.dot(self.weights[idx], state) - self.theta[idx]
                
                new_val = 1 if h > 0 else 0
                
                if abs(new_val - state[idx]) > 0.01:
                    state[idx] = new_val
                    converged = False
            
            current_energy = self.energy(state)
            energy_history.append(current_energy)
            
            if verbose and (iteration % 100 == 0 or iteration < 10):
                city_v, pos_v = self.constraint_violations(state)
                print(f"  Iter {iteration:4d}: E = {current_energy:10.2f}, "
                      f"City_v = {city_v}, Pos_v = {pos_v}")
            
            # Check convergence
            if converged:
                if verbose:
                    print(f"  Converged at iteration {iteration}")
                break
        
        return state, iteration + 1, energy_history
    
    def solve(self, num_attempts: int = 10, max_iter: int = 2000) -> Tuple[np.ndarray, dict]:
        """
        Attempt to solve TSP with multiple random initializations.
        
        Args:
            num_attempts: Number of random initializations
            max_iter: Maximum iterations per attempt
            
        Returns:
            Best solution and statistics
        """
        best_solution = None
        best_tour_length = float('inf')
        valid_solutions = []
        
        stats = {
            'attempts': num_attempts,
            'valid_tours': 0,
            'best_tour_length': None,
            'best_tour': None,
            'avg_iterations': 0,
            'all_tour_lengths': []
        }
        
        total_iterations = 0
        
        print(f"\nSolving {self.n}-city TSP with {num_attempts} random attempts...")
        print("-" * 70)
        
        for attempt in range(num_attempts):
            # Initialize with noisy one-hot per row/column
            state = self._initialize_state()
            
            # Run network
            final_state, iters, energy_hist = self.update_async(
                state, max_iter=max_iter, verbose=False
            )
            
            total_iterations += iters
            
            # Extract tour and check validity
            tour = self.extract_tour(final_state)
            is_valid = self.is_valid_tour(final_state)
            tour_len = self.tour_length(tour)
            
            stats['all_tour_lengths'].append(tour_len)
            
            if is_valid:
                valid_solutions.append((final_state, tour, tour_len))
                stats['valid_tours'] += 1
                print(f"  Attempt {attempt+1:2d}: ✓ Valid tour, length = {tour_len:.2f}, "
                      f"{iters} iterations")
                
                if tour_len < best_tour_length:
                    best_tour_length = tour_len
                    best_solution = final_state
                    stats['best_tour'] = tour
            else:
                city_v, pos_v = self.constraint_violations(final_state)
                print(f"  Attempt {attempt+1:2d}: ✗ Invalid (C_v={city_v}, P_v={pos_v}), "
                      f"approx_len = {tour_len:.2f}")
        
        stats['best_tour_length'] = best_tour_length
        stats['avg_iterations'] = total_iterations / num_attempts
        
        print("-" * 70)
        print(f"Results: {stats['valid_tours']}/{num_attempts} valid tours found")
        if stats['valid_tours'] > 0:
            print(f"Best tour length: {best_tour_length:.2f}")
        print(f"Average iterations: {stats['avg_iterations']:.1f}")
        
        return best_solution, stats
    
    def _initialize_state(self) -> np.ndarray:
        """
        Initialize state with noisy permutation matrix.
        Start with valid permutation and add small noise.
        """
        # Start with identity permutation
        V = np.eye(self.n)
        
        # Add small random noise and re-normalize
        noise = np.random.uniform(0, 0.1, (self.n, self.n))
        V = V + noise
        
        # Normalize rows and columns (approximately)
        V = V / V.sum(axis=1, keepdims=True)
        
        # Binarize
        V = (V > 0.5).astype(float)
        
        # If not valid, randomly permute
        if not self.is_valid_tour(V.flatten()):
            perm = np.random.permutation(self.n)
            V = np.eye(self.n)[perm]
        
        return V.flatten()
    
    def visualize_tour(self, state: np.ndarray, title: str = "TSP Tour", ax=None):
        """
        Visualize TSP tour on city map.
        
        Args:
            state: Tour configuration
            title: Plot title
            ax: Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        tour = self.extract_tour(state)
        is_valid = self.is_valid_tour(state)
        tour_len = self.tour_length(tour)
        
        # Plot cities
        ax.scatter(self.cities[:, 0], self.cities[:, 1], 
                  c='red', s=200, zorder=5, edgecolors='black', linewidths=2)
        
        # Label cities
        for i, (x, y) in enumerate(self.cities):
            ax.text(x, y, str(i), fontsize=14, ha='center', va='center',
                   color='white', weight='bold', zorder=6)
        
        # Draw tour edges
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            city1, city2 = tour[i], tour[j]
            
            x_vals = [self.cities[city1, 0], self.cities[city2, 0]]
            y_vals = [self.cities[city1, 1], self.cities[city2, 1]]
            
            color = 'blue' if is_valid else 'orange'
            alpha = 0.7 if is_valid else 0.5
            
            ax.plot(x_vals, y_vals, color=color, linewidth=2, alpha=alpha, zorder=3)
            
            # Add arrow
            dx = x_vals[1] - x_vals[0]
            dy = y_vals[1] - y_vals[0]
            ax.arrow(x_vals[0] + 0.3*dx, y_vals[0] + 0.3*dy, 
                    0.4*dx, 0.4*dy,
                    head_width=0.3, head_length=0.2, fc=color, ec=color,
                    alpha=alpha, zorder=4)
        
        status = "Valid" if is_valid else "Invalid"
        ax.set_title(f"{title}\n{status} Tour, Length = {tour_len:.2f}", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')


def generate_random_cities(n: int, seed: int = None) -> np.ndarray:
    """Generate random city coordinates in unit square."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(0, 100, (n, 2))


def demo_tsp_10_cities():
    """
    Demonstrate TSP solving with Hopfield/Tank network for 10 cities.
    """
    print("=" * 70)
    print("TRAVELING SALESMAN PROBLEM (10 CITIES) VIA HOPFIELD/TANK NETWORK")
    print("=" * 70)
    
    # Generate 10 random cities
    cities = generate_random_cities(n=10, seed=42)
    
    # Create network with carefully tuned parameters
    # High penalties to enforce constraints, lower distance weight
    network = TSPHopfieldTank(cities, penalty_A=500, penalty_B=500, distance_weight=1.0)
    
    # Solve with multiple attempts
    solution, stats = network.solve(num_attempts=30, max_iter=2000)
    
    # Create output directory
    output_dir = 'lab06'
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize city locations
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(cities[:, 0], cities[:, 1], c='red', s=200, 
              zorder=5, edgecolors='black', linewidths=2)
    for i, (x, y) in enumerate(cities):
        ax.text(x, y, str(i), fontsize=14, ha='center', va='center',
               color='white', weight='bold', zorder=6)
    ax.set_title('10 City Locations', fontsize=16, fontweight='bold')
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tsp_cities.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir}/tsp_cities.png")
    plt.close()
    
    # Visualize best solution
    if solution is not None and stats['valid_tours'] > 0:
        fig, ax = plt.subplots(figsize=(10, 10))
        network.visualize_tour(solution, "Best TSP Solution Found", ax)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/tsp_best_solution.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/tsp_best_solution.png")
        plt.close()
    
    # Plot tour length distribution
    if len(stats['all_tour_lengths']) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(stats['all_tour_lengths'], bins=15, color='#2E86AB', 
               edgecolor='black', alpha=0.7)
        if stats['best_tour_length'] is not None and stats['best_tour_length'] < float('inf'):
            ax.axvline(stats['best_tour_length'], color='red', linestyle='--',
                      linewidth=2, label=f'Best: {stats["best_tour_length"]:.2f}')
        ax.set_xlabel('Tour Length', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Tour Lengths Across Attempts', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/tsp_tour_length_distribution.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/tsp_tour_length_distribution.png")
        plt.close()
    
    # Generate report
    report_path = f'{output_dir}/tsp_10_cities_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TSP (10 CITIES): HOPFIELD/TANK NETWORK SOLUTION\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Problem Description:\n")
        f.write("-" * 70 + "\n")
        f.write("Find the shortest tour visiting all 10 cities exactly once\n")
        f.write("and returning to the starting city.\n\n")
        
        f.write("Network Architecture:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  - Number of cities (n): {network.n}\n")
        f.write(f"  - Total neurons (n²): {network.N}\n")
        f.write(f"  - Encoding: V_ip = 1 if city i at position p, else 0\n\n")
        
        f.write("Weight Matrix Size:\n")
        f.write(f"  - Directed weights: N×(N-1) = {network.N}×{network.N-1} = {network.N * (network.N - 1)}\n")
        f.write(f"  - Unique undirected pairs: N×(N-1)/2 = {network.N * (network.N - 1) // 2}\n")
        f.write(f"  - Plus N = {network.N} threshold values\n\n")
        
        f.write("Energy Function:\n")
        f.write("-" * 70 + "\n")
        f.write("E = (A/2) Σ_i (Σ_p V_ip - 1)²           [each city once]\n")
        f.write("  + (B/2) Σ_p (Σ_i V_ip - 1)²           [one city per position]\n")
        f.write("  + (D/2) Σ_p Σ_{i≠j} d_ij V_ip V_j,p±1  [minimize tour length]\n\n")
        
        f.write("Parameters:\n")
        f.write(f"  - Constraint penalty A: {network.A}\n")
        f.write(f"  - Constraint penalty B: {network.B}\n")
        f.write(f"  - Distance weight D: {network.D}\n\n")
        
        f.write("Weight Design Rationale:\n")
        f.write("-" * 70 + "\n")
        f.write("1. Constraint C1 (each city once):\n")
        f.write("   - Negative coupling w = -A for same city, different positions\n")
        f.write("   - Prevents multiple occurrences of same city\n\n")
        f.write("2. Constraint C2 (one city per position):\n")
        f.write("   - Negative coupling w = -B for same position, different cities\n")
        f.write("   - Ensures valid permutation matrix\n\n")
        f.write("3. Distance minimization:\n")
        f.write("   - Coupling w = -D×d_ij for cities i,j at adjacent positions\n")
        f.write("   - Favors shorter edges in the tour\n\n")
        f.write("4. Penalty balance:\n")
        f.write("   - A, B >> D ensures constraints satisfied first\n")
        f.write("   - Then distance minimization refines tour quality\n\n")
        
        f.write("Experimental Results:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  - Total attempts: {stats['attempts']}\n")
        f.write(f"  - Valid tours found: {stats['valid_tours']}\n")
        f.write(f"  - Success rate: {stats['valid_tours']/stats['attempts']:.1%}\n")
        f.write(f"  - Average iterations: {stats['avg_iterations']:.1f}\n")
        
        if stats['best_tour_length'] is not None and stats['best_tour_length'] < float('inf'):
            f.write(f"  - Best tour length: {stats['best_tour_length']:.2f}\n")
            f.write(f"  - Best tour sequence: {stats['best_tour']}\n")
        else:
            f.write("  - No valid tour found (all attempts violated constraints)\n")
        f.write("\n")
        
        if len(stats['all_tour_lengths']) > 0:
            f.write("Tour Length Statistics:\n")
            f.write(f"  - Mean: {np.mean(stats['all_tour_lengths']):.2f}\n")
            f.write(f"  - Std Dev: {np.std(stats['all_tour_lengths']):.2f}\n")
            f.write(f"  - Min: {np.min(stats['all_tour_lengths']):.2f}\n")
            f.write(f"  - Max: {np.max(stats['all_tour_lengths']):.2f}\n\n")
        
        f.write("Key Observations:\n")
        f.write("-" * 70 + "\n")
        f.write("1. Complexity:\n")
        f.write("   - TSP is NP-hard; 10! = 3,628,800 possible tours\n")
        f.write("   - Hopfield/Tank provides heuristic approximation\n")
        f.write("   - Not guaranteed to find global optimum\n\n")
        
        f.write("2. Constraint Enforcement:\n")
        f.write(f"   - High penalties (A=B={network.A}) prioritize feasibility\n")
        f.write("   - Network must satisfy permutation constraints first\n")
        f.write("   - Then optimize tour length within feasible space\n\n")
        
        f.write("3. Convergence Challenges:\n")
        f.write("   - Balance between constraint satisfaction and optimization\n")
        f.write("   - Local minima can trap invalid or suboptimal tours\n")
        f.write("   - Multiple random restarts improve solution quality\n\n")
        
        f.write("4. Scaling:\n")
        f.write(f"   - For n={network.n}: {network.N} neurons, {network.N * (network.N - 1)} weights\n")
        f.write("   - Complexity grows as O(n⁴) for weight computations\n")
        f.write("   - Practical limit ~20-30 cities for discrete Hopfield\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("Conclusion:\n")
        f.write("=" * 70 + "\n\n")
        f.write("The Hopfield/Tank network encodes the 10-city TSP as an energy\n")
        f.write("minimization problem with 100 neurons and 9900 directed weights.\n")
        f.write("Constraint penalties ensure valid tours (permutation matrices),\n")
        f.write("while distance couplings favor shorter tours. The discrete binary\n")
        f.write("formulation faces challenges with local minima, but multiple random\n")
        f.write("initializations yield reasonable approximate solutions.\n\n")
        f.write("For improved results, consider:\n")
        f.write("  - Continuous relaxation with sigmoid activations\n")
        f.write("  - Simulated annealing schedule for penalty weights\n")
        f.write("  - Hybrid approaches combining Hopfield with local search\n")
    
    print(f"✓ Saved: {report_path}")
    print("\n" + "=" * 70)
    print("TSP-10 demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    np.random.seed(42)
    demo_tsp_10_cities()
