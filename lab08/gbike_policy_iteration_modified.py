import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import lru_cache
from collections import defaultdict
import os

# -----------------------
# Parameters (changeable)
# -----------------------
MAX_BIKES = 20            # max bikes per location
MAX_MOVE = 5              # max bikes moved overnight (net)
RENTAL_REWARD = 10        # INR per rental
MOVE_UNIT_COST = 2        # INR per bike moved (beyond the free shuttle)
FREE_SHUTTLE = 1          # number of bikes moved free from loc1->loc2
PARKING_THRESHOLD = 10    # >10 bikes triggers a parking penalty
PARKING_COST_FLAT = 4     # flat INR if threshold exceeded at a location
GAMMA = 0.9               # discount factor
THETA = 1e-3              # policy evaluation threshold

# Poisson parameters
LAM_RENT_1 = 3.0
LAM_RENT_2 = 4.0
LAM_RET_1 = 3.0
LAM_RET_2 = 2.0

# output files
POLICY_CSV = "gbike_optimal_policy_free_shuttle_parking_max20.csv"
VALUE_CSV = "gbike_value_function_free_shuttle_parking_max20.csv"
POLICY_PNG = "policy_heatmap.png"
VALUE_PNG = "value_heatmap.png"

# -----------------------
# Helpers: Poisson probs
# -----------------------
@lru_cache(None)
def poisson_pmf(n, lam):
    return math.exp(-lam) * (lam ** n) / math.factorial(n)

def poisson_probs(lam, max_n=15):
    """
    Return a numpy array of probabilities for 0..(max_n-1).
    The tail probability (>= max_n) is folded into the last bucket.
    """
    probs = [poisson_pmf(n, lam) for n in range(max_n)]
    tail = 1.0 - sum(probs)
    probs[-1] += tail
    return np.array(probs)

# choose cutoffs (sufficiently large to cover probability mass)
CUTOFF = 15
req1_probs = poisson_probs(LAM_RENT_1, CUTOFF)
req2_probs = poisson_probs(LAM_RENT_2, CUTOFF)
ret1_probs = poisson_probs(LAM_RET_1, CUTOFF)
ret2_probs = poisson_probs(LAM_RET_2, CUTOFF)

# -----------------------
# Precompute per-location dynamics
# -----------------------
def compute_location_dynamics(req_probs, ret_probs):
    """
    For each possible number of bikes b available at the start of the day,
    compute:
      - trans[b]: dict mapping next_b -> probability
      - expected_rentals[b]: expected number of rentals given start b
    """
    trans = {}
    expected_rentals = {}
    max_n_req = len(req_probs)
    max_n_ret = len(ret_probs)
    for b in range(MAX_BIKES + 1):
        outcomes = defaultdict(float)
        exp_r = 0.0
        for r, p_r in enumerate(req_probs):
            actual_r = min(b, r)
            prob_r = p_r
            bikes_after_r = b - actual_r
            exp_r += prob_r * actual_r
            for ret, p_ret in enumerate(ret_probs):
                prob = prob_r * p_ret
                next_b = min(bikes_after_r + ret, MAX_BIKES)
                outcomes[next_b] += prob
        trans[b] = dict(outcomes)
        expected_rentals[b] = exp_r
    return trans, expected_rentals

loc1_trans, loc1_exp_r = compute_location_dynamics(req1_probs, ret1_probs)
loc2_trans, loc2_exp_r = compute_location_dynamics(req2_probs, ret2_probs)

# -----------------------
# State and actions
# -----------------------
states = [(i,j) for i in range(MAX_BIKES+1) for j in range(MAX_BIKES+1)]
actions = list(range(-MAX_MOVE, MAX_MOVE+1))

def valid_actions(state):
    i, j = state
    valid = []
    for a in actions:
        i_after = i - a
        j_after = j + a
        # must be within capacity limits
        if i_after < 0 or i_after > MAX_BIKES or j_after < 0 or j_after > MAX_BIKES:
            continue
        # cannot move more than available at source
        if a > 0 and a > i:
            continue
        if a < 0 and -a > j:
            continue
        valid.append(a)
    return valid

# -----------------------
# Precompute dynamics & expected reward for state-action pairs
# -----------------------
dynamics_cache = {}
for s in states:
    i, j = s
    for a in valid_actions(s):
        i_after = i - a
        j_after = j + a

        # Move cost with free shuttle from loc1 -> loc2
        if a > 0:
            paid_moves = max(0, a - FREE_SHUTTLE)
            move_cost_total = MOVE_UNIT_COST * paid_moves
        else:
            move_cost_total = MOVE_UNIT_COST * abs(a)

        # Parking cost (based on overnight counts after move)
        parking_cost_total = 0
        if i_after > PARKING_THRESHOLD:
            parking_cost_total += PARKING_COST_FLAT
        if j_after > PARKING_THRESHOLD:
            parking_cost_total += PARKING_COST_FLAT

        # expected rental revenue (INR)
        expected_rentals_income = (loc1_exp_r[i_after] + loc2_exp_r[j_after]) * RENTAL_REWARD

        # joint next-state distribution via outer product
        outcomes = defaultdict(float)
        for next_i, p_i in loc1_trans[i_after].items():
            for next_j, p_j in loc2_trans[j_after].items():
                outcomes[(next_i, next_j)] += p_i * p_j

        expected_reward = expected_rentals_income - move_cost_total - parking_cost_total
        dynamics_cache[(s,a)] = (expected_reward, dict(outcomes))

# -----------------------
# Policy iteration routines
# -----------------------
# initialize policy to 0 where valid (no move)
policy = {}
for s in states:
    va = valid_actions(s)
    policy[s] = 0 if 0 in va else va[0]

# initialize V(s)
V = {s: 0.0 for s in states}

def policy_evaluation(policy, V, theta=THETA):
    """Iterative policy evaluation for current policy."""
    while True:
        delta = 0.0
        for s in states:
            a = policy[s]
            exp_reward, trans = dynamics_cache[(s,a)]
            v_new = exp_reward + GAMMA * sum(p * V[next_s] for next_s, p in trans.items())
            delta = max(delta, abs(V[s] - v_new))
            V[s] = v_new
        if delta < theta:
            break
    return V

def policy_improvement(V, policy):
    """Policy improvement step: greedy wrt current V."""
    policy_stable = True
    for s in states:
        old_a = policy[s]
        best_a = old_a
        best_val = -1e12
        for a in valid_actions(s):
            exp_reward, trans = dynamics_cache[(s,a)]
            q = exp_reward + GAMMA * sum(p * V[next_s] for next_s, p in trans.items())
            if q > best_val:
                best_val = q
                best_a = a
        policy[s] = best_a
        if best_a != old_a:
            policy_stable = False
    return policy_stable

# -----------------------
# Run policy iteration
# -----------------------
MAX_ITERS = 100
iters = 0
while True:
    iters += 1
    V = policy_evaluation(policy, V)
    stable = policy_improvement(V, policy)
    print(f"Policy Iteration loop {iters} done. Stable = {stable}")
    if stable or iters >= MAX_ITERS:
        break

print("Policy iteration finished after", iters, "iterations.")

# -----------------------
# Export results
# -----------------------
# Convert to matrices for saving and plotting (rows = bikes at loc1, cols = bikes at loc2)
policy_matrix = np.zeros((MAX_BIKES+1, MAX_BIKES+1), dtype=int)
value_matrix = np.zeros((MAX_BIKES+1, MAX_BIKES+1), dtype=float)
for i in range(MAX_BIKES+1):
    for j in range(MAX_BIKES+1):
        policy_matrix[i, j] = policy[(i, j)]
        value_matrix[i, j] = V[(i, j)]

# Save CSVs
pd.DataFrame(policy_matrix, index=range(MAX_BIKES+1), columns=range(MAX_BIKES+1)) \
    .to_csv(POLICY_CSV)
pd.DataFrame(np.round(value_matrix, 2), index=range(MAX_BIKES+1), columns=range(MAX_BIKES+1)) \
    .to_csv(VALUE_CSV)

print("Saved CSVs:", POLICY_CSV, VALUE_CSV)

# -----------------------
# Plot heatmaps and save
# -----------------------
plt.figure(figsize=(8,7))
plt.imshow(policy_matrix, origin='lower', aspect='auto')
plt.title("Optimal Policy (action: move loc1->loc2)")
plt.xlabel("Bikes at Location 2")
plt.ylabel("Bikes at Location 1")
cbar = plt.colorbar()
cbar.set_label("Action (pos => loc1->loc2)")
plt.tight_layout()
plt.savefig(POLICY_PNG, dpi=300)
plt.close()

plt.figure(figsize=(8,7))
plt.imshow(value_matrix, origin='lower', aspect='auto')
plt.title("State Value Function V(i,j)")
plt.xlabel("Bikes at Location 2")
plt.ylabel("Bikes at Location 1")
cbar = plt.colorbar()
cbar.set_label("V(i,j) (INR)")
plt.tight_layout()
plt.savefig(VALUE_PNG, dpi=300)
plt.close()

print("Saved images:", POLICY_PNG, VALUE_PNG)
print("Done.")