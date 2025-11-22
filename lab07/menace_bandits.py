import json
import random
import math
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------
# Utility functions for tic-tac-toe / MENACE
# --------------------------

def board_to_list(board_str):
    """Convert 9-char string to list of chars."""
    return list(board_str)

def list_to_board_str(lst):
    return ''.join(lst)

def rotate_board(board):
    """Rotate 3x3 board (list length 9) 90 degrees clockwise."""
    b = board
    return [b[6], b[3], b[0],
            b[7], b[4], b[1],
            b[8], b[5], b[2]]

def reflect_board(board):
    """Reflect board horizontally (mirror)."""
    b = board
    return [b[2], b[1], b[0],
            b[5], b[4], b[3],
            b[8], b[7], b[6]]

def all_symmetries(board):
    """Yield all 8 symmetries (rotations and reflections) of the board."""
    b = board[:]
    for _ in range(4):
        yield b
        yield reflect_board(b)
        b = rotate_board(b)

def canonical_board(board):
    """
    Return lexicographically smallest string among all symmetric transforms.
    Board is list of 9 chars: 'X', 'O', or '.'.
    """
    best = None
    for sym in all_symmetries(board):
        s = ''.join(sym)
        if best is None or s < best:
            best = s
    return best

def legal_moves(board):
    return [i for i, c in enumerate(board) if c == '.']

def check_winner(board):
    """
    Check winner on board (list of 9 chars).
    Return 'X' or 'O' or None if no winner; also returns True if draw.
    """
    lines = [(0,1,2),(3,4,5),(6,7,8),
             (0,3,6),(1,4,7),(2,5,8),
             (0,4,8),(2,4,6)]
    for a,b,c in lines:
        if board[a] != '.' and board[a] == board[b] == board[c]:
            return board[a], False
    if '.' not in board:
        return None, True  # draw
    return None, False

# --------------------------
# MENACE class
# --------------------------

class Menace:
    """
    A pedagogical MENACE implementation.
    - Uses canonicalization (symmetry reduction).
    - Stores matchboxes as dict: canonical_state -> Counter(move -> beads)
    - Plays as 'X' by default (first player); opponent policy can be random or greedy.
    """

    def __init__(self, init_beads=4, win_add=3, draw_add=1, lose_remove=1, min_beads=1):
        self.boxes = dict()
        self.init_beads = init_beads
        self.win_add = win_add
        self.draw_add = draw_add
        self.lose_remove = lose_remove
        self.min_beads = min_beads

    def ensure_box(self, canon_state):
        """Initialize box for canonical board if absent."""
        if canon_state not in self.boxes:
            # legal moves from the canonical state's perspective
            board = list(canon_state)
            moves = legal_moves(board)
            self.boxes[canon_state] = Counter({m: self.init_beads for m in moves})

    def choose_move_from_box(self, canon_state):
        """Weighted random selection from beads (returns absolute move index)."""
        box = self.boxes[canon_state]
        # If box empty (all zero) reinitialize to init_beads to avoid deadlock
        if not box:
            # reinit using legal moves of state
            board = list(canon_state)
            moves = legal_moves(board)
            self.boxes[canon_state] = Counter({m: self.init_beads for m in moves})
            box = self.boxes[canon_state]
        total = sum(box.values())
        r = random.uniform(0, total)
        upto = 0.0
        for move, w in box.items():
            upto += w
            if r <= upto:
                return move
        # fallback
        return random.choice(list(box.keys()))

    def play_one_game(self, opponent='random'):
        """
        Let MENACE ('X') play against opponent.
        Returns outcome: 1 if MENACE wins, 0 draw, -1 loss.
        Opponent can be 'random' or 'greedy' (greedy picks center/corner heuristic).
        """

        board = ['.'] * 9
        history = []  # list of (canonical_state, chosen_move)
        turn = 0  # 0 -> MENACE (X), 1 -> opponent (O)

        while True:
            winner, is_draw = check_winner(board)
            if winner == 'X':
                self.apply_reinforcement(history, 1)
                return 1
            elif winner == 'O':
                self.apply_reinforcement(history, -1)
                return -1
            elif is_draw:
                self.apply_reinforcement(history, 0)
                return 0

            if turn == 0:
                # MENACE turn
                canon = canonical_board(board)
                self.ensure_box(canon)
                move = self.choose_move_from_box(canon)
                board[move] = 'X'
                history.append((canon, move))
            else:
                # Opponent turn
                if opponent == 'random':
                    moves = legal_moves(board)
                    move = random.choice(moves)
                    board[move] = 'O'
                elif opponent == 'greedy':
                    # simple heuristic: take center, then corner, else random
                    if board[4] == '.':
                        board[4] = 'O'
                    else:
                        corners = [i for i in [0,2,6,8] if board[i]=='.']
                        if corners:
                            board[random.choice(corners)] = 'O'
                        else:
                            board[random.choice(legal_moves(board))] = 'O'
                else:
                    # default to random
                    board[random.choice(legal_moves(board))] = 'O'
            turn = 1 - turn

    def apply_reinforcement(self, history, outcome):
        """
        Update bead counts based on outcome for MENACE.
        outcome: 1 (win), 0 (draw), -1 (loss)
        """
        for canon_state, move in history:
            box = self.boxes.get(canon_state)
            if box is None:
                continue
            if outcome == 1:
                box[move] += self.win_add
            elif outcome == 0:
                box[move] += self.draw_add
            else:
                box[move] = max(self.min_beads, box[move] - self.lose_remove)

    def train(self, n_games=1000, opponent='random', verbose=False):
        """Train MENACE for n_games."""
        results = []
        for i in range(n_games):
            res = self.play_one_game(opponent)
            results.append(res)
            if verbose and (i+1) % (n_games//10 if n_games>=10 else 1) == 0:
                wins = sum(1 for r in results if r==1)
                draws = sum(1 for r in results if r==0)
                losses = sum(1 for r in results if r==-1)
                print(f'Game {i+1}/{n_games}: W/D/L = {wins}/{draws}/{losses}')
        return results

    def save(self, fname):
        """Save boxes to JSON file."""
        # Convert Counters to regular dicts with string keys
        dump = {k: dict(v) for k,v in self.boxes.items()}
        with open(fname, 'w') as f:
            json.dump(dump, f)

    def load(self, fname):
        with open(fname, 'r') as f:
            dump = json.load(f)
        self.boxes = {k: Counter(v) for k,v in dump.items()}


# --------------------------
# Bandit environments
# --------------------------

class BinaryBandit:
    """Binary bandit with two arms (Bernoulli)."""

    def __init__(self, p1=0.1, p2=0.2, seed=None):
        self.ps = np.array([p1, p2], dtype=float)
        if seed is not None:
            np.random.seed(seed)

    def pull(self, a):
        """Pull arm a in {0,1}. Return reward 0 or 1."""
        return 1 if random.random() < self.ps[a] else 0

class BinaryBanditA(BinaryBandit):
    def __init__(self, seed=None):
        super().__init__(p1=0.1, p2=0.2, seed=seed)

class BinaryBanditB(BinaryBandit):
    def __init__(self, seed=None):
        super().__init__(p1=0.8, p2=0.9, seed=seed)

class NonStationaryBandit:
    """
    10-armed non-stationary bandit:
    - means initialized to mu0 (default 0)
    - each timestep each mean undergoes mu += N(0, walk_std)
    - reward when pulling arm a is Normal(mu[a], reward_std)
    """

    def __init__(self, k=10, mu0=0.0, walk_std=0.01, reward_std=1.0, seed=None):
        self.k = k
        self.mu = np.array([mu0]*k, dtype=float)
        self.walk_std = walk_std
        self.reward_std = reward_std
        self.t = 0
        if seed is not None:
            np.random.seed(seed)

    def step(self):
        """Advance underlying means by one random-walk step."""
        self.mu += np.random.normal(0.0, self.walk_std, size=self.k)
        self.t += 1

    def pull(self, a):
        """Agent pulls arm a; environment updates means (random walk) then returns reward."""
        # update all means first (as specified)
        self.step()
        return np.random.normal(self.mu[a], self.reward_std)

    def current_best(self):
        return np.argmax(self.mu)

# --------------------------
# Agents
# --------------------------

class EpsilonGreedySampleAvgAgent:
    """Epsilon-greedy using sample-average update (good for stationary)."""

    def __init__(self, k=2, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.Q = np.zeros(k, dtype=float)
        self.N = np.zeros(k, dtype=int)

    def select_action(self):
        if random.random() < self.epsilon:
            return random.randrange(self.k)
        else:
            # tie-breaker: choose randomly among max
            maxv = np.max(self.Q)
            candidates = np.where(self.Q == maxv)[0]
            return int(np.random.choice(candidates))

    def update(self, a, r):
        self.N[a] += 1
        self.Q[a] += (r - self.Q[a]) / self.N[a]

    def reset(self):
        self.Q[:] = 0.0
        self.N[:] = 0

class EpsilonGreedyConstAgent:
    """Epsilon-greedy with constant step-size alpha (good for non-stationary)."""

    def __init__(self, k=10, epsilon=0.1, alpha=0.1):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = np.zeros(k, dtype=float)

    def select_action(self):
        if random.random() < self.epsilon:
            return random.randrange(self.k)
        else:
            maxv = np.max(self.Q)
            candidates = np.where(self.Q == maxv)[0]
            return int(np.random.choice(candidates))

    def update(self, a, r):
        self.Q[a] += self.alpha * (r - self.Q[a])

    def reset(self):
        self.Q[:] = 0.0

# --------------------------
# Experiment runners and plotting
# --------------------------

def run_binary_bandit_experiment(bandit_cls, agent, T=1000, runs=100, seed=None):
    """
    Runs multiple independent trials of an agent on a binary bandit.
    Returns averaged reward over time and percent optimal action (if bandit true best known).
    """
    rewards = np.zeros(T, dtype=float)
    opt_actions = np.zeros(T, dtype=float)

    for run in range(runs):
        if seed is not None:
            random.seed(seed + run)
            np.random.seed(seed + run)
        bandit = bandit_cls()
        agent.reset()
        # determine true best arm
        true_best = np.argmax(bandit.ps)
        for t in range(T):
            a = agent.select_action()
            r = bandit.pull(a)
            agent.update(a, r)
            rewards[t] += r
            if a == true_best:
                opt_actions[t] += 1

    rewards /= runs
    opt_actions = 100.0 * (opt_actions / runs)  # percentage
    return rewards, opt_actions

def run_nonstationary_experiment(env, agent, T=10000, runs=50, seed=None, record_opt=True):
    """
    Runs non-stationary experiments.
    env: class constructor (no-arg) or a factory function returning a new env per run.
    agent: agent instance (with reset method)
    """
    avg_rewards = np.zeros(T, dtype=float)
    opt_pct = np.zeros(T, dtype=float) if record_opt else None

    for run in range(runs):
        if seed is not None:
            random.seed(seed + run)
            np.random.seed(seed + run)
        environment = env()
        agent.reset()
        for t in range(T):
            a = agent.select_action()
            r = environment.pull(a)
            agent.update(a, r)
            avg_rewards[t] += r
            if record_opt:
                opt = (a == environment.current_best())
                opt_pct[t] += 1 if opt else 0

    avg_rewards /= runs
    if record_opt:
        opt_pct = 100.0 * (opt_pct / runs)
    return avg_rewards, opt_pct

def plot_results(rewards, opt_pct=None, title='Results', xlabel='Time steps', save_prefix=None):
    """
    Plot results. If `save_prefix` is provided, the figure is saved to
    `<save_prefix>.png` instead of calling `plt.show()`.
    """
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(rewards)
    plt.title(title + ' — Average Reward')
    plt.xlabel(xlabel)
    plt.ylabel('Reward')
    if opt_pct is not None:
        plt.subplot(1,2,2)
        plt.plot(opt_pct)
        plt.title(title + ' — % Optimal Action')
        plt.xlabel(xlabel)
        plt.ylabel('% Optimal')
    plt.tight_layout()
    if save_prefix:
        fname = f"{save_prefix}.png"
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()

# --------------------------
# Example usage / main
# --------------------------

def example_run_all():
    # MENACE training demo
    print("Training a small MENACE vs random opponent for 200 games...")
    menace = Menace(init_beads=4, win_add=3, draw_add=1, lose_remove=1, min_beads=1)
    res = menace.train(n_games=200, opponent='random', verbose=True)
    print("MENACE training finished. Wins:", sum(1 for r in res if r==1))

    # Binary bandit experiments
    print("Running binary bandit experiments (sample-average epsilon-greedy)...")
    agent = EpsilonGreedySampleAvgAgent(k=2, epsilon=0.1)
    rewardsA, optA = run_binary_bandit_experiment(BinaryBanditA, agent, T=500, runs=200, seed=0)
    rewardsB, optB = run_binary_bandit_experiment(BinaryBanditB, agent, T=500, runs=200, seed=1)
    plot_results(rewardsA, optA, title='BinaryBanditA (sample-avg eps-greedy)', save_prefix='binaryA_sampleavg')
    plot_results(rewardsB, optB, title='BinaryBanditB (sample-avg eps-greedy)', save_prefix='binaryB_sampleavg')

    # Non-stationary experiment: compare sample-average vs const-alpha
    print("Running non-stationary 10-armed bandit comparison...")
    def env_factory():
        return NonStationaryBandit(k=10, mu0=0.0, walk_std=0.01, reward_std=1.0, seed=None)
    agent1 = EpsilonGreedySampleAvgAgent(k=10, epsilon=0.1)  # not ideal for non-stationary
    agent2 = EpsilonGreedyConstAgent(k=10, epsilon=0.1, alpha=0.1)
    T = 5000
    rewards1, opt1 = run_nonstationary_experiment(env_factory, agent1, T=T, runs=50, seed=42)
    rewards2, opt2 = run_nonstationary_experiment(env_factory, agent2, T=T, runs=50, seed=42)
    plot_results(rewards1, opt1, title='Nonstationary: sample-avg eps-greedy', save_prefix='nonstationary_sampleavg')
    plot_results(rewards2, opt2, title='Nonstationary: const-alpha eps-greedy', save_prefix='nonstationary_constalpha')

if __name__ == '__main__':
    # Run example if executed directly
    example_run_all()