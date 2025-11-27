"""
Core function for two player zero sum game using linear programming:
    INPUT:
        A : 2D numpy array of shape (m, n)
            Player 1's payoff matrix (row player).
                - Rows = strategies for Player 1
                - Columns = strategies for Player 2
                - A[i, j] = payoff (advantage) for Player 1 when:
                    Player 1 uses row i
                    Player 2 uses column j
            Player 2's payoff is -A[i, j].
    OUTPUT:
        p : 1D numpy array of length m
            Optimal mixed strategy for Player 1 (row player).
            p[i] = probability of playing row i.
        q : 1D numpy array of length n
            Optimal mixed strategy for Player 2 (column player).
            q[j] = probability of playing column j.
        v : float
            Value of the game to Player 1.
            This is the long-run expected payoff for Player 1
            if both players play optimally (use p and q).
            - v > 0 : game favors Player 1
            - v = 0 : fair game
            - v < 0 : game favors Player 2
"""
import numpy as np
from scipy.optimize import linprog

def solve_zero_sum_game(A):
    """
    Solve a two-player zero-sum game.
    
    Input:
        A = payoff matrix for Player 1 (rows)
        
    Output:
        p = optimal strategy for Player 1
        q = optimal strategy for Player 2
        v = game value (expected payoff)
    """
    A = np.array(A, dtype=float)
    m, n = A.shape  # m = number of rows, n = number of columns
    
    # --- Solve for Player 1 (maximize v) ---
    # We want: maximize v
    # Where: sum(p[i] * A[i,j]) >= v for all j 
    #       -> meaing if player 1 randomizes with probability, then even with player 2's best move player 1 still gets >= v (at least v)

    #             sum(p[i]) = 1  -> meaning sum of prob is 1

    #             p[i] >= 0  -> meaning prob cant be neg
    
    # linprog minimizes, so we minimize "-v"
    c = np.zeros(m + 1)
    c[-1] = -1  # minimize "-v" (which maximizes v)
    
    # Inequality constraints: "-sum(p[i] * A[i,j])" + v <= 0
    A_ub = np.hstack([-A.T, np.ones((n, 1))])
    b_ub = np.zeros(n)
    
    # Equality constraint: sum(p[i]) = 1
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1
    b_eq = np.array([1])
    
    # Bounds: p[i] >= 0, v is free
    bounds = [(0, None)] * m + [(None, None)]
    
    # Solve
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not result.success:
        raise ValueError("Could not solve for Player 1")
    
    p = result.x[:m]  # get the probabilities
    v = result.x[-1]  # get the game value
    
    # idk what this does tbh
    p[p < 0] = 0
    p = p / p.sum()  # normalize to sum to 1
    
    # --- Solve for Player 2 (minimize v) ---
    # Player 2 wants to minimize Player 1's payoff
    # We can solve the same problem with matrix -A.T -> taking transpose of A and multiply by -1 thus flipping the matrix
    
    B = -A.T  # flip the game for Player 2
    
    # Same setup but for Player 2
    c2 = np.zeros(n + 1)
    c2[-1] = -1
    
    A_ub2 = np.hstack([-B.T, np.ones((m, 1))])
    b_ub2 = np.zeros(m)
    
    A_eq2 = np.zeros((1, n + 1))
    A_eq2[0, :n] = 1
    b_eq2 = np.array([1])
    
    bounds2 = [(0, None)] * n + [(None, None)]
    
    # vaguely understand this
    result2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2,
                      bounds=bounds2, method='highs')
    
    if not result2.success:
        raise ValueError("Could not solve for Player 2")
    
    q = result2.x[:n]
    q[q < 0] = 0
    q = q / q.sum()
    
    return p, q, v


# --- Example: Rock-Paper-Scissors ---
if __name__ == "__main__":
    # Payoff matrix for Player 1
    #           Rock  Paper  Scissors
    # Rock       0     -1      1
    # Paper      1      0     -1
    # Scissors  -1      1      0
    
    A = np.array([
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0]
    ])
    
    p, q, v = solve_zero_sum_game(A)

    # Prints for RPS:

    print("Rock-Paper-Scissors Solution:")
    print("-" * 40)
    print(f"Player 1 strategy: {p}")
    print(f"Player 2 strategy: {q}")
    print(f"Game value: {v:.6f}")
    print()
    print("Interpretation:")
    print(f"Player 1 should play:")
    print(f"  Rock: {p[0]*100:.1f}%")
    print(f"  Paper: {p[1]*100:.1f}%")
    print(f"  Scissors: {p[2]*100:.1f}%")
    print()
    print(f"Player 2 should play:")
    print(f"  Rock: {q[0]*100:.1f}%")
    print(f"  Paper: {q[1]*100:.1f}%")
    print(f"  Scissors: {q[2]*100:.1f}%")


# --- General Examples ---

# Example 1: A 2x2 game
print("=== 2x2 Game ===")
A = np.array([
    [3, -1],
    [-2, 4]
])
p, q, v = solve_zero_sum_game(A)
# Prints for example 1:
print(f"Player 1 strategy: {p}")
print(f"Player 2 strategy: {q}")
print(f"Game value: {v}\n")
print("Interpretation:")
print(f"Player 1 should play:")
print(f"  Move 1: {p[0]*100:.1f}%")
print(f"  Move 2: {p[1]*100:.1f}%")
print()
print(f"Player 2 should play:")
print(f"  Move 1: {q[0]*100:.1f}%")
print(f"  Move 2: {q[1]*100:.1f}%")

# Example 2: A 3x2 game (3 strategies for P1, 2 for P2)
print("=== 3x2 Game ===")
A = np.array([
    [2, -1],
    [0, 3],
    [1, 1]
])
p, q, v = solve_zero_sum_game(A)
# Prints for example 2:
print(f"Player 1 strategy: {p}")
print(f"Player 2 strategy: {q}")
print(f"Game value: {v}\n")

# Example 3: A 4x4 game
print("=== 4x4 Game ===")
A = np.array([
    [1, -2, 3, 0],
    [0, 1, -1, 2],
    [2, 0, 1, -1],
    [-1, 3, 0, 1]
])
p, q, v = solve_zero_sum_game(A)
# Prints for example 3:
print(f"Player 1 strategy: {p}")
print(f"Player 2 strategy: {q}")
print(f"Game value: {v}\n")

# Example 4: custom
print("=== Custom Game ===")
A = np.array([
    # Put your payoff matrix here
    [5, 2, -3],
    [1, 4, 0],
])
p, q, v = solve_zero_sum_game(A)
# Prints for example 4:
print(f"Player 1 strategy: {p}")
print(f"Player 2 strategy: {q}")
print(f"Game value: {v}")