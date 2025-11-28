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
import itertools

def solve_zero_sum_game(A, tol=1e-8):
    """
    Solve a two-player zero-sum game using support enumeration.
    
    Input:
        A : 2D numpy array of shape (m, n)
            Payoff matrix for Player 1 (row player).
    
    Output:
        p : 1D numpy array of length m
            Optimal mixed strategy for Player 1.
        q : 1D numpy array of length n
            Optimal mixed strategy for Player 2.
        v : float
            Value of the game for Player 1.
    """
    A = np.array(A, dtype=float)
    m, n = A.shape

    # Helper to normalize a probability vector and clip small negatives
    def normalize_prob(x):
        x = np.array(x, dtype=float)
        x[x < 0] = 0.0
        s = x.sum()
        if s < tol:
            return None  # invalid
        return x / s

    # Try supports of size 1, 2, ..., min(m, n)
    max_support_size = min(m, n)

    for k in range(1, max_support_size + 1):
        # All subsets S of rows, size k
        for S in itertools.combinations(range(m), k):
            # All subsets T of columns, size k
            for T in itertools.combinations(range(n), k):
                S = list(S)
                T = list(T)

                # Extract the k x k submatrix A_ST
                A_ST = A[np.ix_(S, T)]

                # === Step 1: Solve for q_T and v from row-player's perspective ===
                # Equations:
                # For each i in S: sum_{j in T} A[i,j] * q_j - v = 0   (k equations)
                # Plus: sum_{j in T} q_j = 1                          (1 equation)
                #
                # Unknowns: q_T (k variables), v (1 variable) => k+1 unknowns
                # Build matrix M_q (shape (k+1, k+1)) and vector b_q (shape (k+1,))
                M_q = np.zeros((k + 1, k + 1))
                b_q = np.zeros(k + 1)

                # Rows 0..k-1: A_ST[i,:] * q - v = 0
                # Columns 0..k-1 for q, last column for v
                M_q[0:k, 0:k] = A_ST
                M_q[0:k, -1] = -1.0  # coefficient of v
                b_q[0:k] = 0.0

                # Last equation: sum q_j = 1
                M_q[-1, 0:k] = 1.0
                M_q[-1, -1] = 0.0
                b_q[-1] = 1.0

                # Try to solve M_q x_q = b_q
                try:
                    x_q = np.linalg.solve(M_q, b_q)
                except np.linalg.LinAlgError:
                    continue  # singular system, skip this support pair

                q_T = x_q[0:k]
                v_q = x_q[-1]

                q_T = normalize_prob(q_T)
                if q_T is None:
                    continue

                # Build full q vector
                q = np.zeros(n)
                q[T] = q_T

                # Check row best-response conditions:
                #   payoff_i = (A q)_i
                #   For i in S: payoff_i ≈ v
                #   For i not in S: payoff_i <= v + tol
                row_payoffs = A @ q  # shape (m,)
                if not np.all(np.abs(row_payoffs[S] - v_q) <= 1e-5):
                    continue
                if np.any(row_payoffs[[i for i in range(m) if i not in S]] > v_q + 1e-5):
                    continue

                # === Step 2: Solve for p_S and v from column-player's perspective ===
                # Equations:
                # For each j in T: sum_{i in S} p_i * A[i,j] - v = 0   (k equations)
                # Plus: sum_{i in S} p_i = 1                          (1 equation)
                #
                # Unknowns: p_S (k vars), v (1 var)
                M_p = np.zeros((k + 1, k + 1))
                b_p = np.zeros(k + 1)

                # Columns 0..k-1 for p, last column for v
                # For each j in T, equation: sum_i p_i * A[i,j] - v = 0
                # That is (p^T A_ST[:, j_idx]) - v = 0
                for j_idx in range(k):
                    M_p[j_idx, 0:k] = A_ST[:, j_idx]
                    M_p[j_idx, -1] = -1.0
                    b_p[j_idx] = 0.0

                # Last equation: sum p_i = 1
                M_p[-1, 0:k] = 1.0
                M_p[-1, -1] = 0.0
                b_p[-1] = 1.0

                try:
                    x_p = np.linalg.solve(M_p, b_p)
                except np.linalg.LinAlgError:
                    continue

                p_S = x_p[0:k]
                v_p = x_p[-1]

                p_S = normalize_prob(p_S)
                if p_S is None:
                    continue

                # Check that v from both sides is (approximately) the same
                if abs(v_p - v_q) > 1e-5:
                    continue

                v = 0.5 * (v_p + v_q)

                # Build full p vector
                p = np.zeros(m)
                p[S] = p_S

                # Check column best-response conditions:
                #   col_payoff_j = (p^T A)_j
                #   For j in T: col_payoff_j ≈ v
                #   For j not in T: col_payoff_j >= v - tol
                col_payoffs = p @ A  # shape (n,)
                if not np.all(np.abs(col_payoffs[T] - v) <= 1e-5):
                    continue
                if np.any(col_payoffs[[j for j in range(n) if j not in T]] < v - 1e-5):
                    continue

                # If we reach here, (p, q, v) is a valid equilibrium
                return p, q, float(v)

    # If no support pair works, something is wrong (or the game is degenerate)
    raise ValueError("No equilibrium found by support enumeration (possible degeneracy).")



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