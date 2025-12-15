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
    
    This method uses ONLY linear algebra (numpy) and NO linear programming.
    It is the mathematically correct equilibrium-finding method for
    small games (2×2, 3×3, 4×4).

    ----------------------------------------------------
    IDEA OF SUPPORT ENUMERATION (IN MATH TERMS)
    ----------------------------------------------------
    A zero-sum game has a payoff matrix A (m × n).
    A mixed strategy equilibrium (p*, q*, v) satisfies:

        For every row i in the support of p:
            (A q*)_i = v          (equal payoff)
        For every other row k:
            (A q*)_k ≤ v          (no profitable deviation)

        For every column j in the support of q:
            (p^T A)_j = v         (equal payoff)
        For every other column ℓ:
            (p^T A)_ℓ ≥ v         (column player can't reduce value)

    Because rows in support must give the EXACT SAME payoff v,
    we can solve the equalities using linear equations.

    The unknowns are:
        - probabilities p over support S
        - probabilities q over support T
        - the game value v

    Once a FEASIBLE solution satisfies all inequalities,
    we have found the Nash equilibrium.
    """

    A = np.array(A, dtype=float)
    m, n = A.shape

    # Utility function: normalize probability vectors
    def normalize_prob(x):
        """
        Ensures all probabilities are >= 0 and sum to 1.
        If impossible, return None.
        """
        x = np.array(x, dtype=float)
        x[x < 0] = 0
        s = x.sum()
        if s < tol:
            return None
        return x / s

    # Maximum support size = min(m, n)
    max_support_size = min(m, n)

    # Try support sizes 1, 2, ..., k
    for k in range(1, max_support_size + 1):

        # Enumerate all supports S ⊆ rows, |S| = k
        for S in itertools.combinations(range(m), k):

            # Enumerate all supports T ⊆ columns, |T| = k
            for T in itertools.combinations(range(n), k):

                S = list(S)
                T = list(T)

                # Submatrix A_ST (k × k)
                A_ST = A[np.ix_(S, T)]

                # --------------------------------------------------------
                # PART 1 — Solve for q_T and v so that rows in S tie at v
                # --------------------------------------------------------
                #
                # For each row i ∈ S:
                #       sum_{j∈T}  A[i,j] * q[j] = v
                #
                # Plus normalization:
                #       sum_{j∈T} q[j] = 1
                #
                # Unknowns: q_T (length k) and v → total k+1 unknowns.
                #
                # Write system in matrix form:
                #     [ A_ST    |  -1 ] [q] = [0]
                #     [  1      |   0 ] [v]   [1]
                #
                # This is (k+1) × (k+1).
                #

                M_q = np.zeros((k + 1, k + 1))
                b_q = np.zeros(k + 1)

                # Equality: A_ST * q - v = 0
                M_q[:k, :k] = A_ST
                M_q[:k, -1] = -1  # coefficient of v

                # Normalization: sum(q) = 1
                M_q[-1, :k] = 1
                b_q[-1] = 1

                # Solve for q_T and v
                try:
                    x_q = np.linalg.solve(M_q, b_q)
                except np.linalg.LinAlgError:
                    continue

                q_T = x_q[:k]
                v_q = x_q[-1]

                # Normalize q
                q_T = normalize_prob(q_T)
                if q_T is None:
                    continue

                # Build full q vector
                q = np.zeros(n)
                q[T] = q_T

                # Check row best-response inequalities
                row_payoffs = A @ q
                # Rows in S must equal v
                if not np.all(np.abs(row_payoffs[S] - v_q) <= 1e-5):
                    continue
                # Other rows must be ≤ v
                remaining_rows = [i for i in range(m) if i not in S]
                if remaining_rows:
                    if np.any(row_payoffs[remaining_rows] > v_q + 1e-5):
                        continue

                # --------------------------------------------------------
                # PART 2 — Solve for p_S and v so that columns in T tie
                # --------------------------------------------------------
                #
                # For each column j ∈ T:
                #       sum_{i∈S} p[i] A[i,j] = v
                #
                # Plus normalization:
                #       sum_{i∈S} p[i] = 1
                #
                # Unknowns: p_S (length k) and v → k+1 unknowns.
                #
                # System:
                #     [ A_ST^T | -1 ] [p_S] = [0]
                #     [   1    |  0 ] [ v ]   [1]
                #

                M_p = np.zeros((k + 1, k + 1))
                b_p = np.zeros(k + 1)

                # Column equalities: p^T A[:,j] = v
                # i.e. A_ST rows correspond to S
                M_p[:k, :k] = A_ST.T
                M_p[:k, -1] = -1

                # Normalization: sum(p) = 1
                M_p[-1, :k] = 1
                b_p[-1] = 1

                try:
                    x_p = np.linalg.solve(M_p, b_p)
                except np.linalg.LinAlgError:
                    continue

                p_S = x_p[:k]
                v_p = x_p[-1]

                p_S = normalize_prob(p_S)
                if p_S is None:
                    continue

                # v must be consistent
                if abs(v_p - v_q) > 1e-5:
                    continue

                v = 0.5 * (v_p + v_q)

                # Build full p
                p = np.zeros(m)
                p[S] = p_S

                # Check column inequalities
                col_payoffs = p @ A
                if not np.all(np.abs(col_payoffs[T] - v) <= 1e-5):
                    continue
                remaining_cols = [j for j in range(n) if j not in T]
                if remaining_cols:
                    if np.any(col_payoffs[remaining_cols] < v - 1e-5):
                        continue

                # If all checks pass → equilibrium found
                return p, q, float(v)

    raise ValueError("No equilibrium found via support enumeration (degenerate game?).")


# Test Cases:
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
    print()
    print()


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
print()
print()

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
print("Interpretation:")
print(f"Player 1 should play:")
print(f"  Move 1: {p[0]*100:.1f}%")
print(f"  Move 2: {p[1]*100:.1f}%")
print()
print(f"Player 2 should play:")
print(f"  Move 1: {q[0]*100:.1f}%")
print(f"  Move 2: {q[1]*100:.1f}%")
print()
print()

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
print("Interpretation:")
print(f"Player 1 should play:")
print(f"  Move 1: {p[0]*100:.1f}%")
print(f"  Move 2: {p[1]*100:.1f}%")
print(f"  Move 3: {p[2]*100:.1f}%")
print(f"  Move 4: {p[3]*100:.1f}%")
print()
print(f"Player 2 should play:")
print(f"  Move 1: {q[0]*100:.1f}%")
print(f"  Move 2: {q[1]*100:.1f}%")
print(f"  Move 3: {q[2]*100:.1f}%")
print(f"  Move 4: {q[3]*100:.1f}%")
print()
print()

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