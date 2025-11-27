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
