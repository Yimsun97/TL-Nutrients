import numpy as np

from numba import jit


@jit
def tdma(a, b, c, d):
    """Solution of a linear system of algebraic equations with a
        tri-diagonal matrix of coefficients using the Thomas-algorithm.

    Args:
        a(np.array): an array containing lower diagonal (a[0] is not used)
        b(np.array): an array containing main diagonal
        c(np.array): an array containing lower diagonal (c[-1] is not used)
        d(np.array): right hand side of the system
    Returns:
        x(np.array): solution array of the system

    """
    a = a.copy()
    b = b.copy()
    c = c.copy()
    d = d.copy()

    n = len(b)
    x = np.zeros(n)

    # elimination:

    for k in range(1, n):
        q = a[k] / b[k - 1]
        b[k] = b[k] - c[k - 1] * q
        d[k] = d[k] - d[k - 1] * q

    # back substitution:

    q = d[n - 1] / b[n - 1]
    x[n - 1] = q

    for k in range(n - 2, -1, -1):
        q = (d[k] - c[k] * q) / b[k]
        x[k] = q

    return x


class FDM:
    def __init__(self, dt, dx, D, U, K, init_conds, bnd_conds, time, distance,
                 S, rules=False):
        self.dt = dt
        self.dx = dx
        self.D = D
        self.K = K
        self.init_conds = init_conds
        self.bnd_conds = bnd_conds
        self.time = time
        self.distance = distance
        self.n_time = int(time / dt) + 1
        self.n_distance = int(distance / dx) + 1

        if type(D) not in [float, int]:
            if D.shape != (self.n_time, self.n_distance):
                raise ValueError('Shape of D not matched.')
            self.D = D
        else:
            self.D = np.ones((self.n_time, self.n_distance)) * D
        if type(K) not in [float, int]:
            if K.shape != (self.n_time, self.n_distance):
                raise ValueError('Shape of K not matched.')
            self.K = K
        else:
            self.D = np.ones((self.n_time, self.n_distance)) * D
        if type(U) not in [float, int]:
            if U.shape != (self.n_time, self.n_distance):
                raise ValueError('Shape of U not matched.')
            self.U = U
        else:
            self.U = np.ones((self.n_time, self.n_distance)) * U

        if type(S) not in [float, int]:
            if S.shape != (self.n_time, self.n_distance):
                raise ValueError('Shape of S not matched.')
            self.S = S
        else:
            self.S = np.ones((self.n_time, self.n_distance)) * S

        if rules:
            # print rules
            print(f"Courant number (mean) is {self.Courant()}")
            print(f"Neumann number is {self.Neumann()}")

    def implicit(self):
        C = np.zeros((self.n_time, self.n_distance))

        # Assign initial conditions and boundary conditions
        C = self.__init_concentration(C)
        delta = np.zeros(self.n_distance - 1)

        # Iterate
        for j in range(1, self.n_time):
            alpha, beta, gamma, alpha_, beta_ = self.__get_coeff(j)
            # A = self.__get_diag_matrix(alpha, beta, gamma, alpha_, beta_)
            Alpha, Beta, Gamma = self.__get_tridiagonal(alpha, beta, gamma, alpha_, beta_)
            for i in range(self.n_distance - 1):
                cij_eff = 1 / self.dt - (self.U[j - 1, i + 1] + self.U[j - 1, i]) / 2 / self.dx
                cij_minus_eff = (self.U[j - 1, i + 1] + self.U[j - 1, i]) / 2 / self.dx \
                                - self.K[j - 1, i] / 2.0
                delta[i] = C[j - 1, i + 1] * cij_eff + C[j - 1, i] * cij_minus_eff + \
                           self.S[j - 1, i]
            delta[0] = delta[0] - alpha[0] * C[j, 0]
            # ct = np.linalg.solve(A, delta)
            ct = tdma(np.concatenate([np.array([0]), Alpha]),
                      Beta,
                      np.concatenate([Gamma, np.array([0])]),
                      delta)
            C[j, 1:] = ct
        return C

    def stable_solution(self):
        c0t = self.bnd_conds
        U = self.U
        D = self.D
        K = self.K
        x = np.arange(self.n_distance) * self.dx
        C = c0t * np.exp(self.U * x / 2 / D * (1 - np.sqrt(1 + 4 * K * D / U ** 2.0)))
        return C

    def __get_diag_matrix(self, alpha, beta, gamma, alpha_, beta_):
        rows, cols = np.indices((self.n_distance - 1, self.n_distance - 1))

        def kth_diag(k):
            row_vals = np.diag(rows, k=k)
            col_vals = np.diag(cols, k=k)
            return row_vals, col_vals

        diag_mat = np.zeros((self.n_distance - 1, self.n_distance - 1))
        a_idx = kth_diag(-1)
        b_idx = kth_diag(0)
        c_idx = kth_diag(1)
        Alpha, Beta, Gamma = self.__get_tridiagonal(alpha, beta, gamma, alpha_, beta_)
        diag_mat[a_idx[0], a_idx[1]] = Alpha
        diag_mat[b_idx[0], b_idx[1]] = Beta
        diag_mat[c_idx[0], c_idx[1]] = Gamma
        return diag_mat

    def __get_tridiagonal(self, alpha, beta, gamma, alpha_, beta_):
        Alpha = alpha[1:self.n_distance - 1]
        Alpha[-1] = alpha_
        Beta = beta[:self.n_distance - 1]
        Beta[-1] = beta_
        Gamma = gamma[:self.n_distance - 2]
        return Alpha, Beta, Gamma

    def __get_coeff(self, time):

        dt = self.dt
        dx = self.dx
        D = self.D[time]
        K = self.K[time]

        alpha = -D / dx ** 2.0
        beta = 1 / dt + 2.0 * D / dx ** 2.0 + K / 2.0
        gamma = -D / dx ** 2.0
        alpha_ = alpha[-1] - gamma[-1]
        beta_ = beta[-1] + 2.0 * gamma[-1]
        return alpha, beta, gamma, alpha_, beta_

    def __init_concentration(self, C):
        if isinstance(self.init_conds, np.ndarray):
            if self.init_conds.shape[0] != self.n_distance:
                raise ValueError(f"Length of time step is {self.init_conds.shape[0]},"
                                 f"but expect {self.n_distance}.")
        if isinstance(self.bnd_conds, np.ndarray):
            if self.bnd_conds.shape[0] != self.n_time:
                raise ValueError(f"Length of distance step is {self.bnd_conds.shape[0]},"
                                 f"but expect {self.n_time}.")

        if isinstance(self.bnd_conds, (int, float, np.ndarray)):
            C[:, 0] = self.bnd_conds
        else:
            raise TypeError("Type of init_conds should be int, float or numpy.ndarray.")
        if isinstance(self.init_conds, (int, float, np.ndarray)):
            C[0, :] = self.init_conds
        else:
            raise TypeError("Type of init_conds should be int, float or numpy.ndarray.")

        return C

    def Courant(self):
        return self.dt * np.mean(self.U) / self.dx

    def Neumann(self):
        return np.mean(self.D) * self.dt / self.dx ** 2.0
