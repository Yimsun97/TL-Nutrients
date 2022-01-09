import numpy as np


def get_dispersion(U, B, h, J, method='seo'):
    methods = ['mcquivey', 'liu1', 'liu2', 'iwasa', 'seo']
    if method not in methods:
        raise NotImplementedError("Such method have not been implemented.")

    g = 9.81
    u_star = np.sqrt(g * h * J)
    if method == 'mcquivey':
        D = 0.058 * U * h / J
    elif method == 'liu1':
        D = 0.6 * u_star * (B * h) ** 2.0 / h ** 3.0
    elif method == 'liu2':
        D = 0.51 * u_star * (B * h) ** 2.0 / h ** 3.0
    elif method == 'iwasa':
        D = 2.0 * (B / h) ** 1.5 * h * u_star
    else:
        D = 5.915 * (B / h) ** 0.620 * (U / u_star) ** 1.428
    return D


def get_degradation(temp, di, delta_t, h, sim_var, p):
    if sim_var == 'NH3':
        var_in = di.df_interp(temp, 'Temp')
        t_sim_len = (di.t_sim_end - di.t_sim_start).days
        time_series = np.linspace(0, t_sim_len, int(t_sim_len / delta_t) + 1)
        temp_field = var_in(time_series)
        temp_field = np.tile(temp_field[:, np.newaxis], (1, h.shape[-1]))
        # coeff = p[0] * p[1] ** (temp_field - 20)
        coeff = p[0] * np.exp(p[1] * (temp_field - 20))
    else:
        coeff = p[0] / (h ** p[1])
    if np.any(np.isinf(coeff)):
        raise ValueError("Inf exists in coeff. Please check again.")
    return coeff
