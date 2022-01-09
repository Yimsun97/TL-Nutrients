import numpy as np
import pandas as pd


def get_loadings(loads, precip, di, delta_t, delta_x, cs_locs, h_field, b_field, r_df, sim_var):
    non_point = ['agriculture', 'urban']
    point = ['domestic_urban', 'domestic_rural', 'livestock', 'industry']
    loads['non_point'] = pd.DataFrame(np.zeros_like(list(loads.values())[0]),
                                      index=list(loads.values())[0].index,
                                      columns=list(loads.values())[0].columns)
    for s in non_point:
        loads['non_point'] += loads[s]

    loads['point'] = pd.DataFrame(np.zeros_like(list(loads.values())[0]),
                                  index=list(loads.values())[0].index,
                                  columns=list(loads.values())[0].columns)
    for s in point:
        loads['point'] += loads[s]

    t_sim_len = (di.t_sim_end - di.t_sim_start).days
    time_series = np.linspace(0, t_sim_len, int(t_sim_len / delta_t) + 1)
    precip_interp = di.df_interp(precip, 'Precip')
    precip_arr = precip_interp(time_series) / (1 / delta_t)

    def find_index(arr_1, arr_2):
        idx = np.zeros_like(arr_2, dtype=int)
        for i, value in enumerate(arr_2):
            idx[i] = np.argmin(np.abs(value - arr_1))
        return idx

    x_grids = np.concatenate([np.array([0]), r_df.L.values.cumsum()])
    x_ = x_grids[1:] - delta_x
    xx = np.linspace(cs_locs[0], cs_locs[-1], int(r_df.L.sum() / delta_x) + 1)

    unit_coeff_sx = 1e3 / 365
    idx = find_index(xx, x_grids[:-1])
    s_field = np.zeros_like(h_field)
    for i, t in enumerate(time_series):
        area = h_field[i, idx] * b_field[i, idx]
        loads_point = loads['point'].loc[:, sim_var].values * unit_coeff_sx
        loads_non_point = loads['non_point'].loc[:, sim_var].values * precip_arr[i] / precip.sum()[0] * 1e3
        s_field[i, idx] = (loads_point + loads_non_point) / (0.001 * area)

    return s_field
