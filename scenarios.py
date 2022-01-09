import numpy as np
from matplotlib import pyplot as plt

from hyperdata import *
from coefficients import get_dispersion, get_degradation
from finite_difference import FDM
from loadings import get_loadings
from postprocess import *
from uncertainty import *
import warnings

# warnings.filterwarnings("ignore")

rough = 0.07

vel_field, b_field, h_field, j_field, s_field = field.get_fields(di, R, Q, reach_runoff, reach_loads, rough)
s_field = get_loadings(loads, precip, di, delta_t, delta_x, cs_locs, h_field, b_field, R, sim_var)

vel_field, b_field, h_field, j_field, s_field_const = field.get_fields(di, R, Q, reach_runoff, reach_loads, rough)
s_field_var = get_loadings(loads, precip, di, delta_t, delta_x, cs_locs, h_field, b_field, R, sim_var)
plot_ts_variable(s_field_const[:, 0])
plot_ts_variable(s_field_var[:, 0])

if sim_var == 'TP':
    p = [1.19684, 1.00013]
else:
    p = [3.76763, 0.00083]

K = get_degradation(temp, di, delta_t, h_field, sim_var, p)


def scenario_time_series(S):
    D = get_dispersion(vel_field, b_field, h_field, j_field)
    fdm = FDM(delta_t, delta_x, D, vel_field, K, tp_ic, tp_bc, T, L, S=S)
    C_im = fdm.implicit()
    df_C_im = process_array(C_im, delta_x, t_sim_start, t_sim_end)
    # 此处匹配处理
    idx = [np.argmin(np.abs(x - df_C_im.columns)) for x in cs_locs[1:]]
    df_C_matched = df_C_im.iloc[:, idx]
    df_C_fumin = WQ['fumin'].set_index('time')
    df_C_malishu = WQ['malishu'].set_index('time')

    fumin = pd.concat([df_C_fumin.loc[:, sim_var], df_C_matched.iloc[:, 0]], axis=1).dropna()
    fumin.columns = ['obs', 'pred']
    malishu = pd.concat([df_C_malishu.loc[:, sim_var], df_C_matched.iloc[:, 1]], axis=1).dropna()
    malishu.columns = ['obs', 'pred']

    return df_C_im

# base line
base = scenario_time_series(s_field_var)
# s1
S = s_field * 10
s1 = scenario_time_series(s_field_const)

# plot_scenarios(fm, mls, fm_s, mls_s, sim_var=sim_var)
id = 714
plt.figure()
plt.plot(base.iloc[id], 'red', label='Baseline')
plt.plot(s1.iloc[id], 'blue', label='S1')
plt.xlabel('x (km)')
plt.ylabel(f'{sim_var} (mg/L)')
plt.title(f'Time: {base.iloc[id].name}')
plt.legend()
plt.show()

plot_animation(base, s1, [0, 36], [0, 1], sim_var, filename=f'results/sc_anim_{sim_var}.mp4')
