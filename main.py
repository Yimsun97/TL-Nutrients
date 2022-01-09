import numpy as np

from hyperdata import *
from coefficients import get_dispersion, get_degradation
from finite_difference import FDM
from loadings import get_loadings
from postprocess import *
from calibration import parameter_define
from uncertainty import *
import warnings

warnings.filterwarnings("ignore")
np.random.seed(1024)

rough = 0.07
vel_field, b_field, h_field, j_field, s_field_const = field.get_fields(di, R, Q, reach_runoff, reach_loads, rough)
s_field_var = get_loadings(loads, precip, di, delta_t, delta_x, cs_locs, h_field, b_field, R, sim_var)

plot_velocity(vel_field, fig_name='results/velocity.png')
plot_ts_variable(s_field_const[:, 0])
plot_ts_variable(s_field_var[:, 0])
# plot_ts_variable(h_field[:, 0] * b_field[:, 0])
# plot_ts_variable(precip.values[:, 1])

# np.save('results/bcs.npy', tp_bc)
# np.save('results/ics.npy', tp_ic)
# np.save('results/velocities.npy', vel_field)
# np.save('results/s.npy', s_field_const)
# tp_bc = np.load('results/bcs.npy')
# tp_ic = np.load('results/ics.npy')
# vel_field = np.load('results/velocities.npy')
# s_field_const = np.load('results/s.npy')

# D = get_dispersion(vel_field, b_field, h_field, j_field)
# fdm = FDM(delta_t, delta_x, D, vel_field, K, tp_ic, tp_bc, T, L, S=s_field_const)
#
# import time
# start_time = time.time()
# C_im = fdm.implicit()
# print(time.time() - start_time)
#
# # C_stable = fdm.stable_solution()
# df_C_im = process_array(C_im, delta_x, t_sim_start, t_sim_end)
# image_plot(C_im, extent_x=(0.0, L), extent_y=(0.0, T))
# time_series_plot(C_im, int(cs_locs[1] / delta_x))
# time_series_plot(C_im, int(cs_locs[-1] / delta_x))


def pde_calibration(p, eval=False):
    vel_field, b_field, h_field, j_field, s_field = field.get_fields(di, R, Q, reach_runoff, reach_loads, rough)
    s_field = get_loadings(loads, precip, di, delta_t, delta_x, cs_locs, h_field, b_field, R, sim_var)
    D = get_dispersion(vel_field, b_field, h_field, j_field)
    K = get_degradation(temp, di, delta_t, h_field, sim_var, p)
    fdm = FDM(delta_t, delta_x, D, vel_field, K, tp_ic, tp_bc, T, L, S=s_field)
    C_im = fdm.implicit()
    df_C_im = process_array(C_im, delta_x, t_sim_start, t_sim_end)
    # 此处匹配处理
    idx = [np.argmin(np.abs(x - df_C_im.columns)) for x in cs_locs[1:]]
    # match_conds = df_C_im.columns.isin(np.around(cs_locs[1:], decimals=1))
    df_C_matched = df_C_im.iloc[:, idx]
    df_C_fumin = WQ['fumin'].set_index('time')
    df_C_malishu = WQ['malishu'].set_index('time')

    diff_fumin = (df_C_fumin.loc[:, sim_var] - df_C_matched.iloc[:, 0].rename(sim_var))
    diff_malishu = (df_C_malishu.loc[:, sim_var] - df_C_matched.iloc[:, 1].rename(sim_var))

    # 升尺度
    diff_fumin = diff_fumin.resample('D').mean().dropna().copy()
    diff_malishu = diff_malishu.resample('D').mean().dropna().copy()

    obj = np.sum(diff_fumin.values ** 2.0) + np.sum(diff_malishu.values ** 2.0)
    obj = np.sqrt(obj / (len(diff_fumin) + len(diff_malishu)))
    obj = np.min([obj, 1e4])
    if eval:
        fumin = pd.concat([df_C_fumin.loc[:, sim_var], df_C_matched.iloc[:, 0]], axis=1).dropna()
        fumin.columns = ['obs', 'pred']
        malishu = pd.concat([df_C_malishu.loc[:, sim_var], df_C_matched.iloc[:, 1]], axis=1).dropna()
        malishu.columns = ['obs', 'pred']
        return fumin, malishu
    else:
        return obj


# D = 20 * 24.0
# K = 0.03 * 24.0
n_dim = 2
lb = [0.01, 1.0] if sim_var == 'TP' else [1e-5, 1e-5]
ub = [10, 3.0] if sim_var == 'TP' else [5.0, 1.0]

# # %%
# n_group = 30
# n_iter = 15
# pco = 0.7
# pm = 0.01
# ga = parameter_define(pde_calibration, n_dim, lb, ub, n_group, n_iter, pco, pm)
#
# # time_series_plot(C_im, int(cs_locs[1] / delta_x))
# # time_series_plot(C_im, int(cs_locs[-1] / delta_x))
#
# # ga.BestIndi.Phen[0]
# # xx = np.array([6.69060538, 0.09355986])
# fumin, malishu = pde_calibration(ga.BestIndi.Phen[0], eval=True)
# obs_vs_pred(fumin.obs, fumin.pred, sim_var, [0, 2], f'results/{sim_var}_fumin.png')
# obs_vs_pred(malishu.obs, malishu.pred, sim_var, [0, 2], f'results/{sim_var}_malishu.png')
# print(f"NSE of CS fumin: {NSE(fumin.obs, fumin.pred)}.")
# print(f"NSE of CS malishu: {NSE(malishu.obs, malishu.pred)}.")

# %%
n_samples = 500
res_sceua, spot = spotpy_optimizier(pde_calibration, n_dim, lb, ub, n_samples)
sp.analyser.plot_parametertrace(res_sceua, fig_name=f'results/ua/params_trace_{sim_var}.png')
sp.analyser.plot_posterior_parametertrace(res_sceua, fig_name=f'results/ua/params_post_{sim_var}.png')
posterior = sp.analyser.get_posterior(res_sceua)
sp.analyser.plot_parameterInteraction(posterior, fig_name=f'results/ua/params_inter_{sim_var}.png')

fig, ax = plt.subplots(n_dim, 2, figsize=(10, 8))
parameters = sp.parameter.get_parameters_array(spot)
for par_id in range(len(parameters)):
    sp.analyser.plot_parameter_trace(ax[par_id][0], res_sceua, parameters[par_id])
    # plot_posterior_parameter_histogram(ax[par_id][1], res_sceua, parameters[par_id])
    ax[par_id][1].hist(res_sceua['par' + parameters[par_id]['name']])
ax[-1][0].set_xlabel('Iterations')
ax[-1][1].set_xlabel('Parameter range')
plt.show()
fig.savefig(f'results/ua/params_hist_{sim_var}.png', dpi=300, bbox_inches='tight')

plot_posterior_time_series(res_sceua, pde_calibration, 95,
                           sim_var, fig_path='results/ua/')

best_params = sp.analyser.get_best_parameterset(res_sceua, maximize=False)
best_f, best_m = pde_calibration(best_params[0], eval=True)
print(f"NSE of CS fumin: {NSE(best_f.obs, best_f.pred)}.")
print(f"NSE of CS malishu: {NSE(best_m.obs, best_m.pred)}.")
print(f"RMSE of CS fumin: {RMSE(best_f.obs, best_f.pred)}.")
print(f"RMSE of CS malishu: {RMSE(best_m.obs, best_m.pred)}.")

print(f"alpha lower CI: {np.percentile(res_sceua['paralpha'], 2.5)}")
print(f"alpha upper CI: {np.percentile(res_sceua['paralpha'], 97.5)}")
print(f"beta lower CI: {np.percentile(res_sceua['parbeta'], 2.5)}")
print(f"beta upper CI {np.percentile(res_sceua['parbeta'], 97.5)}")
