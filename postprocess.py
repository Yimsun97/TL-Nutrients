import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from sklearn.metrics import r2_score
import spotpy as sp
import seaborn as sns


def process_array(image_data, dx, sim_start, sim_end):
    n_rows = image_data.shape[0]
    n_cols = image_data.shape[1]

    # index = np.arange(n_rows) * dt
    index = pd.date_range(sim_start, sim_end, periods=n_rows)
    column = np.arange(n_cols) * dx
    data_process = pd.DataFrame(image_data, index=index, columns=column)
    return data_process


def time_series_plot(C, cs_id):
    plt.figure()
    C_ = C[:, cs_id]
    plt.plot(C_)
    plt.xlabel("t")
    plt.show()


def cross_section_plot(C, ts_id):
    plt.figure()
    C_ = C[ts_id, :]
    plt.plot(C_)
    plt.xlabel("x")
    plt.show()


def method_comp_plot(C_ex_df, C_im_df, C_pde_df=None, ts_id=None, cs_id=None):
    if ts_id is not None:
        plt.figure()
        plt.plot(C_ex_df.columns, C_ex_df.iloc[ts_id, :], label="Explicit method")
        plt.plot(C_im_df.columns, C_im_df.iloc[ts_id, :], label="Implicit method")
        if C_pde_df is not None:
            plt.plot(C_pde_df.columns, C_pde_df.iloc[ts_id, :], label="pde package")
        plt.xlabel("x")

    if cs_id is not None:
        plt.figure()
        plt.plot(C_ex_df.index, C_ex_df.iloc[:, cs_id], label="Explicit method")
        plt.plot(C_im_df.index, C_im_df.iloc[:, cs_id], label="Implicit method")
        if C_pde_df is not None:
            plt.plot(C_pde_df.index, C_pde_df.iloc[:, cs_id], label="pde package")
        plt.xlabel("t")

    plt.legend()
    plt.show()


def image_plot(C, extent_x, extent_y):
    plt.figure()

    extent = np.r_[extent_x, extent_y]
    plt.imshow(C, extent=extent, origin="lower", )
    # adjust some settings
    plt.xlabel("x")
    plt.ylabel("Time")
    plt.xlim(extent_x)
    plt.ylim(extent_y)
    plt.colorbar()
    plt.gca().set_aspect('auto')
    plt.show()


def animation_plot(C_ex_df, C_im_df, extent_x, extent_y,
                   C_pde_df=None, filename=None):
    fig, ax = plt.subplots()

    x = C_ex_df.columns.values

    def animate(i):
        plt.cla()
        ax.plot(x, C_ex_df.iloc[i, :].values, label="Explicit method")
        ax.plot(x, C_im_df.iloc[i, :].values, label="Implicit method")
        if C_pde_df is not None:
            ax.plot(x, C_pde_df.iloc[i, :].values, label="pde package")
        ax.text(0.1, 0.1, f"{C_ex_df.index[i]: .2f} hours", transform=ax.transAxes)
        ax.set_xlim(extent_x)
        ax.set_ylim(extent_y)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("COD (mg/L)")
        ax.legend(loc=0)

    # Init only required for blitting to give a clean slate.
    def init():
        ax.plot(x, C_ex_df.iloc[0, :].values, label="Explicit method")
        ax.plot(x, C_im_df.iloc[0, :].values, label="Implicit method")
        if C_pde_df is not None:
            ax.plot(x, C_pde_df.iloc[0, :].values, label="pde package")
        ax.text(0.1, 0.1, f"{C_ex_df.index[0]: .2f} hours", transform=ax.transAxes)
        ax.set_xlim(extent_x)
        ax.set_ylim(extent_y)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("COD (mg/L)")
        ax.legend(loc=0)

    ani = animation.FuncAnimation(fig, animate, C_ex_df.shape[0],
                                  init_func=init,
                                  interval=250,
                                  # blit=True
                                  )
    ax.set_xlim(extent_x)
    ax.set_ylim(extent_y)
    plt.draw()
    plt.show()
    if filename:
        ani.save(filename)


def stable_comp(x, c_stable, c_changing):
    plt.figure()
    plt.plot(x, c_stable, 'r*', label="Stable solution")
    plt.plot(x, c_changing, label="25 h COD of implicit method")
    plt.xlabel("x (km)")
    plt.ylabel("COD (mg/L)")
    plt.legend()
    plt.show()


def plot_velocity(u, fig_name=None):
    unit_coeff_ux = 86400 / 1e3
    u_mean = u.mean(axis=1) / unit_coeff_ux
    u_max = u.max(axis=1) / unit_coeff_ux
    u_min = u.min(axis=1) / unit_coeff_ux
    plt.figure()
    plt.plot(u_mean, label='Mean Velocity')
    plt.plot(u_max, label='Max Velocity')
    plt.plot(u_min, label='Min Velocity')
    plt.xlabel('Time (No. of time steps)')
    plt.ylabel('U (m/s)')
    plt.legend()
    if fig_name:
        plt.savefig(fig_name, dpi=500, bbox_inches='tight')

    plt.show()


def ga_vis(res, fig_name=None):
    plt.figure()
    plt.plot(res.log['gen'], res.log['f_min'])
    plt.xlabel('Number of Generation')
    plt.ylabel('Value of Objective Function')

    if fig_name:
        plt.savefig(fig_name, dpi=500, bbox_inches='tight')

    plt.show()
    print('f_min / mean(f_min)= {}'.format(res.log['f_min'][-1] /
                                           np.mean(res.log['f_min'])))


def obs_vs_pred(obs, pred, sim_var='TP', ylim=None, fig_path=None):
    plt.figure(figsize=(20, 5))
    plt.plot(obs, 'b.', label='Observed')
    plt.plot(pred, 'C1', label='Predicted')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel(f'{sim_var} (mg/L)')
    if ylim is not None:
        plt.ylim(ylim)
    if fig_path:
        plt.savefig(fig_path, dpi=300)
    plt.show()


def NSE(obs, pred):
    return r2_score(obs, pred)


def RMSE(obs, pred):
    rmse = np.sqrt(np.sum((obs - pred) ** 2.0) / len(obs))
    return rmse


def ua_iteration(res_ua, fig_path=None):
    fig = plt.figure(figsize=(9, 5))
    plt.plot(res_ua['like1'])
    plt.show()
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    if fig_path:
        fig.savefig(fig_path, dpi=300)


def plot_posterior_parameter_histogram(ax, results, parameter):
    # This function is the last 100 runs
    # ax.hist(results['par' + parameter['name']][-100:],
    #         # bins=5
    #         )
    sns.displot(x=results['par' + parameter['name']][-100:], ax=ax, kind='kde')
    # ax.set_ylabel('Density')
    # ax.set_xlim(parameter['minbound'], parameter['maxbound'])


def get_posterior(results, percentage=10, maximize=True):
    """
    Get the best XX% of your result array (e.g. best 10% model runs would be a threshold setting of 0.9)

    :results: Expects an numpy array which should have as first axis an index "like1". This will be sorted .
    :type: array

    :percentage: Optional, ratio of values that will be deleted.
    :type: float

    :maximize: If True (default), higher "like1" column values are assumed to be better.
               If False, lower "like1" column values are assumed to be better.

    :return: Posterior result array
    :rtype: array
    """
    if maximize:
        index = np.where(results['like1'] >= np.percentile(results['like1'], 100.0 - percentage))
    else:
        index = np.where(results['like1'] <= np.percentile(results['like1'], percentage))
    return results[index]


def plot_posterior_time_series(res, pde, percentage, sim_var='TP', figsize=(16, 9),
                               fig_path='results/ua/'):
    best_params = sp.analyser.get_best_parameterset(res, maximize=False)
    best_f, best_m = pde(best_params[0], eval=True)
    post = get_posterior(res, percentage=percentage, maximize=False)
    post_params = sp.analyser.get_parameters(post)
    f = []
    m = []
    for i, res_ in enumerate(post_params):
        f_, m_ = pde(res_, eval=True)
        f.append(f_.pred.rename(f'{i}'))
        m.append(m_.pred.rename(f'{i}'))

    f_cat = pd.concat(f, axis=1)
    m_cat = pd.concat(m, axis=1)
    f_cat.loc[:, 'Max'] = f_cat.max(axis=1)
    f_cat.loc[:, 'Min'] = f_cat.min(axis=1)
    m_cat.loc[:, 'Max'] = m_cat.max(axis=1)
    m_cat.loc[:, 'Min'] = m_cat.min(axis=1)

    plt.rcParams.update({'font.size': 18})
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(1, 1, 1)
    ax.plot(best_f.index, best_f.obs, 'b.', label='Observed', alpha=0.5)
    ax.plot(best_f.index, best_f.pred, 'C1', label='Predicted', alpha=0.7)
    ax.fill_between(f_cat.index, f_cat.Max, f_cat.Min,
                    facecolor='grey',
                    alpha=0.9,
                    # zorder=0,
                    # linewidth=0,
                    label=f'{percentage}% CI', )
    ax.set_ylim(0, 2)
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{sim_var} (mg/L)')
    ax.legend()
    plt.show()
    fig.savefig(f'{fig_path}/ts_fumin_{sim_var}.png', dpi=600, bbox_inches='tight')

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(1, 1, 1)
    ax.plot(best_m.index, best_m.obs, 'b.', label='Observed', alpha=0.5)
    ax.plot(best_m.index, best_m.pred, 'C1', label='Predicted', alpha=0.7)
    ax.fill_between(m_cat.index, m_cat.Max, m_cat.Min,
                    facecolor='grey',
                    alpha=0.9,
                    # zorder=0,
                    # linewidth=0,
                    label=f'{percentage}% CI')
    ax.set_ylim(0, 2)
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{sim_var} (mg/L)')
    ax.legend()
    plt.show()
    fig.savefig(f'{fig_path}/ts_malishu_{sim_var}.png', dpi=600, bbox_inches='tight')


def plot_ts_variable(vars):
    plt.figure()
    plt.plot(vars)
    plt.xlabel('No. of time steps')
    plt.ylabel('S (mg/L/d)')
    plt.show()


def plot_scenarios(fumin, malishu, fumin_s, malishu_s, sim_var='TP', figsize=(16, 9),
                   fig_path='results/ua/'):
    plt.rcParams.update({'font.size': 18})
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(1, 1, 1)
    ax.plot(fumin.index, fumin.obs, 'b.', label='Observed', alpha=0.5)
    ax.plot(fumin.index, fumin.pred, 'C1', label='Predicted', alpha=0.7)
    ax.plot(fumin_s.index, fumin_s.pred, 'C2', label='Predicted', alpha=0.7)
    ax.set_ylim(0, 2)
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{sim_var} (mg/L)')
    ax.legend()
    plt.show()
    fig.savefig(f'{fig_path}/sc_fumin_{sim_var}.png', dpi=600)

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(1, 1, 1)
    ax.plot(malishu.index, malishu.obs, 'b.', label='Observed', alpha=0.5)
    ax.plot(malishu.index, malishu.pred, 'C1', label='Predicted', alpha=0.7)
    ax.plot(malishu_s.index, malishu_s.pred, 'C2', label='Predicted', alpha=0.7)
    ax.set_ylim(0, 2)
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{sim_var} (mg/L)')
    ax.legend()
    plt.show()
    fig.savefig(f'{fig_path}/sc_malishu_{sim_var}.png', dpi=600)


def plot_animation(C1, C2, extent_x, extent_y, sim_var, filename=None):
    fig, ax = plt.subplots()

    x = C1.columns.values

    def animate(i):
        plt.cla()
        ax.plot(x, C1.iloc[i, :].values, label="Baseline")
        ax.plot(x, C2.iloc[i, :].values, label="S1")
        ax.text(0.1, 0.1, f"{C1.index[i]}", transform=ax.transAxes)
        ax.set_xlim(extent_x)
        ax.set_ylim(extent_y)
        ax.set_xlabel("x (km)")
        ax.set_ylabel(f"{sim_var} (mg/L)")
        ax.legend(loc=0)

    # Init only required for blitting to give a clean slate.
    def init():
        ax.plot(x, C1.iloc[0, :].values, label="Baseline")
        ax.plot(x, C2.iloc[0, :].values, label="S1")
        ax.text(0.1, 0.1, f"{C1.index[0]}", transform=ax.transAxes)
        ax.set_xlim(extent_x)
        ax.set_ylim(extent_y)
        ax.set_xlabel("x (km)")
        ax.set_ylabel(f"{sim_var} (mg/L)")
        ax.legend(loc=0)

    ani = animation.FuncAnimation(fig, animate, C1.shape[0],
                                  init_func=init,
                                  interval=250,
                                  # blit=True
                                  )
    ax.set_xlim(extent_x)
    ax.set_ylim(extent_y)
    plt.draw()
    plt.show()
    if filename:
        ani.save(filename)
