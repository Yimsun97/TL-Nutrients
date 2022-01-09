import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
from scipy import interpolate
from interpolation import DataInterp, SpatialInterp, load_interp


def data_reader():
    flow = pd.read_csv('data/Q.csv', header=0,
                       parse_dates=['Date'])

    station_list = ['malishu', 'fumin', 'xiaoyuba',
                    'shahe', 'tongxianqiao', 'shilongba']
    wq = dict()
    na_list = ['F', 'hd', 'Z', 'M', 'lw', 'B', 'T', 'D', 'lr', 'lp', 'L']
    for i in station_list:
        wq_ = pd.read_excel('data/WQ.xlsx', sheet_name=i, header=0, parse_dates=['time'])
        for j in wq_.columns:
            if j != 'time':
                wq_.loc[:, j] = pd.to_numeric(wq_.loc[:, j], errors='coerce').copy()

        # 设置监测数据合理下限值
        wq_.loc[wq_.NH3 < 0.10, 'NH3'] = np.nan
        wq_.loc[wq_.TP < 0.02, 'TP'] = np.nan

        wq[i] = wq_

    R = pd.read_excel('data/Reaches.xlsx', header=0, sheet_name='Reaches')
    U_to_R = pd.read_excel('data/Reaches.xlsx', header=0, sheet_name='HRU_to_Reaches')

    sheet_names = ['domestic_urban', 'domestic_rural', 'agriculture', 'livestock',
                   'industry', 'urban', 'total']
    # loads = {}
    # for name in sheet_names:
    #     loads[name] = pd.read_excel('data/Loads.xlsx', header=0, sheet_name=)
    loads = pd.read_excel('data/Loads.xlsx', header=0, sheet_name=sheet_names, index_col=0)

    runoff = pd.read_csv('data/PrecipRunoff.csv', header=0)
    temp = pd.read_csv('data/Temp.csv', header=0, parse_dates=['time'])
    precip = pd.read_csv('data/Precip.csv', header=0, parse_dates=['Date'])
    return flow, wq, R, U_to_R, loads, runoff, temp, precip


class GetField:
    def __init__(self, sim_var, delta_t, delta_x, ):
        self.sim_var = sim_var
        self.delta_t = delta_t
        self.delta_x = delta_x
        self.time_series = None
        self.xx = None

    def get_boundary_conditions(self, di, wq_df):
        # 生成边界条件
        var_in = di.df_interp(wq_df, self.sim_var)
        t_sim_len = (di.t_sim_end - di.t_sim_start).days
        time_series = np.linspace(0, t_sim_len, int(t_sim_len / self.delta_t) + 1)
        self.time_series = time_series
        return var_in(time_series)

    def get_initial_conditions(self, r_df, cs_locs, var_t0):
        # 生成初始条件
        x = np.array(cs_locs)
        y = np.array(var_t0)
        func_interp = interpolate.interp1d(
            x,
            y,
            kind='slinear'
        )
        xx = np.linspace(cs_locs[0], cs_locs[-1], int(r_df.L.sum() / self.delta_x) + 1)
        self.xx = xx
        return func_interp(xx)

    def get_fields(self, di, r_df, q_df, reach_runoff, reach_loads, rough=0.1):
        # 生成速度、负荷场
        unit_coeff_ux = 86400 / 1e3
        unit_coeff_sx = 1e3 / 365
        si = SpatialInterp(self.delta_x, r_df, di.df_interp(q_df, 'Q'), reach_runoff,
                           reach_loads.loc[:, self.sim_var], rough=rough)

        # print(vel_field.shape[0])

        def find_index(arr_1, arr_2):
            idx = np.zeros_like(arr_2, dtype=int)
            for i, value in enumerate(arr_2):
                idx[i] = np.argmin(np.abs(value - arr_1))
            return idx

        vel_field = np.zeros((len(self.time_series), len(self.xx)))
        b_field = np.zeros((len(self.time_series), len(self.xx)))
        h_field = np.zeros((len(self.time_series), len(self.xx)))
        j_field = np.zeros((len(self.time_series), len(self.xx)))
        s_field = np.zeros((len(self.time_series), len(self.xx)))
        bx_func = si.get_bx()
        jx_func = si.get_jx()
        for i, t in enumerate(self.time_series):
            hx_func, _, ux_func = si.get_haux(t)
            vel_field[i] = ux_func(self.xx) * unit_coeff_ux
            b_field[i] = bx_func(self.xx)
            h_field[i] = hx_func(self.xx)
            j_field[i] = jx_func(self.xx)
            sx_arr = si.get_sx(t)
            s_field[i] = 0
            idx = find_index(self.xx, sx_arr[:, 0])
            s_field[i, idx] = sx_arr[:, 1] * unit_coeff_sx
            # print(i)
        return vel_field, b_field, h_field, j_field, s_field


if __name__ == '__main__':

    from hyperdata import *

    field = GetField(sim_var, delta_t, delta_x)
    tp_bc = field.get_boundary_conditions(di, WQ['xiaoyuba'])
    tp_ic = field.get_initial_conditions(R, cs_locs, var_start)
    vel_field, b_field, h_field, j_field, s_field = field.get_fields(di, R, Q, reach_runoff, reach_loads, rough)
    np.save('results/bcs.npy', tp_bc)
    np.save('results/ics.npy', tp_ic)
    np.save('results/velocities.npy', vel_field)
    np.save('results/s.npy', s_field)

    # plt.figure()
    # plt.plot(Q, 'r.')
    # plt.show()

    # plt.figure()
    # plt.plot(WQ['malishu'].CODcr, 'b.')
    # plt.show()
    # plt.figure()
    # plt.plot(WQ['malishu'].NH3, 'b.')
    # plt.show()
    #
    # plt.figure()
    # plt.plot(WQ['malishu'].TP, 'C0', linestyle='dotted', label='Malishu')
    # plt.plot(WQ['fumin'].TP, 'C1', linestyle='dotted', label='Fumin')
    # plt.plot(WQ['xiaoyuba'].TP, 'C2', linestyle='dotted', label='Xiaoyuba')
    # plt.legend()
    # plt.show()
    #
    # plt.figure()
    # plt.plot(WQ['malishu'].TP.resample('D').mean(), 'C0', linestyle='dotted', label='Malishu')
    # plt.plot(WQ['fumin'].TP.resample('D').mean(), 'C1', linestyle='dotted', label='Fumin')
    # plt.plot(WQ['xiaoyuba'].TP.resample('D').mean(), 'C2', linestyle='dotted', label='Xiaoyuba')
    # plt.legend()
    # plt.xlim([datetime.date(2019, 6, 1), datetime.date(2019, 10, 1)])
    # plt.show()

    # plt.figure(); plt.plot(tp_ic); plt.show()
    # plt.figure(); plt.plot(tp_bc); plt.show()
    # plt.figure(); plt.imshow(vel_field); plt.show()
