import numpy as np
import pandas as pd

from scipy import interpolate
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
from datetime import datetime


# %%
class DataInterp:

    def __init__(self, t_sim_start, t_sim_end, t_interp_start,
                 method='slinear', date_format='%Y/%m/%d'):

        if method not in ["nearest", "zero", "slinear",
                          "quadratic", "cubic"]:
            raise ValueError('No such interpolation type.')

        self.t_sim_start = datetime.strptime(t_sim_start, date_format)
        self.t_sim_end = datetime.strptime(t_sim_end, date_format)
        self.t_interp_start = datetime.strptime(t_interp_start, date_format)
        self.method = method
        self.date_format = date_format

    def _date_transformer(self, date):
        if isinstance(date, str):
            date = datetime.strptime(date, self.date_format)
        return (date - self.t_sim_start).days

    def data_timelim(self, dat_in):
        t_start = self.t_interp_start
        t_end = self.t_sim_end

        if np.dtype('<M8[ns]') not in dat_in.dtypes.to_list():
            raise ValueError('No date data.')

        time_idx = dat_in.dtypes == np.dtype('<M8[ns]')
        time_colname = [x for x, y in time_idx.to_dict().items() if y is True]
        time = dat_in.loc[:, time_colname[0]]
        time.apply(lambda x: self._date_transformer(x))

        dat_out = dat_in[((time <= t_end) & (time >= t_start)).values].copy()
        return dat_out.reset_index()

    def array_interp(self, x_p, y_p):

        if x_p.dtype == np.dtype('<M8[ns]'):
            if x_p[0] > self.t_interp_start:
                raise ValueError('Interpolation start time should not be earlier than the first observed data.')
            else:
                x_start = x_p.iloc[0]
                x_p = (x_p - x_start).dt.total_seconds() / 86400

        func_interp = interpolate.interp1d(x_p, y_p, kind=self.method)
        return func_interp

    def df_interp(self, df_in, sim_var, method='slinear'):

        # 调整参数method的值
        if method != 'slinear':
            self.method = method

        # 先对输入数据框作裁剪
        df_in = self.data_timelim(df_in)

        # 再对数据框作插值
        time_idx = df_in.dtypes == np.dtype('<M8[ns]')
        time_name = [x for x, y in time_idx.to_dict().items() if y is True]

        # 如果有nan值，去掉后插值
        df = df_in.loc[:, [time_name[0], sim_var]].dropna(subset=[sim_var])
        output = self.array_interp(df.loc[:, time_name[0]], df.loc[:, sim_var])

        return output


class SpatialInterp:
    def __init__(self, delta_x, r_df, q_in, reach_runoff, reach_loads, rough):
        self.delta_x = delta_x
        self.r_df = r_df
        self.q_in = q_in
        self.reach_runoff = reach_runoff
        self.reach_loads = reach_loads
        self.rough = rough
        self.x_grids = np.concatenate([np.array([0]), r_df.L.values.cumsum()])

    def get_qx(self, time):
        qx0 = self.q_in(time)
        runoff_ = self.reach_runoff.values.cumsum().ravel() * 1e4 / 86400 / 365
        func_interp = interpolate.interp1d(
            self.x_grids,
            np.concatenate([np.array([0]), runoff_]),
            kind='next'
        )
        return lambda x: func_interp(x) + qx0

    def get_bx(self):
        b = self.r_df.B.values
        bx = interpolate.interp1d(
            self.x_grids,
            np.concatenate([np.array([b[0]]), b]),
            kind='next'
        )
        return bx

    def get_jx(self):
        j = self.r_df.J.values
        jx = interpolate.interp1d(
            self.x_grids,
            np.concatenate([np.array([j[0]]), j]),
            kind='next'
        )
        return jx

    def get_haux(self, time):
        q = self.get_qx(time)
        b = self.get_bx()
        j = self.get_jx()

        # h = np.vectorize(lambda x: self.solve_manning(q(x), b(x), j(x)))
        h = lambda x: (self.rough * q(x) / b(x) / j(x) ** (1.0 / 2)) ** (3.0 / 5)
        area = lambda x: b(x) * h(x)
        velocity = lambda x: q(x) / area(x)
        return h, area, velocity

    def solve_manning(self, q, b, j):
        def func(h):
            return (b * h / (b + 2 * h)) ** (2.0 / 3) * j ** (1.0 / 2) * b * h / self.rough - q
        depth = fsolve(func, np.array([0.1]))
        return depth

    def get_sx(self, time):
        h, a, u = self.get_haux(time)
        x_ = self.x_grids[1:] - self.delta_x
        areas_ = np.zeros_like(x_)
        for i in range(len(areas_)):
            areas_[i] = a(x_[i])
        loads_ = self.reach_loads.values / (0.001 * areas_)
        # sx = interpolate.interp1d(
        #     self.x_grids,
        #     np.concatenate([np.array([loads_[0]]), loads_]),
        #     kind='next'
        # )
        return np.column_stack([self.x_grids[:-1], loads_])


def load_interp(loads, u_to_r, r_df, sim_var):
    var_loads = loads.reset_index().loc[:, ['UID', sim_var]]
    merged = pd.merge(u_to_r, r_df, on='RID', how='left')
    merged.loc[:, sim_var] = 0
    for uid in var_loads.UID:
        loads_uid = var_loads.loc[var_loads.UID == uid, sim_var]
        reach_len_rid = merged.loc[merged.UID == uid, 'L'].values
        loads_rid = loads_uid.values / reach_len_rid.sum() * reach_len_rid
        merged.loc[merged.UID == uid, sim_var] = loads_rid
    output = merged.groupby('RID').sum().loc[:, [sim_var]]
    return output


if __name__ == '__main__':
    Q = pd.read_csv('data/Q.csv', index_col=0, parse_dates=['Date'], header=0)
    WQ = pd.read_excel('data/WQ.xlsx', index_col=0, parse_dates=['time'], header=0)
    R = pd.read_excel('data/Reaches.xlsx', header=0, sheet_name='Reaches')
    U_to_R = pd.read_excel('data/Reaches.xlsx', header=0, sheet_name='HRU_to_Reaches')
    loads = pd.read_csv('data/Loads.csv', header=0)
    runoff = pd.read_csv('data/PrecipRunoff.csv', header=0)

    t_sim_start = '2019/1/26'
    t_sim_end = '2019/10/22'
    t_interp_start = '2019/1/26'
    di = DataInterp(t_sim_start, t_sim_end, t_interp_start)

    reach_loads = load_interp(loads, U_to_R, R, 'TP')
    reach_runoff = load_interp(runoff, U_to_R, R, 'Runoff')
