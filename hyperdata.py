from interpolation import DataInterp, SpatialInterp, load_interp
from preprocess import data_reader, GetField

Q, WQ, R, U_to_R, loads, runoff, temp, precip = data_reader()

sim_var = 'TP'
# t_sim_start = '2019/1/26'
t_sim_start = '2019/2/24'
t_sim_end = '2019/10/22'
t_interp_start = t_sim_start
delta_t = 1.0 / 6
delta_x = 0.1
# D = 20 * 24.0
K = 0.02 * 24.0
U = 1.78
cs_locs = [0, 4.97, R.L.sum()]
var_start = [0.03, 0.35, 0.22] if sim_var == 'NH3' else [0.221, 0.437, 0.524]
rough = 0.02

di = DataInterp(t_sim_start, t_sim_end, t_interp_start)
reach_runoff = load_interp(runoff, U_to_R, R, 'Runoff')
reach_loads = load_interp(loads['total'], U_to_R, R, sim_var)

# Calculate concentration
T = (di.t_sim_end - di.t_sim_start).days
L = cs_locs[-1] - cs_locs[0]

field = GetField(sim_var, delta_t, delta_x)
tp_bc = field.get_boundary_conditions(di, WQ['xiaoyuba'])
tp_ic = field.get_initial_conditions(R, cs_locs, var_start)
