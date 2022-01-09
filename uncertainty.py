import numpy as np
import spotpy as sp
from spotpy.objectivefunctions import rmse


class spot_setup(object):
    """
    """

    def __init__(self, pde_func, lb, ub, obj_func=None):
        self.pde_func = pde_func
        self.obj_func = obj_func

        self.params = [
            sp.parameter.Uniform('alpha', lb[0], ub[0]),
            sp.parameter.Uniform('beta', lb[1], ub[1]),
        ]

    def parameters(self):
        return sp.parameter.generate(self.params)

    def simulation(self, vector):
        x = np.array(vector)
        simulations = [self.pde_func(x)]
        return simulations

    def evaluation(self):
        observations = [0]
        return observations

    def objectivefunction(self, simulation, evaluation, params=None):

        # SPOTPY expects to get one or multiple values back,
        # that define the performence of the model run
        if not self.obj_func:
            # This is used if not overwritten by user
            like = rmse(evaluation, simulation)
        else:
            # Way to ensure on flexible spot setup class
            like = self.obj_func(evaluation, simulation)
        return like


def spotpy_optimizier(opt_target, n_dim, lb, ub, n_samples):
    # Initialize your model with a setup file
    spot = spot_setup(opt_target, lb, ub)
    sampler = sp.algorithms.sceua(spot)
    sampler.sample(n_samples, ngs=n_dim+1)  # Run the model
    # sampler = sp.algorithms.lhs(spot)
    # sampler.sample(n_samples)  # Run the model
    res_sceua = sampler.getdata()  # Load the results
    return res_sceua, spot
