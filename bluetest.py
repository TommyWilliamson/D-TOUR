from bluest import BLUEProblem
import numpy as np
import configparser
from titanwrapper import buildKeplerian
from Astro import propagator
import os
from mpi4py import MPI
from numpy.random import RandomState
import math
import pickle
import sys
from matplotlib import pyplot as plt
comm = MPI.COMM_WORLD
mpiRank = comm.Get_rank()
mpiSize = comm.Get_size()
print('RANK:',mpiRank,'\n')
RNG = RandomState(mpiRank)  # sets a different seed for each worker's random number generator
RNG.randn()

# Useful class to suppress TITAN output when necessary
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

cfg = configparser.ConfigParser()
# 5 models
n_models = 5

# overload BLUEProblem and create a new class, MyProblem.
class MyProblem(BLUEProblem):
    def sampler(self, ls):
        L = len(ls)
        mu_SMA = 6521000
        sigma_SMA = 1000
        mu_LAN = 0.0
        sigma_LAN = 1.0
        SMA = np.random.normal(mu_SMA,sigma_SMA)
        LAN = np.random.normal(mu_LAN,sigma_LAN)

        params=np.array([SMA,LAN])
        samples = [np.array(params) for i in range(L)]
        return samples

    def evaluate(self, ls, samples):
        eccentricity = 0.001
        argument_of_periapsis = 0.0
        inclination = 5
        mean_anomaly = 0.0
        forecast = 600.0

        L = len(ls)
        out = [[0 for i in range(L)]] # only one output, output 0

        cfg.read('controlVariates.cfg')
        cfg_orb = cfg
        cfg_orb['MPI']['Rank']=str(mpiRank)
        cfg_orb['Propagator']['Progress_checkin'] = str(50)

        cfg_orb['Propagator']['Critical_altitude'] = '0'
        cfg_orb['Propagator']['Checkpoint_value'] = str(3e6)

        for i in range(L):
            deltaT=math.factorial(ls[i]+1)
            cfg_orb['Propagator']['Prop_timestep'] = str(deltaT)

            cfg_orb['Propagator']['Density_criterion'] = str(1000**(ls[i]+1))
            cfg_orb['Propagator']['Altitude_criterion'] = '2000'

            dir = os.getcwd()

            if not os.path.exists(dir + '/DTOURtemp'):
                os.makedirs(dir + '/DTOURtemp')

            # Then write to a temp cfg file
            filepath = dir + '/DTOURtemp/CVTemp'+str(mpiRank)+'.cfg'
            with open(filepath, 'w') as configfile:
                cfg_orb.write(configfile)


            state = buildKeplerian(samples[i][0], eccentricity, argument_of_periapsis, samples[i][1], inclination, mean_anomaly)
            if ls[i]<1:
                with HiddenPrints():
                    _, hOut, tOut, _, _ = propagator.runPropagator(state=state, forecast=forecast, cfgfile=filepath)
            else:
                _, hOut, tOut, _, _ = propagator.runPropagator(state=state, forecast=forecast, cfgfile=filepath)
            out[0][i] = hOut[-1] # high-fidelity model corresponding to model index 0



        return out


# Default verbose option is True for debugging, you can see everything that goes on
# under the hood if you set it to True. Advised if something breaks!
# 32 (or even 20) samples are typically enough for application runs. For debugging and Maths papers, set it to 1000.
# These samples won't be re-used. Sample re-use introduces bias and is not implemented here yet.
problem = MyProblem(n_models, covariance_estimation_samples=20, verbose=True)


# define budget
budget=5e4

MLMC_data = problem.setup_mlmc( budget=budget)
MFMC_data = problem.setup_mfmc( budget=budget)
MLBLUE_data = problem.setup_solver(K=n_models, budget=budget)

if mpiRank==0:
    print("Covariance matrix:\n")
    print(problem.get_covariance())
    print("\nCorrelation matrix:\n")
    print(problem.get_correlation())
    print("\nCost vector:\n")
    print(problem.get_costs())

    #

    # MLMC
    print("\n\nMLMC\n")
    for key, item in MLMC_data.items(): print(key, ": ", item)
    print("\n\nMFMC\n")
    for key, item in MFMC_data.items(): print(key, ": ", item)
    print("\n\nMLBLUE\n")
    for key, item in MLBLUE_data.items(): print(key, ": ", item)

sol_MLBLUE = problem.solve(K=n_models,budget=budget)


if mpiRank==0:
    print("\n\nMLBLUE\n")

    print("MLBLUE solution: ", sol_MLBLUE[0])

    print(sol_MLBLUE)

    with open('solution.pkl','wb') as file:
        pickle.dump(sol_MLBLUE, file)


# MLBLUE is more sensitive than the other methods to integer projection,
# always good to check all methods. This does not require any sampling.
    print("\nCost comparison. MLMC: %f, MFMC: %f, MLBLUE: %f" % (MLMC_data["total_cost"], MFMC_data["total_cost"], MLBLUE_data["total_cost"]))


