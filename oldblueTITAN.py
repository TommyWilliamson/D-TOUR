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
import subprocess
import pandas as pd
import datetime as dt
from scipy import sparse
import pathlib

comm = MPI.COMM_WORLD
mpiRank = comm.Get_rank()
mpiSize = comm.Get_size()

RNG = RandomState(mpiRank)  # sets a different seed for each worker's random number generator
RNG.randn()

n_outputs = 2
# Useful class to suppress TITAN output when necessary
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
# overload BLUEProblem and create a new class, MyProblem.
class MyProblem(BLUEProblem):
    def sampler(self, ls):
        L = len(ls)
        mu_flight = -16
        sigma_flight = 0.1
        mu_heading = 45
        sigma_heading = 1.0
        flight_angle = RNG.normal(mu_flight,sigma_flight)
        heading_angle = RNG.normal(mu_heading,sigma_heading)

        params=np.array([flight_angle, heading_angle])
        samples = [np.array(params) for i in range(L)]
        return samples

    def evaluate(self, ls, samples):
        global cfg
        global allSamples
        L = len(ls)
        out = [[0 for i in range(L)] for n in range(n_outputs)]
        TITANcfg=cfg


        timeVectors = np.array([10, 20, 30, 50, 100])
        for i in range(L):
            deltaT=timeVectors[ls[i]]
            TITANcfg['Time']['Time_step'] = str(deltaT)

            directory = os.getcwd()

            if not os.path.exists(directory + '/DTOURtemp'):
                os.makedirs(directory + '/DTOURtemp')

            TITANcfg['Trajectory']['Flight_path_angle'] = str(samples[i][0])
            TITANcfg['Trajectory']['Heading_angle'] = str(samples[i][1])

            outputDir=directory + '/DTOURtemp/Output' + str(mpiRank)
            if os.path.exists(outputDir):
                TITANcfg['Options']['Load_mesh'] = 'True'
            TITANcfg['Options']['Output_folder'] = outputDir
            # Then write to a temp cfg file
            filepath = directory + '/DTOURtemp/BLUETemp'+str(mpiRank)+'.cfg'
            with open(filepath, 'w') as configfile:
                TITANcfg.write(configfile)
            if mpiRank==0:
                #print('Beginning sim of dt =',deltaT, 'approx duration of',dt.timedelta(seconds=400))
                subprocess.run(args=('python TITAN.py -c' + filepath), shell=True, stdout=subprocess.DEVNULL)
            else:
                subprocess.run(args=('python TITAN.py -c' + filepath), shell=True, stdout=subprocess.DEVNULL)
            data_csv = outputDir + '/Data/data.csv'

            with open(data_csv, 'r') as file:
                data = pd.read_csv(file)

            lat = data.loc[:, 'Latitude'].iat[-1]
            lon = data.loc[:, 'Longitude'].iat[-1]
            allSamples=sparse.vstack((allSamples,np.zeros(shape=(1,10))),format='lil')
            allSamples[-1,i] = lat
            allSamples[-1,i+L] = lon
            out[0][i] = lat



        return out




def runBLUE(cfgFile,budget):
    ## This function runs a TITAN sim through a BLUE analysis
    # in future a more robust system for handling which variables are uncertain will be developed
    # currently only altitude fragmentation triggers are considered.
    global cfg
    global allSamples

    # Additionally the multilevel models are defined solely by a vector of timesteps
    cfg = configparser.ConfigParser()
    cfg.read(cfgFile)
    # if mpiRank==0: subprocess.run(args=('python TITAN.py -c' + cfgFile), shell=True)
    # cfg['Options']['Load_mesh'] = 'False'
    # 5 models
    n_models = 5
    allSamples=sparse.lil_matrix(np.zeros(shape=(1, 2*n_models)))

    # Default verbose option is True for debugging, you can see everything that goes on
    # under the hood if you set it to True. Advised if something breaks!
    # 32 (or even 20) samples are typically enough for application runs. For debugging and Maths papers, set it to 1000.
    # These samples won't be re-used. Sample re-use introduces bias and is not implemented here yet.
    MPI.Comm.barrier(comm)
    problem = MyProblem(n_models,n_outputs=n_outputs, covariance_estimation_samples=20, verbose=True)

    MLBLUE_data = problem.setup_solver(K=n_models, budget=budget)
    costs = problem.get_costs()


    if mpiRank==0:
        print("\n\n\n BEGINNING BLUE SAMPLING \n")
        for key, item in MLBLUE_data.items(): print(key, ": ", item)
        print("\n\n")

        with open('BLUEdata.pkl', 'wb') as file:
            pickle.dump(MLBLUE_data, file)
        with open('costs.pkl', 'wb') as file:
            pickle.dump(costs, file)

    sol_MLBLUE = problem.solve(K=n_models,budget=budget)


    if mpiRank==0:
        print("\n\nFINSIHED\n")

        print("MLBLUE solution: ", sol_MLBLUE[0])
        print(sol_MLBLUE)
        with open('solutionTITAN.pkl','wb') as file: pickle.dump(sol_MLBLUE, file)
        with open('allSamplesTITAN.pkl','wb') as file: pickle.dump(allSamples, file)

