import numpy as np
from scipy import stats
from titanwrapper import buildKeplerian
from Astro import propagator
import random
import os
import swami
import configparser
from matplotlib import pyplot as plt
import pickle
import datetime as dt
import statistics

mcm = swami.MCM()
cfg = configparser.ConfigParser()

n_cores = 22

def vanillaMC(states,cfgfile,forecast):
    nSamples = max(states.shape) #note only work for n>6

    output = np.empty(shape=[6, nSamples])
    houtput = np.empty(shape=[nSamples])
    toutput = np.empty(shape=[nSamples])
    cost = []

    
    for i in range(nSamples):
        then=dt.datetime.now()
        stateOut, hOut, tOut, _, _ = propagator.runPropagator(state=states[:,i], forecast=forecast,
                                                                cfgfile=cfgfile)
        now=dt.datetime.now()
        deltaT=now-then

        output[:,i] = stateOut[-1,:]
        houtput[ i] = hOut[-1]
        toutput[i] = tOut[-1]
        if not i==0:
            cost.append(deltaT.seconds + 1e-6 * deltaT.microseconds)
            print('Cost of ',statistics.mean(cost))
        print('\n\n',100 * i / nSamples, '%\n\n')
    return output, houtput, toutput, cost
def getStats(inputfile,deltaT,nSamples):
    cfg.read(inputfile)
    cfg_orb = cfg
    cfg_orb['Propagator']['Prop_timestep'] = str(deltaT)
    cfg_orb['Propagator']['Progress_checkin'] = str(10)
    cfg_orb['Propagator']['Density_criterion'] = str(1e4)
    cfg_orb['Propagator']['Altitude_criterion'] = '2000'
    cfg_orb['Propagator']['Critical_altitude'] = '0'
    cfg_orb['Propagator']['Checkpoint_value'] = str(3e6)

    dir = os.getcwd()

    if not os.path.exists(dir + '/DTOURtemp'):
        os.makedirs(dir + '/DTOURtemp')

    # Then write to a temp cfg file
    filepath = dir + '/DTOURtemp/CVTemp.cfg'
    with open(filepath, 'w') as configfile:
        cfg_orb.write(configfile)

    eccentricity = 0.001
    argument_of_periapsis = 0.0
    inclination = 5
    mean_anomaly = 0.0
    forecast = 1000.0

    mu_SMA = 6521000
    sigma_SMA = 1000
    mu_LAN = 0.0
    sigma_LAN = 1.0

    states=np.empty(shape=[6,nSamples])
    alts = np.empty(shape=[nSamples])


    for i in range(nSamples):

        SMA = random.normalvariate(mu_SMA,sigma_SMA)
        LAN = random.normalvariate(mu_LAN,sigma_LAN)

        states[:,i] = buildKeplerian(SMA,eccentricity,argument_of_periapsis,LAN,inclination,mean_anomaly)
        alts[i] = np.linalg.norm(states[:3,i])
        print(100 * i / nSamples, '%')

    statesOut,h,t,cost = vanillaMC(states,filepath,forecast)
    return (states, statistics.mean(cost))




states120,cost120=getStats('controlVariates.cfg',120,50)
states60,cost60=getStats('controlVariates.cfg',60,50)
states10,cost10=getStats('controlVariates.cfg',10,50)
states1_n25, cost1_n25=getStats('controlVariates.cfg',1,25)
print('Costs...\ndt=120',cost120,'\ndt=60',cost60,'\ndt=10',cost10,'\ndt=1.0',cost1_n25)

with open('statistics120.pkl','wb') as file:
    pickle.dump(states120,file)
with open('statistics60.pkl','wb') as file:
    pickle.dump(states60,file)
with open('statistics10.pkl','wb') as file:
    pickle.dump(states10,file)
with open('statistics1.pkl','wb') as file:
    pickle.dump(states1_n25,file)