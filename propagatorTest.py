import configparser
from titanwrapper import buildKeplerian
import os
from Astro import propagator

cfg = configparser.ConfigParser()

inputfile='controlVariates.cfg'
cfg.read(inputfile)
cfg_orb = cfg
cfg_orb['Propagator']['Prop_timestep'] = '10'
cfg_orb['Propagator']['Progress_checkin'] = str(10)
cfg_orb['Propagator']['Density_criterion'] = str(1e4)
cfg_orb['Propagator']['Altitude_criterion'] = '2000'
cfg_orb['Propagator']['Critical_altitude'] = '0'
cfg_orb['Propagator']['Checkpoint_value'] = str(1e5)

dir = os.getcwd()

if not os.path.exists(dir + '/DTOURtemp'):
    os.makedirs(dir + '/DTOURtemp')

# Then write to a temp cfg file
filepath = dir + '/DTOURtemp/CVTemp.cfg'
with open(filepath, 'w') as configfile:
    cfg_orb.write(configfile)

eccentricity = 0.0001
argument_of_periapsis = 0.0
inclination = 5
mean_anomaly = 0.0
forecast = 1000.0
SMA = 6521000
LAN = 0.0

state = buildKeplerian(SMA,eccentricity,argument_of_periapsis,LAN,inclination,mean_anomaly)

stateOut, hOut, tOut, _, _ = propagator.runPropagator(state=state, forecast=forecast,
                                                                cfgfile=filepath)