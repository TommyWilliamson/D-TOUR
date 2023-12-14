import configparser
import os
import math
import numpy as np
import datetime as dt
from Astro import propagator
import pandas as pd
import matplotlib.pyplot as plt

mu = 3.986004418e14  # TODO implement body switching
rad = 6371.0 * 1000  # in m

cfg = configparser.ConfigParser()


def constructWind(state):
    # Need to go from state vector (ECEF) to wind coords...

    # Lat and long are easy enough...
    x = state[:3]
    v = state[3:]

    altitude = np.linalg.norm(np.linalg.norm(x) - rad)

    unit_x = x / np.linalg.norm(x)
    xy_u = np.array([x[0], x[1], 0]) / np.linalg.norm(np.array([x[0], x[1], 0]))

    latitude = np.degrees(np.arccos(np.dot(unit_x, xy_u)))
    longitude = np.degrees(math.atan2(x[1], x[0]))

    velocity = np.linalg.norm(v)

    # Flight path angle is very close to 0.0
    unit_v = v / velocity

    # Describe northward and eastward vectors...
    east = np.cross([0, 0, 1], unit_x)
    east /= np.linalg.norm(east)

    north = np.cross(unit_x, east)
    north /= np.linalg.norm(north)

    flight_path = 90 - np.degrees(np.arccos((np.dot(unit_x, unit_v))))

    heading = math.asin(np.linalg.norm(np.cross(north, unit_v)))
    if np.dot(east, unit_v) > 0:
        if np.dot(north, unit_v) > 0:
            heading = math.degrees(heading)
        else:
            heading = 180 - math.degrees(heading)
    else:
        if np.dot(north, unit_v) > 0:
            heading = -math.degrees(heading)
        else:
            heading = -180 + math.degrees(heading)

    return altitude, velocity, flight_path, heading, latitude, longitude


inputfile = 'test.cfg'

dir = os.getcwd()

cfg.read(inputfile)

new_cfg = cfg

data_csv = dir + '/checkpoint.csv'

with open(data_csv, 'r') as file:
    state = pd.read_csv(file)
state = state.to_numpy()

inputState = state[:, 1]

filepath = dir + '/new.cfg'

new_cfg['Propagator']['Density_criterion'] = str(1e5)
new_cfg['Propagator']['Progress_checkin'] = str(10)
new_cfg['Propagator']['Critical_altitude'] = '90'
new_cfg['Propagator']['Prop_timestep'] = '1.0'
new_cfg['Propagator']['Progress_checkin'] = str(100)
new_cfg['Propagator']['Density_criterion'] = str(1e5)
new_cfg['Propagator']['Altitude_criterion'] = '500'
new_cfg['Propagator']['Checkpoint_value'] = '10000'

with open(filepath, 'w') as configfile:
    new_cfg.write(configfile)

stateBC, hBC, tBC, entryFlag3, BC_cfg = propagator.runPropagator(state=inputState, forecast=1e6,
                                                                 cfgfile=filepath)

altitude, velocity, flight_path, heading, latitude, longitude = constructWind(stateBC[-1, :])
#
cfg_BC = new_cfg
#
#
cfg_BC['Trajectory']['Altitude'] = str(altitude)
cfg_BC['Trajectory']['Velocity'] = str(velocity)
cfg_BC['Trajectory']['Flight_path_angle'] = str(flight_path)
cfg_BC['Trajectory']['Heading_angle'] = str(heading)
cfg_BC['Trajectory']['Latitude'] = str(latitude)
cfg_BC['Trajectory']['Longitude'] = str(longitude)
cfg_BC['Options']['Output_folder'] = 'BC'

dir = os.getcwd()

if not dir[-6:] == '/TITAN':
    if os.path.exists('TITAN'):
        os.chdir('TITAN')
        dir = os.getcwd()
    else:
        print('Error TITAN Directory not found!')

titanfile = dir + '/TITAN_' + inputfile

with open(titanfile, 'w') as configfile:
    cfg_BC.write(configfile)

print('\n Handing over to TITAN...\n')
os.system('python TITAN.py -c' + titanfile)

new_cfg['Ballistics']['Method'] = 'NAIVE'

with open(filepath, 'w') as configfile:
    new_cfg.write(configfile)

stateNAIVE, hNAIVE, tNAIVE, entryFlag3, NAIVE_cfg = propagator.runPropagator(state=inputState, forecast=1e6,
                                                                             cfgfile=filepath)
altitude, velocity, flight_path, heading, latitude, longitude = constructWind(stateNAIVE[-1, :])

cfg_titan = cfg

cfg_titan['Trajectory']['Altitude'] = str(altitude)
cfg_titan['Trajectory']['Velocity'] = str(velocity)
cfg_titan['Trajectory']['Flight_path_angle'] = str(flight_path)
cfg_titan['Trajectory']['Heading_angle'] = str(heading)
cfg_titan['Trajectory']['Latitude'] = str(latitude)
cfg_titan['Trajectory']['Longitude'] = str(longitude)
cfg_titan['Options']['Output_folder'] = 'NAIVE'

dir = os.getcwd()

if not dir[-6:] == '/TITAN':
    if os.path.exists('TITAN'):
        os.chdir('TITAN')
        dir = os.getcwd()
    else:
        print('Error TITAN Directory not found!')

titanfile = dir + '/TITAN_' + inputfile

with open(titanfile, 'w') as configfile:
    cfg_titan.write(configfile)

print('\n Handing over to TITAN...\n')
os.system('python TITAN.py -c' + titanfile)

print('Finished!')
# Let's check the result
data_csv = dir + '/' + cfg_titan['Options']['Output_folder'] + '/Data/data.csv'

with open(data_csv, 'r') as file:
    dataNAIVE = pd.read_csv(file)

data_csv = dir + '/' + cfg_BC['Options']['Output_folder'] + '/Data/data.csv'

with open(data_csv, 'r') as file:
    dataBC = pd.read_csv(file)

# Pulling necessary data
altNAIVE = dataNAIVE.loc[:, 'Altitude'] / 1000
altBC = dataBC.loc[:, 'Altitude'] / 1000
tNAIVE2 = dataNAIVE.loc[:, 'Time']
tBC2 = dataBC.loc[:, 'Time']

# alt.shift(len(hHifi))
newAltBC = pd.Series([0])
j = 0
for i in range(len(hBC) + len(altBC)):
    if i < len(hBC):
        newAltBC = pd.concat((newAltBC, pd.Series(0)))
    else:
        newAltBC = pd.concat((newAltBC, pd.Series(altBC[j])))
        j += 1

newAltNAIVE = pd.Series([0])
j = 0
for i in range(len(hNAIVE) + len(altNAIVE)):
    if i < len(hNAIVE):
        newAltNAIVE = pd.concat((newAltNAIVE, pd.Series(0)))
    else:
        newAltNAIVE = pd.concat((newAltNAIVE, pd.Series(altNAIVE[j])))
        j += 1

fig = plt.figure()

# First subplot
ax = fig.add_subplot(1, 1, 1)

ax.plot(tNAIVE, hNAIVE, label='Naive Drag Orbital')
ax.plot(tNAIVE2, newAltNAIVE, label='TITAN Continued from Naive Drag')

ax.plot(tBC, hBC, label='FMF Drag Orbital')
ax.plot(tBC2, newAltBC, label='TITAN Continued from FMF Drag')
ax.set_ylabel('Altitude change in km')
ax.set_xlabel('Time in s')
