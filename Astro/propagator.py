import configparser
import os
import math
import random
import numpy
import numpy as np
import swami
import matplotlib.pyplot as plt
from scipy import integrate
import datetime as dt
import pytz
import pandas as pd
import subprocess
global densPos
global dens
global epoch


mcm = swami.MCM()
cfg = configparser.ConfigParser()
# Define body variables (could update to handle other bodies in future)

densPos = np.array([0, 0, 0])


# Define atmospheric model options
densCrit = 1e9 # m Absolute distance between atmospheric model calls
#densCrit=0.0 # set to '0.0' to calculate density only once
hCrit = 5e3  # m Height distance between atmospheric model calls
mu = 3.986004418e14  # TODO implement body switching
rad = 6371.0 * 1000  # in m
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
    east = np.cross([0,0,1],unit_x)
    east /= np.linalg.norm(east)

    north = np.cross(unit_x,east)
    north /= np.linalg.norm(north)

    flight_path = 90- np.degrees(np.arccos((np.dot(unit_x, unit_v))))

    heading = math.asin(np.linalg.norm(np.cross(north,unit_v)))
    if np.dot(east,unit_v)>0:
        if np.dot(north,unit_v)>0:
            heading = math.degrees(heading)
        else:
            heading = 180-math.degrees(heading)
    else:
        if np.dot(north, unit_v) > 0:
            heading = -math.degrees(heading)
        else:
            heading = -180 + math.degrees(heading)




    return altitude, velocity, flight_path, heading, latitude, longitude

import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

##Define Equation of State
def satEoS(t, state):
    ## Inputs
    # t - time in seconds
    # state - 6 element row vector of form...
    # [x1,x2,x3,v1,v2,v3]

    ## Outputs
    # stateout - 6 element row vector of form...
    # [v1,v2,v3,a1,a2,a3]

    earth_mu = 3.986004418e14  # m^3s^−2 (G*M)
    earth_rad = 6356.7523 * 1000  # m

    recalculate = 0



    # Get satellite BC, initial model is very basic...
    if 'Checkpoint_state' in cfg['Propagator'] and not t==0.0:
        cfg['Options']['Load_mesh']='True'
        checkpointStr=cfg['Propagator']['Checkpoint_state'].replace('[', '').replace(']', '').replace('\n', ' ').split()
        # Convert the string values to floats and create a numpy array
        checkpointState = np.array([float(value) for value in checkpointStr])
        if numpy.array_equal(np.around(checkpointState,decimals=2),np.around(state,decimals=2)):
            recalculate = 1

    BC=computeBC(state,cfg,recalculate)

    # Pull x and v from state vector
    x = np.array(state[:3])
    v = np.array(state[3:])

    # Call atmospheric model
    rho = getDensity(x, t,  cfg)

    # Compute derivatives
    norm_x = np.linalg.norm(x)
    norm_v = np.linalg.norm(v)
    unit_x = x / norm_x
    unit_v = v / norm_v
    #print(norm_x-earth_rad)
    g = earth_mu / (norm_x ** 2)
    dyn_pressure = 0.5 * rho * norm_v**2
    pitch_axis = np.cross(-unit_x,-unit_v)
    yaw_axis = np.cross(pitch_axis,unit_v)
    #print(math.degrees(math.asin(np.linalg.norm(np.cross(-unit_x,yaw_axis)))))
    a_g = -g * unit_x
    a_d = dyn_pressure * (-unit_v * BC[0])
    a_l = dyn_pressure * (yaw_axis * BC[1])
    a_y = dyn_pressure * (pitch_axis * BC[2])

    a = a_g + a_d + a_l + a_y

    # Reconstruct state vector
    stateout = np.concatenate((v, a))

    return stateout


def getDensity(x, t, cfg):
    ## Inputs

    # x - xyz positon vector in ITRF space (m)
    # t - time since sim initialisation (s)
    # densCrit - m Absolute distance between atmospheric model calls
    # hCrit - m Height distance between atmospheric model calls
    # uqSwitch - determine how density is output, 0 : mean, 1 : +1sigma, 2 : -1sigma, 3 : normal sampling

    ## Outputs

    # dens - density in kgm^-3

    # Values which are inaccessible to EoS
    densCrit=float(cfg['Propagator']['Density_criterion'])
    hCrit=float(cfg['Propagator']['Altitude_criterion'])
    earth_rad = 6371.0 * 1000  # TODO implement body switching

    global densPos
    global dens
    if densCrit==0.0 and np.linalg.norm(densPos)!=0.0:
        return dens

    # Determine distance between density evaluation position and vehicle current position
    norm_dens = np.linalg.norm(x - densPos)
    h = np.linalg.norm(x) - earth_rad
    h_dens = np.linalg.norm(densPos) - earth_rad
    dh_dens = abs(h - h_dens)

    if h<0:
        print('Negative altitude in density code! h=',h/1000,'km')

    # Determine if density needs to be recalculated
    if norm_dens >= densCrit or dh_dens >= hCrit:
    #     if norm_dens >= densCrit:
    #         print('Calculating Density due to densCrit')
    #     else:
    #         print('Calculating Density due to hCrit')

        # Build inputs to DTM2020 model

        # Time data
        delta_t = dt.timedelta(seconds=t)
        time = dt.datetime.strptime(cfg['PreEntry']['Epoch'],'%Y/%m/%d/%H:%M:%S') + delta_t


        # Geodesic Coords
        x_u = x / np.linalg.norm(x)
        xy_u = np.array([x[0], x[1], 0]) / np.linalg.norm(np.array([x[0], x[1], 0]))

        latitude = np.arccos(np.clip(np.dot(x_u, xy_u), -1.0, 1.0))
        longitude = math.atan2(x[1], x[0])

        # Local Time
        if longitude < 0:
            timezone = pytz.timezone('Etc/GMT{0}'.format(int(longitude)))
        else:
            timezone = pytz.timezone('Etc/GMT+{0}'.format(int(longitude)))
        local_time = pytz.utc.localize(time).astimezone(timezone)


        # Run DTM2020
        out = mcm.run(
            altitude=h / 1000,
            latitude=latitude,
            longitude=longitude,
            local_time=local_time.hour,
            day_of_year=time.timetuple().tm_yday,
            f107=70, #TODO Determine accurate driver information
            f107m=69,
            kp1=1,
            kp2=1,
            get_uncertainty=True,
        )

        # Redefine density evaluation position and density...
        densPos = np.array(x)
        dens = out.dens
        uqSwitch=int(cfg['Propagator']['Atmosphere_UQ'])

        # Uncertainty Handling
        if uqSwitch == 1:
            if out.dens_std:
                dens += out.dens_std
            elif out.dens_unc:
                dens *= (1 + out.dens_unc)
        elif uqSwitch == 2:
            if out.dens_std:
                dens -= out.dens_std
            elif out.dens_unc:
                dens *= (1 - out.dens_unc)
        elif uqSwitch == 3:
            if out.dens_std:
                dens = random.normalvariate(dens,out.dens_std)
            elif out.dens_unc:
                dens = random.normalvariate(dens,dens*out.dens_unc)
    return dens
def computeBC(state,cfg,recalculate):
    ## Inputs
    # cfg['Ballistics']['Reference_area'] - wetted area in m^2 (needed for naive BC calc)
    # cfg['Ballistics']['Drag_coefficient'] - drag coefficient, assumed to be 2.2 if not specified
    # cfg['Ballistics']['BCMass'] - mass in kg
    # cfg['Ballistics']['Method'] - either 'NAIVE' or 'FMF'
    # cfg['Ballistics']['BC_fidelity'] - if FMF choose either 'High' or 'Low' #TODO determine hifi functionality

    # recalculate - bool, to recalculate BC if one already exists

    ## Outputs
    # BC - ballistic coefficient kgm^-2
    # TODO create actual vehicle class

    if 'Ballistic_override' in cfg['Ballistics']:
        BC = np.array([float(cfg['Ballistics']['Ballistic_override']),0,0])
        return BC

    if recalculate or (not 'BCD' or not 'BCL' or not 'BCY' in cfg['Ballistics']):
        print('Recalculating BC...')
        if cfg['Ballistics']['Method'] =='NAIVE':
            cfg['Ballistics']['Drag_coefficient'] = cfg['Ballistics']['Drag_coefficient'] if 'Drag_coefficient' in cfg[
                'Ballistics'] else '2.2'

            cfg['Ballistics']['Lift_coefficient'] = cfg['Ballistics']['Lift_coefficient'] if 'Lift_coefficient' in cfg[
                'Ballistics'] else '0.0'

            cfg['Ballistics']['Side_coefficient'] = cfg['Ballistics']['Side_coefficient'] if 'Side_coefficient' in cfg[
                'Ballistics'] else '0.0'

            BCD = (float(cfg['Ballistics']['Drag_coefficient']) * float(cfg['Ballistics']['Reference_area'])) / float(cfg['Ballistics']['BCMass'])

            BCL = (float(cfg['Ballistics']['Lift_coefficient']) * float(cfg['Ballistics']['Reference_area'])) / float(cfg['Ballistics']['BCMass'])

            BCY = (float(cfg['Ballistics']['Side_coefficient']) * float(cfg['Ballistics']['Reference_area'])) / float(cfg['Ballistics']['BCMass'])
            BC = np.array([BCD, BCL, BCY])
        elif cfg['Ballistics']['BC_fidelity']=='Low':
            BC=titanBC(state, cfg, 0)
        else:
            BC=titanBC(state,cfg, 1)
        cfg['Ballistics']['BCD'] = str(BC[0])
        cfg['Ballistics']['BCL'] = str(BC[1])
        cfg['Ballistics']['BCY'] = str(BC[2])
    else:
        BCD = float(cfg['Ballistics']['BCD'])
        BCL = float(cfg['Ballistics']['BCL'])
        BCY = float(cfg['Ballistics']['BCY'])
        BC = np.array([BCD, BCL, BCY])
    return BC

def titanBC(state,cfg,fidelitySwitch):
    ## Inputs
    # fidelitySwitch : bool, 0 for lofi, 1 for hifi
    ## We want to run a single TITAN iter and reconstruct from the output
    ## our ballistic coefficient

    ## To do this we must transfer over our necessary data to the drag calculator
    dragCfg = cfg

    # And then ensure we only run two Lo-Fi iters and that it writes to a specific directory
    dragCfg['Options']['Num_iters'] = '2'
    if fidelitySwitch:
        dragCfg['Options']['Fidelity'] = 'High'
    else:
        dragCfg['Options']['Fidelity'] = 'Low'

    dir = os.getcwd()

    if not dir[-6:]=='/TITAN':
        if os.path.exists('TITAN'):
            os.chdir('TITAN')
            dir = os.getcwd()
        else: print('Error TITAN Directory not found!')




    if not os.path.exists(dir + '/DTOURtemp'):
        os.makedirs(dir + '/DTOURtemp')

    dragCfg['Options']['Output_folder'] = dir + '/DTOURtemp/dragData'

    ## Reconstruct config file
    altitude, velocity, flight_path, heading, latitude, longitude = constructWind(state)

    dragCfg=cfg

    dragCfg['Trajectory']['Altitude']= str(altitude)
    dragCfg['Trajectory']['Velocity'] = str(velocity)
    dragCfg['Trajectory']['Flight_path_angle'] = str(flight_path)
    dragCfg['Trajectory']['Heading_angle'] = str(heading)
    dragCfg['Trajectory']['Latitude'] = str(latitude)
    dragCfg['Trajectory']['Longitude'] = str(longitude)

    # Then write to a temp cfg file
    filepath = dir + '/DTOURtemp/dragTemp.cfg'
    with open(filepath, 'w') as configfile:
        dragCfg.write(configfile)

    # Finally we can run the file through TITAN
    print('\n\nBeginning Drag Recalculation')
    with HiddenPrints():
        subprocess.run(args=('python TITAN.py -c' + filepath),shell=True, stdout=subprocess.DEVNULL)
    print('Finished calculation')
    # After running we must perform the actual BC recalculation:
    data_csv = dir + '/DTOURtemp/dragData/Data/data.csv'

    with open(data_csv, 'r') as file:
        data = pd.read_csv(file)

    # Pulling necessary data
    F_drag = data.loc[:, 'Drag'][1]
    F_lift = data.loc[:, 'Lift'][1]
    F_side = data.loc[:, 'Crosswind'][1]
    A_ref = data.loc[:, 'Aref'][1]
    rho = data.loc[:, 'Density'][1]
    v_inf = data.loc[:, 'Velocity'][1]
    mass = data.loc[:, 'Mass'][1]

    BCD = (0.5 * rho * v_inf ** 2) * (F_drag / mass)
    BCL = (0.5 * rho * v_inf ** 2) * (F_lift / mass)
    BCY = (0.5 * rho * v_inf ** 2) * (F_side / mass)

    BC = np.array([BCD, BCL, BCY])

    print('BC is ',BC,' (inverse of ',1 / BC, ')')
    return BC


def runPropagator(state,forecast,cfgfile):
    ## Inputs
    # epoch - a datetime object defining the start time of the sim
    # state - 6 element row vector of form...
    #         [x1,x2,x3,v1,v2,v3]
    # forecast - time in seconds from epoch to run the sim up to
    # cfgfile - string pointing to .cfg file
    # options - a dict of settings including 't_end' (final time) and 't_max' (max step size), optionally 'writeData' to save files as .csvs
    # neededData - a str defining what is returned by the propagator...
    # 'state' : the final state vector
    # 'statetime' : the final state vector and a deltatime object corresponding to the total sim time
    # 'plots' : show plots
    # 'full'  : all of the above (default)
    ## Outputs
    # stateSeries
    # hSeries
    # tSeries
    # entryFlag



    ## Initialise Sim...
    cfg.read(cfgfile)
    t_max=float(cfg['Propagator']['Prop_timestep'])

    earth_rad = 6371.0 * 1000 #TODO implement body switching

    # Begin sim
    print('Initialising Sim...')
    t1 = dt.datetime.now()  # for duration estimation

    # Call solver
    propagator = integrate.solve_ivp(fun=satEoS, t_span=(0, t_max), y0=state, method='RK45')

    x = state[:3]



    stateSeries = np.array([state, propagator.y[:, -1]])
    hSeries = np.array((np.linalg.norm(x) - earth_rad) / 1000, (np.linalg.norm(propagator.y[:3, -1]) - earth_rad) / 1000)
    tSeries = np.array(0,propagator.t[-1])
    norb=[]
    entryFlag = 0

    # Begin full sim
    i = 0
    progress=float(cfg['Propagator']['Progress_checkin'])
    h_checkpoint=0
    checkpointCriterion=float(cfg['Propagator']['Checkpoint_value'])

    while propagator.t[-1] < forecast:
        i += 1 # Increment iters
        state_new = propagator.y[:, -1]
        propagator = integrate.solve_ivp(fun=satEoS, t_span=(propagator.t[-1], propagator.t[-1] + t_max), y0=state_new, method='RK45') # call propagator
        h = (np.linalg.norm(propagator.y[:3, -1]) - earth_rad)  / 1000

        mu = 3.986004418e14
        v_inf = math.sqrt(mu / np.linalg.norm(propagator.y[:3, -1]))
        v = propagator.y[3:, -1]
        x = propagator.y[:3, -1]
        nonOrb = np.linalg.norm(np.cross((propagator.y[:3, -1] / np.linalg.norm(propagator.y[:3, -1])), propagator.y[3:, -1]))

        stateSeries = numpy.append(stateSeries, [propagator.y[:, -1]], axis=0)
        hSeries = numpy.append(hSeries, h)
        tSeries = numpy.append(tSeries, propagator.t[-1])
        norb.append(nonOrb-v_inf)

        if 1000*h > h_checkpoint+checkpointCriterion or 1000*h < h_checkpoint - checkpointCriterion:
            h_checkpoint=1000*h
            cfg['Propagator']['Checkpoint_state']=str(propagator.y[:,-1])
            delta_t = dt.timedelta(seconds=propagator.t[-1])
            epoch_checkpoint = dt.datetime.strptime(cfg['PreEntry']['Epoch'], '%Y/%m/%d/%H:%M:%S') + delta_t

            cfg['Propagator']['Checkpoint_epoch']=dt.datetime.strftime(epoch_checkpoint,format='%Y/%m/%d/%H:%M:%S')

        if h <float(cfg['Propagator']['Critical_altitude']):
            print('Low altitude, object has \'entered\' atmosphere h=',h,'km total duration of ',dt.timedelta(seconds=propagator.t[-1]))
            entryFlag=1

            if 'Plots' in cfg['Propagator']:
                # Set up a figure twice as tall as it is wide
                fig = plt.figure()

                fig.suptitle('Vehicle position and altitude over ' + str(dt.timedelta(seconds=propagator.t[-1])))

                # First subplot
                ax = fig.add_subplot(1, 2, 1)

                ax.plot(hSeries - 200.0)
                ax.set_ylabel('Altitude change in km')

                # Second subplot
                ax = fig.add_subplot(1, 2, 2)

                ax.plot(norb)

                plt.show()

            return stateSeries,hSeries,tSeries,entryFlag,cfg
        # Print Progress
        if i % progress == 0:

            print(100 * propagator.t[-1] / forecast, '%', 'h=',np.min(hSeries),'km non-orbityness=',nonOrb-v_inf)


    ## Output data and plot
    delta = dt.datetime.now() - t1
    print('Finished sim! (actual duration was', delta, ') \n h=',h,'km total duration of ',dt.timedelta(seconds=propagator.t[-1]))

    if 'Write_data' in cfg['Propagator']:
        pd.DataFrame(hSeries).to_csv(str('data/altitude_'+dt.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')+'.csv'))
        pd.DataFrame(stateSeries).to_csv(str('data/position_'+dt.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')+'.csv'))

    if 'Plots' in cfg['Propagator']:
        # Set up a figure twice as tall as it is wide
        fig = plt.figure()

        fig.suptitle('Vehicle position and altitude over '+str(dt.timedelta(seconds=propagator.t[-1])))

        # First subplot
        ax = fig.add_subplot(1, 2, 1)

        ax.plot(hSeries - 200.0)
        ax.set_ylabel('Altitude change in km')

        # Second subplot
        ax = fig.add_subplot(1, 2, 2)

        ax.plot(norb)

        plt.show()
    return stateSeries,hSeries,tSeries,entryFlag, cfg
