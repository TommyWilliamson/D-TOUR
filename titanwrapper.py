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

def buildKeplerian(SMA,eccentricity,omega,OMEGA,inclination,mean_anomaly):
    o=math.radians(omega)
    O=math.radians(OMEGA)
    i=math.radians(inclination)
    M=math.radians(mean_anomaly)

    semiminor = math.sqrt(SMA**2-(SMA**2)*eccentricity**2)

    #print('Minimum alt of',(semiminor-rad)/1000)




    E=M
    eplot=[]
    for j in range(10000):
        E-=(E-eccentricity*math.sin(E)-M)/(1-eccentricity*math.cos(E))
        eplot.append(E)

    true_anomaly=2*math.atan2(math.sqrt(1+eccentricity)*math.sin(E/2),math.sqrt(1-eccentricity)*math.cos(E/2))

    dist=SMA*(1-eccentricity*math.cos(E))
    vel=math.sqrt(mu*SMA)/dist

    x_inPlane=np.array([dist*math.cos(true_anomaly),dist*math.sin(true_anomaly),0])
    v_inPlane=np.array([-vel*math.sin(E),vel*math.sqrt(1-eccentricity**2)*math.cos(E),0])

    ## Rotation
    inclination_matrix = np.array([
        [1, 0, 0],
        [0, math.cos(-i), -math.sin(-i)],
        [0, math.sin(-i), math.cos(-i)],
    ])

    OMEGA_matrix = np.array([
        [math.cos(-O), -math.sin(-O), 0],
        [math.sin(-O), math.cos(-O), 0],
        [0, 0, 1]
    ])

    omega_matrix = np.array([
        [math.cos(-o), -math.sin(-o), 0],
        [math.sin(-o), math.cos(-o), 0],
        [0, 0, 1]
    ])

    x=np.dot(OMEGA_matrix,x_inPlane)
    x=np.dot(inclination_matrix,x)
    x=np.dot(omega_matrix,x)

    v=np.dot(OMEGA_matrix,v_inPlane)
    v=np.dot(inclination_matrix,v)
    v=np.dot(omega_matrix,v)

    state = np.concatenate((x,v))
    return state


def buildCircular(body, altitude, lat, lon, inclination):


    R = rad + altitude
    v_inf = math.sqrt(mu / R)
    period = 2 * math.pi * math.sqrt(R ** 3 / mu)

    x = np.empty(3)  # In-plane position


    x[0] = R * math.cos(math.radians(lat)) * math.cos(math.radians(lon))
    x[1] = R * math.cos(math.radians(lat)) * math.sin(math.radians(lon))
    x[2] = R * math.sin(math.radians(lat))

    # Inclination rotation matrix
    # inclination_matrix = np.array([
    #     [math.cos(math.radians(inclination)), -math.sin(math.radians(inclination)), 0],
    #     [math.sin(math.radians(inclination)), math.cos(math.radians(inclination)), 0],
    #     [0, 0, 1]
    # ])
    inclination_matrix = np.array([
        [1, 0, 0],
        [0, math.cos(math.radians(inclination)), -math.sin(math.radians(inclination))],
        [0, math.sin(math.radians(inclination)), math.cos(math.radians(inclination))],
    ])

    # In-plane velocity
    v = np.array([-v_inf * math.sin(math.radians(lon)), v_inf * math.cos(math.radians(lon)), 0])

    # Apply inclination rotation
    x_ecef = np.dot(inclination_matrix, x)
    v_ecef = np.dot(inclination_matrix, v)

    # Combine position and velocity into the state vector
    state = np.concatenate([x_ecef, v_ecef])
    return state, period

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


cfg = configparser.ConfigParser()

### Initialise by reading config file configparser.read()
def runWrapper(inputfile):
    inputfile='test.cfg'
    cfg.read(inputfile)
    os.system('cat logo.txt')
    print('Digital-Twin for Orbital Uncertain and Re-entry - D-T.O.U.R - Aerospace Centre for Excellence, Strathclyde')

    try:
        if cfg['PreEntry']['Orbit_construct'] == 'CIRCULAR':
            state, period = buildCircular(body='',
                                          altitude=float(cfg['PreEntry']['Orbit_altitude']),
                                          lat = float(cfg['PreEntry']['Orbit_latitude']),
                                          lon = float(cfg['PreEntry']['Orbit_longitude']),
                                          inclination = float(cfg['PreEntry']['Orbit_inclination'])
                                          )
        if cfg['PreEntry']['Orbit_construct'] == 'STATE':
            state = cfg['PreEntry']['Orbit_state']
        if cfg['PreEntry']['Orbit_construct'] == 'KEPLERIAN':
            state = buildKeplerian(SMA=float(cfg['Orbital']['Semimajor_axis']),
                                          eccentricity=float(cfg['Orbital']['Eccentricity']),
                                          omega=float(cfg['Orbital']['Argument_of_periapsis']),
                                          OMEGA=float(cfg['Orbital']['Longitude_AN']),
                                          inclination=float(cfg['Orbital']['Inclination']),
                                          mean_anomaly=float(cfg['Orbital']['Mean_anomaly'])
                                          )
        Epoch = cfg['PreEntry']['Epoch']
        forecast = cfg['PreEntry']['Forecast_days']

    except configparser.Error as e:
        print(f"Error reading configuration file: {e}")

    Epoch = dt.datetime.strptime(Epoch,'%Y/%m/%d/%H:%M:%S')
    forecast = dt.timedelta(days=float(forecast),seconds=0.1).total_seconds()
    constructWind(state)
    ## Run ad infinitum orbital sim, constant atmosphere model

    cfg_orb=cfg
    cfg_orb['Propagator']['Prop_timestep']= '100.0'
    cfg_orb['Propagator']['Progress_checkin']=str(1e3)
    cfg_orb['Propagator']['Density_criterion']= '0.0'
    cfg_orb['Propagator']['Altitude_criterion']= '1000'
    cfg_orb['Propagator']['Critical_altitude']= '110'

    dir = os.getcwd()

    if not os.path.exists(dir + '/DTOURtemp'):
        os.makedirs(dir + '/DTOURtemp')

    # Then write to a temp cfg file
    filepath = dir + '/DTOURtemp/infTemp.cfg'
    with open(filepath, 'w') as configfile:
        cfg_orb.write(configfile)
    print('\n Running forecast simulation to determine if re-entry occurs in the near future\n')
    #stateConst,hConst,tConst,entryFlag1, cfg_orb=propagator.runPropagator(state=state,forecast=forecast,cfgfile=filepath)

    ## Re-run with adaptive atmosphere if entry occurs within infinitum
    entryFlag1=1
    if entryFlag1:
        cfg_orb['Propagator']['Density_criterion'] = str(1e5)
        cfg_orb['Propagator']['Progress_checkin'] = str(10)
        cfg_orb['Propagator']['Critical_altitude'] = '110'

        with open(filepath, 'w') as configfile:
            cfg_orb.write(configfile)

        print('\n Entry occurred! Running adaptive atmopshere simulation to better evaluate re-entry\n')

        stateAdpt, hAdpt, tAdpt, entryFlag2, cfg_orb = propagator.runPropagator(state=state, forecast=forecast,
                                                                            cfgfile=filepath)
        ## Re-run from final checkpoint at higher fidelity
        cfg_orb['PreEntry']['Epoch']=cfg_orb['Propagator']['Checkpoint_epoch']
        cfg_orb['Propagator']['Prop_timestep'] = '1.0'
        cfg_orb['Propagator']['Progress_checkin'] = str(100)
        cfg_orb['Propagator']['Density_criterion'] = str(1e5)
        cfg_orb['Propagator']['Altitude_criterion'] = '500'
        cfg_orb['Propagator']['Checkpoint_value']= '500'


        with open(filepath, 'w') as configfile:
            cfg_orb.write(configfile)

        print('\n Running \'final approach\' simulation to better hand over to TITAN\n')

        checkpointStr = cfg_orb['Propagator']['Checkpoint_state'].replace('[', '').replace(']', '').replace('\n', '').split()
        # Convert the string values to floats and create a numpy array
        checkpointState = np.array([float(value) for value in checkpointStr])
        pd.DataFrame(checkpointState).to_csv('checkpoint.csv')
        stateHifi, hHifi, tHifi, entryFlag3, cfg_orb = propagator.runPropagator(state=checkpointState, forecast=forecast,
                                                                                cfgfile=filepath)

        ## Exit based on acceleration or height-based triggers

        ## Reconstruct config file
        altitude, velocity, flight_path, heading, latitude, longitude = constructWind(stateHifi[-1,:])

        cfg_titan=cfg

        cfg_titan['Trajectory']['Altitude']= str(altitude)
        cfg_titan['Trajectory']['Velocity'] = str(velocity)
        cfg_titan['Trajectory']['Flight_path_angle'] = str(flight_path)
        cfg_titan['Trajectory']['Heading_angle'] = str(heading)
        cfg_titan['Trajectory']['Latitude'] = str(latitude)
        cfg_titan['Trajectory']['Longitude'] = str(longitude)

        dir = os.getcwd()

        if not dir[-6:] == '/TITAN':
            if os.path.exists('TITAN'):
                os.chdir('TITAN')
                dir = os.getcwd()
            else:
                print('Error TITAN Directory not found!')

        titanfile=dir + '/TITAN_' + inputfile

        with open(titanfile, 'w') as configfile:
            cfg_titan.write(configfile)

        print('\n Handing over to TITAN...\n')
        os.system('python TITAN.py -c' + titanfile)

        print('Finished!')
        # Let's check the result
        data_csv = dir + '/'+cfg_titan['Options']['Output_folder']+'/Data/data.csv'

        with open(data_csv, 'r') as file:
            data = pd.read_csv(file)

        # Pulling necessary data
        x = data.loc[:, 'ECEF_X']
        y = data.loc[:, 'ECEF_Y']
        z = data.loc[:, 'ECEF_Z']
        alt = data.loc[:, 'Altitude']/1000
        t = data.loc[:, 'Time']

        #alt.shift(len(hHifi))
        newAlt=pd.Series([0])
        j = 0
        for i in range(len(hHifi)+len(alt)):
            if i < len(hHifi):
                newAlt = pd.concat((newAlt,pd.Series(0)))
            else:
                newAlt = pd.concat((newAlt,pd.Series(alt[j])))
                j+=1

        trail=pd.Series(data=range(len(hHifi)))*0

        alt2=pd.concat((trail,alt))
        t=t+tHifi[-1]

        fig = plt.figure()

        # First subplot
        ax = fig.add_subplot(1, 2, 1)

        ax.plot(tHifi,hHifi)
        ax.plot(t,alt)
        ax.set_ylabel('Altitude change in km')
        ax.set_xlabel('Time in s')

        # Second subplot
        ax = fig.add_subplot(1, 2, 2, projection='3d')

        xEarth=[]
        yEarth=[]
        zEarth=[]

        for lat in np.arange(-90, 90, 1):
            for lon in np.arange(0, 360, 1):
                xEarth.append(rad*math.cos(math.radians(lat)) * math.cos(math.radians(lon)))
                yEarth.append(rad*math.cos(math.radians(lat)) * math.sin(math.radians(lon)))
                zEarth.append(rad*math.sin(math.radians(lat)))

        ax.set_aspect('equal', 'box')
        ax.plot3D(xEarth, yEarth, zEarth,linewidth=0.1,color='0.5')
        ax.plot3D(x,y,z,color='r',linewidth=2)
        ax.plot3D(stateHifi[:,0],stateHifi[:,1],stateHifi[:,2],color='0.6')
        ax.plot3D(stateAdpt[:, 0], stateAdpt[:, 1], stateAdpt[:, 2],color='0.4')

        plt.show()

        fig2=plt.figure()

        ax2=fig2.add_subplot()

    else:
        print('\n Entry did not occur within specified forecast window, exiting...')









## Run TITAN
