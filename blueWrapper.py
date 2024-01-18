from blueTITAN import runBLUE
import os
import shutil
from mpi4py import MPI

cfgfile = 'Sphere.cfg'

comm = MPI.COMM_WORLD
mpiRank = comm.Get_rank()
mpiSize = comm.Get_size()

directory = os.getcwd()



if not directory[-6:] == '/TITAN':
    if os.path.exists('TITAN'):
        os.chdir('TITAN')
        directory = os.getcwd()
    else:
        print('Error TITAN Directory not found!')

directory = os.getcwd()
if mpiRank==0:
    if os.path.exists(directory + '/DTOURtemp'):
        print('Cleaning temp directory...')
        shutil.rmtree(directory + '/DTOURtemp')
        os.makedirs(directory + '/DTOURtemp')


MPI.Comm.barrier(comm)
runBLUE(cfgfile,1e5)