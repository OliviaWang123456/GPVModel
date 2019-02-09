import scipy.io as sio
import numpy as np
import src.voxelize.transformAndVisVoxels as trans

def main():
    data = trans.readModel('vo.mat')
    sio.savemat('voxel_after.mat', {'voxel':data})

if __name__ == '__main__':
    main()
