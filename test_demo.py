import scipy.io as sio
import numpy as np
import src.voxelize.transformAndVisVoxels as trans
import json

def main():
    # data = trans.readModel('vo.mat')
    # sio.savemat('voxel_after.mat', {'voxel':data})
    with open('pix3d.json', 'r') as f:
        dataSets = json.load(f)
        print(len(dataSets))
        data = dataSets[0]
        print(data.keys())

if __name__ == '__main__':
    main()
