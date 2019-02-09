import torch
from sklearn.metrics import average_precision_score
import numpy as np
import os
import torch.nn as nn
import torch.utils.data as Data

from modules.GPVModel import Network_3D,DecoderNetwork
from dataprocess.dataprocess import TrainDataset
from tensorboardX import SummaryWriter

CWD_PATH = os.getcwd()
BATCH_SIZE = 20

def AutoTest():
    writer = SummaryWriter('log')
    Decoder = DecoderNetwork()
    Encoder = Network_3D()

    Decoder.load_state_dict(torch.load('./ModelDict/decoder_final.pkl'))
    Encoder.load_state_dict(torch.load('./ModelDict/encoder_final.pkl'))

    test_data = TrainDataset(CWD_PATH, loaded=True, test=True)
    testloader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    loss_func_1 = nn.BCELoss()

    if torch.cuda.is_available:
        Image_2D = Image_2D.cuda()
        Encoder = Encoder.cuda()
        Decoder = Decoder.cuda()
        loss_func_1 = loss_func_1.cuda()


    loss_collection = []
    for step, (image, voxel) in enumerate(testloader):
        if torch.cuda.is_available:
            image = image.cuda()
            voxel = voxel.cuda()
        feature_vector, decoder_input = Encoder(voxel)
        recon_voxel = Decoder(decoder_input)
        recon_voxel = recon_voxel.view(BATCH_SIZE, -1)
        real_voxel = voxel.view(BATCH_SIZE, -1)
        loss_1 = loss_func_1(recon_voxel, real_voxel)
        loss_collection.append(loss_1)

        try:
            ave_pre = evaliation(real_voxel.view(-1).cpu(), recon_voxel.view(-1).cpu())
        except BaseException:
            print('eval error')

        if step % 20 == 0:
            niter = step
            writer.add_scalar('Test/Loss_1', loss_1, niter)


            print('=============AUTOENCODER TEST==================')
            print('+++++++LOSS:' + str(loss_1.item()) + '+++++++')
            print('AP: %.5f' % ave_pre)

if __name__ == '__main__':
    AutoTest()