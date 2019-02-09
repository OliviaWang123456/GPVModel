import torch #ok
from sklearn.metrics import average_precision_score
import numpy as np
import os
import torch.nn as nn
import torch.utils.data as Data

from modules.GPVModel import Network_3D,DecoderNetwork
from torchvision.models.alexnet import alexnet
from dataprocess import TrainDataset
from tensorboardX import SummaryWriter

CWD_PATH = os.getcwd()
BATCH_SIZE = 20

def performanceTest():
    writer = SummaryWriter('ShapeNetTest')
    Decoder = DecoderNetwork()
    Encoder = Network_3D()

    my_alexnet = alexnet(pretrained=False)
    features = list(my_alexnet.classifier.children())[:-1]
    features.extend([nn.Linear(4096, 64)])
    my_alexnet.classifier = nn.Sequential(*features)
    Image_2D = my_alexnet

    Decoder.load_state_dict(torch.load('./ModelDict/decoder_final.pkl'))
    Encoder.load_state_dict(torch.load('./ModelDict/encoder_final.pkl'))
    Image_2D.load_state_dict(torch.load('./ModelDict/image_final.pkl'))

    test_data = TrainDataset(CWD_PATH, loaded=True, test=True)
    testloader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    loss_func_1 = nn.BCELoss()

    if torch.cuda.is_available:
        Image_2D = Image_2D.cuda()
        Encoder = Encoder.cuda()
        Decoder = Decoder.cuda()
        loss_func_1 = loss_func_1.cuda()

    for step, (image, voxel) in enumerate(testloader):
        if torch.cuda.is_available:
            image = image.cuda()
            voxel = voxel.cuda()
        feature_vector = Encoder(voxel)
        recon_voxel_3D = Decoder(feature_vector)
        recon_voxel_3D = recon_voxel_3D.view(BATCH_SIZE, -1)
        real_voxel = voxel.view(BATCH_SIZE, -1)
        loss_1 = loss_func_1(recon_voxel_3D, real_voxel)

        try:
            ave_pre = evaluation(real_voxel.view(-1).cpu(), recon_voxel_3D.view(-1).cpu())
        except BaseException:
            print('eval error')

        decoder_input = Image_2D(image)
        recon_voxel_2d = Decoder(decoder_input)
        loss_2 = loss_func_1(recon_voxel_2d, real_voxel)
        ave_pre_2 = evaluation(recon_voxel_2d.view(-1).cpu(), recon_voxel.view(-1).cpu())

        if step % 200 == 0:
            print('=============3D TEST==================')
            print('+++++++LOSS:' + str(loss_1.item()) + '+++++++')
            print('AP: %.5f' % ave_pre)

            print('=============2D TEST==================')
            print('+++++++LOSS:' + str(loss_2.item()) + '+++++++')
            print('AP: %.5f' % ave_pre_2)

        if step % 20 == 0:
            niter = step
            writer.add_scalar('Test/Loss_3D', loss_1.item(), niter)
            writer.add_scalar('Test/Loss_2D', loss_2.item(), niter)
            writer.add_scalar('Test/AP_3D', ave_pre.item(), niter)
            writer.add_scalar('Test/AP_2D', ave_pre_2.item(), niter)

if __name__ == '__main__':
    performanceTest()