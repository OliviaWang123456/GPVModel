import os
import sys
import argparse
import shutil
import time

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torch.nn.init as init
import evaluation.eval
import numpy as np

from modules.GPVModel import DecoderNetwork
from modules.GPVModel import Network_3D
from torchvision.models.alexnet import alexnet
from dataprocess import TrainDataset
from utils.visualize import Visualizer
from torch.autograd import Variable
import visdom

# torch.manual_seed(1)    # reproducible
torch.manual_seed(1)
torch.cuda.manual_seed(1)

EPOCH_1 = 200
EPOCH_2 = 100
EPOCH_3 = 100
BATCH_SIZE = 5
CWD_PATH = os.getcwd()
PRETRAINED = True
GPU_NUM = 4
MOMENTUM= 0.90
LR_1 = 1e-6
LR_2 = 1e-8
LR_3 = 1e-10
AUTO_TRAIN = True
REGRESS_TRAIN = True
TOTAL_TRAIN = True

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, mean=0, std=0.01)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)

def main():
    vis = Visualizer(env='env')
    train_data = TrainDataset(CWD_PATH, loaded=True, test=False)
    test_data = TrainDataset(CWD_PATH, loaded=True, test=True)
    trainloader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    Encoder = Network_3D()
    Decoder = DecoderNetwork()

    my_alexnet = alexnet(pretrained=PRETRAINED)
    features = list(my_alexnet.classifier.children())[:-1]
    features.extend([nn.Linear(4096, 64)])
    my_alexnet.classifier = nn.Sequential(*features)
    Image_2D = my_alexnet

    Encoder.apply(weights_init)
    Decoder.apply(weights_init)

    if not torch.cuda.is_available():
        loss_func_1 = nn.BCELoss()
        loss_func_2 = nn.MSELoss()

    else:
        distributed = GPU_NUM > 1
        if distributed > 1:
            Image_2D = nn.parallel.DataParallel(Image_2D).cuda()
            Encoder = nn.parallel.DataParallel(Encoder).cuda()
            Decoder = nn.parallel.DataParallel(Decoder).cuda()
        else:
            Image_2D = Image_2D.cuda()
            Encoder = Encoder.cuda()
            Decoder = Decoder.cuda()
        loss_func_1 = nn.BCELoss().cuda()
        loss_func_2 = nn.MSELoss().cuda()

    params_1 = [{'params': Encoder.parameters(), 'lr': LR_1}, {'params': Decoder.parameters(), 'lr': LR_1}]
    params_2 = [{'params': Image_2D.parameters(), 'lr': LR_2}]
    params_3 = [{'params': Encoder.parameters(), 'lr': LR_3}, {'params': Image_2D.parameters(), 'lr': LR_3}, {'params': Decoder.parameters(), 'lr': LR_3}]

    optimizer_1 = torch.optim.Adam(params_1)
    optimizer_2 = torch.optim.Adam(params_2)
    optimizer_3 = torch.optim.Adam(params_3)

    if AUTO_TRAIN:
        for epoch in range(EPOCH_1):
            notJointLoss = []
            notJointAP = []
            notJointValiLoss = []
            notJointValiAP = []
            notJointTestLoss = []
            notJointTestAP = []

            for step, (image, voxel) in enumerate(trainloader):
                if torch.cuda.is_available:
                    voxel = voxel.cuda()
                feature_vector = Encoder(voxel)
                recon_voxel = Decoder(feature_vector)
                recon_voxel = recon_voxel.view(BATCH_SIZE, -1)
                real_voxel = voxel.view(BATCH_SIZE, -1)
                loss_1 = loss_func_1(recon_voxel, real_voxel)
                avePre = eval.evaluation(real_voxel.view(-1).cpu(), recon_voxel.view(-1).cpu())
                notJointLoss.append(loss_1)
                notJointAP.append(avePre)

                optimizer_1.zero_grad()
                loss_1.backward()
                optimizer_1.step()

                if step % 200 == 0:
                    print('+++++++EPOCH: '+ str(epoch) + 'LOSS: ' + str(loss_1.item()) + '+++++++')
                    print('AP: %.5f' % avePre)
                    
            for step, (image, voxel) in enumerate(valiloader):
                if torch.cuda.is_available:
                    voxel = Variable(voxel.cuda(), requires_grad=False)
                feature_vector = Encoder(voxel)
                recon_voxel = Decoder(feature_vector)
                recon_voxel = recon_voxel.view(BATCH_SIZE, -1)
                real_voxel = voxel.view(BATCH_SIZE, -1)
                loss_1 = loss_func_1(recon_voxel, real_voxel)
                avePre = eval.evaluation(real_voxel.view(-1).cpu(), recon_voxel.view(-1).cpu())
                notJointValiLoss.append(loss_1)
                notJointValiAP.append(avePre)

                if step % 200 == 0:
                    print('=============WITHOUT JOINT Validation==================')
                    print('+++++++LOSS:' + str(loss_1.item()) + '+++++++')
                    print('AP: %.5f' % avePre)

            for step, (image, voxel) in enumerate(testloader):
                if torch.cuda.is_available:
                    voxel = Variable(voxel.cuda(), requires_grad=False)
                feature_vector = Encoder(voxel)
                recon_voxel = Decoder(feature_vector)
                recon_voxel = recon_voxel.view(BATCH_SIZE, -1)
                real_voxel = voxel.view(BATCH_SIZE, -1)
                loss_1 = loss_func_1(recon_voxel, real_voxel)
                avePre = eval.evaluation(real_voxel.view(-1).cpu(), recon_voxel.view(-1).cpu())
                notJointTestLoss.append(loss_1)
                notJointTestAP.append(avePre)

                if step % 200 == 0:
                    print('=============WITHOUT JOINT TEST==================')
                    print('+++++++LOSS:' + str(loss_1.item()) + '+++++++')
                    print('AP: %.5f' % avePre)

            vis.plot('AUTO_LOSS', [np.mean(np.array(notJointLoss)), np.mean(np.array(notJointValiLoss)), np.mean(np.array(notJointTestLoss))])
            vis.plot('AUTO_AP', [np.mean(np.array(notJointAP)), np.mean(np.array(notJointValiAP)), np.mean(np.array(notJointTestAP))])
            vis.log("epoch:{epoch},lr:{lr},loss:{loss}, AP:{ap_out}\n".format(epoch=epoch,
                                                                              loss=np.mean(np.array(notJointTestLoss)),
                                                                              lr=LR_1,
                                                                              ap_out=np.mean(np.array(notJointTestAP))))

        torch.save(Encoder.state_dict(), './ModelDict/encoder.pkl')
        torch.save(Decoder.state_dict(), './ModelDict/decoder.pkl')

    if REGRESS_TRAIN:
        for epoch in range(EPOCH_2):
            midLoss = []
            midAP = []
            midTestLoss = []
            midTestAP = []

            for step, (image, voxel) in enumerate(trainloader):
                if torch.cuda.is_available:
                    image = image.cuda()
                    voxel = voxel.cuda()
                Encoder.load_state_dict(torch.load('./ModelDict/encoder.pkl'))
                feature_vector = Encoder(voxel)
                regress_vector = Image_2D(image)
                loss_2 = loss_func_2(feature_vector, regress_vector) / 100 / 200
                ap = eval.evaluation(real_voxel.view(-1).cpu(), recon_voxel.view(-1).cpu())
                midLoss.append(loss_2)
                midAP.append(ap)

                optimizer_2.zero_grad()
                loss_2.backward()
                optimizer_2.step()

                if step % 200 == 0:
                    print('+++++++' + str(loss_2.item()) + '+++++++')
                    print('+++++++EPOCH: ' + str(epoch) + 'LOSS: ' + str(loss_2.item()) + '+++++++')
                    print('AP: %.5f' % ap)

            for step, (image, voxel) in enumerate(testloader):
                if torch.cuda.is_available:
                    image = Variable(image.cuda(), requires_grad=False)
                    voxel = Variable(voxel.cuda(), requires_grad=False)

                feature_vector = Encoder(voxel)
                regress_vector = Image_2D(image)
                recon_voxel = Decoder(regress_vector)
                recon_voxel = recon_voxel.view(BATCH_SIZE, -1)
                real_voxel = voxel.view(BATCH_SIZE, -1)

                loss_2 = loss_func_2(feature_vector, regress_vector) / 100 / 200
                ap_test = eval.evaluation(real_voxel.view(-1).cpu(), recon_voxel.view(-1).cpu())
                midTestLoss.append(loss_2)
                midTestAP.append(ap_test)

                if step % 50 == 0:
                    print('=============Medium Validation==================')
                    print('+++++++LOSS: ' + str(loss_2.item()) + '+++++++')
                    print('AP: %.5f' % ap_test)

            vis.plot('MID_LOSS', [np.mean(np.array(midLoss)), np.mean(np.array(midTestLoss))])
            vis.plot('MID_AP', [np.mean(np.array(midAP)), np.mean(np.array(midTestAP))])
            vis.log("epoch:{epoch},lr:{lr},loss:{loss}, AP:{ap_out}\n".format(epoch=epoch,
                                                                              loss=np.mean(np.array(midTestLoss)),
                                                                              lr=LR_2,
                                                                              ap_out=np.mean(np.array(midAP))))

        torch.save(Encoder.state_dict(), './ModelDict/encoder_2.pkl')
        torch.save(Image_2D.state_dict(), './ModelDict/image.pkl')

    if TOTAL_TRAIN:
        for epoch in range(EPOCH_3):
            JointLoss = []
            JointAP = []
            JointTestLoss = []
            JointTestAP = []

            for step, (image, voxel) in enumerate(trainloader):
                if torch.cuda.is_available:
                    image = image.cuda()
                    voxel = voxel.cuda()
                Encoder.load_state_dict(torch.load('./ModelDict/encoder_2.pkl'))
                Decoder.load_state_dict(torch.load('./ModelDict/decoder.pkl'))
                Image_2D.load_state_dict(torch.load('./ModelDict/image.pkl'))

                feature_vector = Encoder(voxel)
                regress_vector = Image_2D(image)
                recon_voxel = Decoder(regress_vector)
                recon_voxel = recon_voxel.view(BATCH_SIZE, -1)
                real_voxel = voxel.view(BATCH_SIZE, -1)

                ap = eval.evaluation(real_voxel.view(-1).cpu(), recon_voxel.view(-1).cpu())
                loss_3 = loss_func_1(recon_voxel, real_voxel) + loss_func_2(feature_vector, regress_vector) * 0.01 * 0.005
                JointLoss.append(loss_3)
                JointAP.append(ap)

                optimizer_3.zero_grad()
                loss_3.backward()
                optimizer_3.step()

                if step % 200 == 0:
                    print('+++++++EPOCH: ' + str(epoch) + 'LOSS: ' + str(loss_3.item()) + '+++++++')
                    print('AP: %.5f' % ap)

            for step, (image, voxel) in enumerate(testloader):
                if torch.cuda.is_available:
                    image = Variable(image.cuda(), requires_grad=False)
                    voxel = Variable(voxel.cuda(), requires_grad=False)

                feature_vector = Encoder(voxel)
                regress_vector = Image_2D(image)
                recon_voxel = Decoder(regress_vector)
                recon_voxel = recon_voxel.view(BATCH_SIZE, -1)
                real_voxel = voxel.view(BATCH_SIZE, -1)

                loss_3 = loss_func_1(recon_voxel, real_voxel) + loss_func_2(feature_vector, regress_vector) * 0.01 * 0.005
                ap_test = eval.evaluation(real_voxel.view(-1).cpu(), recon_voxel.view(-1).cpu())
                JointTestLoss.append(loss_3)
                JointTestAP.append(ap_test)

                if step % 50 == 0:
                    print('=============FINAL Validation==================')
                    print('+++++++LOSS: ' + str(loss_3.item()) + '+++++++')
                    print('AP: %.5f' % ap_test)

            vis.plot('FINAL_LOSS', [np.mean(np.array(JointLoss)), np.mean(np.array(JointTestLoss))])
            vis.plot('FINAL_AP', [np.mean(np.array(JointAP)), np.mean(np.array(JointTestAP))])
            vis.log("epoch:{epoch},lr:{lr},loss:{loss}, AP:{ap_out}\n".format(epoch=epoch,
                                                                              loss=np.mean(np.array(JointTestLoss)),
                                                                              lr=LR_3,
                                                                              ap_out=np.mean(np.array(JointTestAP))))

        torch.save(Decoder.state_dict(), './ModelDict/decoder_final.pkl')
        torch.save(Encoder.state_dict(), './ModelDict/encoder_final.pkl')
        torch.save(Image_2D.state_dict(), './ModelDict/image_final.pkl')

if __name__ == '__main__':
    main()
