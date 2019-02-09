import torch.nn as nn


class Network_3D(nn.Module):
    def __init__(self):
        super(Network_3D, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(20, 96, 7, 1),
            nn.PReLU(),                        
        )
        self.conv2 = nn.Sequential(        
            nn.Conv2d(96, 256, 5, 1),     
            nn.PReLU(),                      
        )
        self.conv3 = nn.Sequential(        
            nn.Conv2d(256, 384, 3, 1),     
            nn.PReLU(),                      
        )
        self.conv4 = nn.Sequential(        
            nn.Conv2d(384, 256, 3, 1),     
            nn.PReLU(),                      
        )
        self.FC = nn.Sequential(
            nn.Linear(6 * 6 * 256, 64),  #TODO
        )
        self.LastLinear = nn.Linear(64, 216)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.FC(x)
        # output = self.LastLinear(x)
        # output = output.view(-1, 6, 6, 6)   #TODO
        return x     # x: input of regression   output: initial input of decoder

class DecoderNetwork(nn.Module):
    def __init__(self):
        super(DecoderNetwork, self).__init__()
        self.trans = nn.Linear(64, 216)
        self.conv5 = nn.Sequential(        
            nn.ConvTranspose2d(6, 256, 3, 1),     #TODO
            nn.PReLU(), 
        )
        self.conv6 = nn.Sequential(        
            nn.ConvTranspose2d(256, 384, 3, 1, 0),   
            nn.PReLU(),                      
        )
        self.conv7 = nn.Sequential(        
            nn.ConvTranspose2d(384, 256, 5, 1, 0),  
            nn.PReLU(),                      
        )
        self.conv8 = nn.Sequential(        
            nn.ConvTranspose2d(256, 96, 7, 1, 0),  
            nn.PReLU(),                      
        )
        self.conv9 = nn.Sequential(        
            nn.ConvTranspose2d(96, 20, 1, 1, 0),
            nn.PReLU(),                      
        )
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.trans(x)
        x = x.view(-1, 6, 6, 6)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x) 
        x = self.conv9(x)    
        output = self.out(x)
        return output 

