#%%
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import softplus, relu
from torch.amp import autocast

from torch.utils.data import DataLoader
from torchvision import datasets as df, transforms as tt
from torchinfo import summary

import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm.autonotebook import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    '-l',
    '--load_weights',
    action='store_false',
)

args = parser.parse_args()

batch_size = 128


compose = tt.Compose([
    tt.Resize(256+8),
    tt.RandomCrop(256),
    tt.ToTensor(),
    tt.Normalize(.5,.5)
])
data = df.CelebA(
    root='/storage/SSD1/.data',
    download=False,
    transform=compose
)
data = DataLoader(
    dataset=data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
)
gpu = torch.device('cuda:1')
lenghtData = len(data)

# %%
def setGrads(model:nn.Module, value:bool):
    try:
        for p in model.parameters():
            p.requires_grad_(value)
    except: print('Não foi possivel modificar os grads do modelo')

class AdaIn(nn.Module):
    def __init__(self, layers, eps=1e-8):
        super().__init__()
        self.gamma = nn.Linear(1024,layers)
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        self.beta = nn.Linear(1024,layers)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
        self.eps = eps
    def forward(self, x:torch.Tensor, style:torch.Tensor):
        mean = x.mean([2,3],keepdim=True)
        std = (x.var([2,3],unbiased=False,keepdim=True)+self.eps).sqrt()
        x = (x-mean)/std
        gamma = self.gamma(style).view(x.size(0),-1,1,1)
        beta = self.beta(style).view(x.size(0),-1,1,1)
        x = gamma*x + beta
        return x
    
class NoiseInject2D(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.gamma = nn.Linear(1024,layers)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
    def forward(self, x:torch.Tensor, style:torch.Tensor):
        assert x.size(0)==style.size(0), 'batch deve ser igual'
        B,C,H,W = x.size()
        modulation = self.gamma(style).view(B,-1,1,1)
        noise = torch.randn(B,1,H,W,device=x.device)
        x = x + modulation*noise
        return x

class ResConvBlock(nn.Module):
    def __init__(self,channels,eps=0.1):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(channels,channels,3,1,1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels,channels,3,1,1),
            nn.InstanceNorm2d(channels)
        )
        self.gamma = nn.Parameter(torch.ones(1,channels,1,1)*eps)
    def forward(self, x:torch.Tensor):
        return x + self.gamma*self.residual(x)
    
class UpSampleBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels,channels//2,3,1,1)
        self.conv2 = nn.Conv2d(channels//2,channels//2,3,1,1)
        self.adain1 = AdaIn(channels//2)
        self.adain2 = AdaIn(channels//2)
        self.noise1 = NoiseInject2D(channels//2)
        self.noise2 = NoiseInject2D(channels//2)
    def forward(self, x:torch.Tensor, style:torch.Tensor):
        x = F.interpolate(x,scale_factor=2,mode='bilinear')
        x = self.conv1(x)
        x = self.noise1(x,style)
        x = self.adain1(x,style)
        x = F.relu(x,inplace=True)
        x = self.conv2(x)
        x = self.noise2(x,style)
        x = self.adain2(x,style)
        x = F.relu(x,inplace=True)
        return x

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mappingNetwork = nn.Sequential(
            nn.Linear(128,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,1024),
        )
        for layer in self.mappingNetwork:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight,nonlinearity='relu')
                nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.mappingNetwork[-1].weight)
        nn.init.zeros_(self.mappingNetwork[-1].bias)
        self.initialMap = nn.Parameter(torch.randn(1,512,4,4))
        self.upsample = nn.ModuleList([UpSampleBlock(n) for n in [512,256,128,64,32,16]])
        self.toRGB = nn.Sequential(
            nn.Conv2d(8,3,3,1,1),
            nn.Tanh()
        )
    def forward(self, x:torch.Tensor):
        style = self.mappingNetwork(x)
        x = self.upsample[0](self.initialMap.expand(x.size(0),-1,-1,-1),style)
        for i in range(1,6):
            x = self.upsample[i](x,style)
        x = self.toRGB(x)
        return x
    
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 256x256
            nn.Conv2d(4,32,4,2,1),              # 0
            # 128x128
            nn.LeakyReLU(.2,inplace=True),
            nn.Conv2d(32,64,4,2,1),             # 2
            # 64x64
            nn.LeakyReLU(.2,inplace=True),
            nn.Conv2d(64,128,4,2,1),             # 4
            # 32x32
            nn.LeakyReLU(.2,inplace=True),
            nn.Conv2d(128,256,4,2,1),            # 6
            # 16x16
            nn.LeakyReLU(.2,inplace=True),
            nn.Conv2d(256,512,4,2,1),           # 8
            # 8x8
            nn.LeakyReLU(.2,inplace=True),
            nn.Conv2d(512,512,4,2,1),           # 10
            # 4x4
            nn.LeakyReLU(.2,inplace=True),
            nn.Conv2d(512,1024,4,2,1),
            # 2x2
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.toRGB16 = nn.Conv2d(256,1,1,1,0)
        self.fc = nn.Linear(1024,1)
    def forward(self, x:torch.Tensor):
        var = (x.var(1,unbiased=False,keepdim=True).mean([2,3],keepdim=True)+1e-8).sqrt()
        var = var.expand(-1,1,x.size(2),x.size(3))
        x = torch.cat([x,var],1)
        x = self.features[0:8](x)
        convMap = self.toRGB16(x).flatten(1).mean(1,keepdim=True)
        x = self.features[8:](x)
        x = self.fc(x)
        return torch.cat([x,convMap],dim=1)
    
gen = Generator().to(gpu)
crit = Critic().to(gpu)
if args.load_weights:
    gen.load_state_dict(torch.load('./generator.weights.pt'))
    crit.load_state_dict(torch.load('./critic.weights.pt'))
# crit = Critic().to(gpu)
# gen = Generator().to(gpu)
optG = torch.optim.Adam(
    gen.parameters(),
    1e-4,
    betas=(.0,.99),
)
optC = torch.optim.Adam(
    crit.parameters(),
    1e-4,
    betas=(.0,.99),
)
gen(torch.randn(1,128,device=gpu))
# summary(gen,(batch_size,128),verbose=1,device=gpu)
# summary(crit,(batch_size,3,256,256),verbose=1,device=gpu)
dLoss, gLoss = None,None
dLossPlot, gLossPlot = [],[],
beta = (len(data)/2-1)/(len(data)/2)

#%%
for epoch in range(5000):
    batchBar = tqdm(data)
    gen.train(),crit.train()
    for i,(batch,_) in enumerate(batchBar):
        batch = batch.to(gpu)

        # Critic forward & backward
        setGrads(crit,True)
        with autocast('cuda',torch.bfloat16):
            noise = torch.randn(batch.size(0),128,device=gpu)
            with torch.no_grad():
                fakeImgs = gen(noise)
            fakeLogits = crit(fakeImgs)
            trueLogits = crit(batch)
            dLoss_ = softplus(-trueLogits).mean() + softplus(fakeLogits).mean()
            alpha = torch.rand(batch.size(0),1,1,1,device=gpu)
            if i%16 == 0:
                batch_grad = batch.requires_grad_(True)
                grads = torch.autograd.grad(
                    crit(batch_grad).sum(),
                    batch_grad,
                    create_graph=True,
                )[0]
                norm = grads.flatten(1).norm(2,1)
                R1 = norm.square().mean()
                dLoss_ += 10/2*16*R1
            optC.zero_grad()
            dLoss_.backward()
            optC.step()

        # Generator forward & backward
            if i%5 == 0:
                setGrads(crit,False)
                noise = torch.randn(batch.size(0),128,device=gpu)
                fakeImgs = gen(noise)
                fakeLogits = crit(fakeImgs)
                gLoss_ = softplus(-fakeLogits).mean()
                optG.zero_grad()
                gLoss_.backward()
                optG.step()
        # Losses Calc
        dLoss = dLoss_.item() if dLoss is None else beta*dLoss + (1-beta)*dLoss_.item() 
        gLoss = gLoss_.item() if gLoss is None else beta*gLoss + (1-beta)*gLoss_.item()
        # Logs
        dictLoss = {
            'dLoss_':f'{dLoss_.item():.4f}',
            'gLoss_':f'{gLoss_.item():.4f}',
        }
        batchBar.set_postfix(dictLoss)
        # if i == 15: break

        # Plots
        if i%(lenghtData//4) == 0:
            k = 7
            res,dpi = 1800,300
            noise = torch.randn(k**2,128,device=gpu)
            with torch.inference_mode():
                imgs = gen(noise)
            imgs = imgs.permute(0,2,3,1).cpu().numpy()*127.5+127.5
            imgs = np.uint8(imgs)
            fig,ax = plt.subplots(k,k,figsize=(res/dpi,res/dpi),dpi=dpi)
            ax = ax.ravel()
            for ii in range(k**2):
                ax[ii].imshow(imgs[ii])
                ax[ii].axis(False)
            plt.tight_layout(pad=0)
            plt.savefig('gen.png')
            plt.close()
        if i%(lenghtData//2)==0:
            dLossPlot.append(dLoss)
            gLossPlot.append(gLoss)
            plt.plot(gLossPlot,label='gLoss')
            plt.plot(dLossPlot,label='dLoss')
            plt.legend()
            plt.tight_layout()
            plt.savefig('loss.png')
            plt.close()
            torch.save(gen.state_dict(),'generator.weights.pt')
            torch.save(crit.state_dict(),'critic.weights.pt')