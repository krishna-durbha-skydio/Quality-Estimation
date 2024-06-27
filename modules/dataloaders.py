# Importing Libraries
import numpy as np

import torch
from torchvision import transforms

class torch_transform:
    def __init__(self, size):
        self.transform1 = transforms.Compose(
            [
                transforms.Resize((size[0],size[1])),
                transforms.ToTensor(),
            ]
        )
        
        self.transform2 = transforms.Compose(
            [
                transforms.Resize((size[0] // 2, size[1] // 2)),
                transforms.ToTensor(),
            ]
        )
    
    def __call__(self, x):
        return self.transform1(x), self.transform2(x)
    

def create_data_loader(image, image_2, batch_size):
    train = torch.utils.data.TensorDataset(image, image_2)
    loader = torch.utils.data.DataLoader(
        train, 
        batch_size=batch_size,
        drop_last=False,
        num_workers=8,
        sampler=None,
        shuffle=False
    )
    return loader


def extract_features(model, loader):
    feat = []
    
    model.eval()
    for step, (batch_im, batch_im_2) in enumerate(loader):
        batch_im = batch_im.type(torch.float32)
        batch_im_2 = batch_im_2.type(torch.float32)
        
        batch_im = batch_im.cuda(non_blocking=True)
        batch_im_2 = batch_im_2.cuda(non_blocking=True)
        
        with torch.no_grad():
            _,_, _, _, model_feat, model_feat_2, _, _ = model(batch_im, batch_im_2)
        
        feat_ = np.hstack((model_feat.detach().cpu().numpy(), model_feat_2.detach().cpu().numpy()))
        feat.extend(feat_)
    return np.array(feat)


def extract_features_temporal(model, loader):
    feat = []
    
    model.eval()
    for step, (batch_im, batch_im_2) in enumerate(loader):
        batch_im = batch_im.type(torch.float32)
        batch_im_2 = batch_im_2.type(torch.float32)
        
        batch_im = batch_im.cuda(non_blocking=True).unsqueeze(0)
        batch_im_2 = batch_im_2.cuda(non_blocking=True).unsqueeze(0)
        
        with torch.no_grad():
            _, _, model_feat, model_feat_2 = model(batch_im, batch_im_2)
        
        feat_ = np.hstack((model_feat.detach().cpu().numpy(), model_feat_2.detach().cpu().numpy()))
        feat.extend(feat_)
        
    return np.array(feat)