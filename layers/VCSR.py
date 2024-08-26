import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
from collections import OrderedDict
from .VCSR_Modules import *
import copy

from layers.base_model import resnet181, ExtractFeature
from layers.location_model import Location_Net_stage_one

from sklearn import cluster, metrics
import numpy as np
from sklearn.preprocessing import normalize
import time


device = torch.device("cuda:1")

    
class BaseModel(nn.Module):
    def __init__(self, opt={}, vocab_words=[]):
        super(BaseModel, self).__init__()
        # text feature
        text_backbone = Skipthoughts_Embedding_Module(
            vocab= vocab_words,
            opt = opt
        )
        
        visual_backbone = ExtractFeature()
        
        self.av_model = Location_Net_stage_one(visual_net=visual_backbone, txt_net=text_backbone)


        self.Eiters = 0

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.maxpool1 = nn.AdaptiveMaxPool1d(1)
        self.maxpool2d = nn.AdaptiveMaxPool2d((1,1))
        
        
        self.linear = nn.Linear(512,49)
        
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        
        
        #self.convN = nn.Conv2d(512,64,1)
        self.convN = nn.Sequential(
            nn.Conv2d(512,64,1),
            #nn.BatchNorm2d(64),
            #nn.Conv2d(64,64,1),
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True)
        )
        #self.co = nn.Conv2d(512,512,1)
        #self.supv_main_model = supv_main_model()
        
        self.convP1 = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            #nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.convN1 = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            #nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        
        self.bn = nn.BatchNorm2d(512)
        
        

    def att_map_weight(self, vis_feat_map, audio_feat_vec, srea, flag):
        
        # normalize visual feature
        B, C, H, W = vis_feat_map.size()
        srea_P = vis_feat_map
        srea_N = vis_feat_map
        sp = 0
        sq = 0
        
        if flag:
           
           #srea = srea.reshape(B, 64, H * W)
           
           srea = F.normalize(srea, p=2, dim=1)
           srea1 = srea
           vis_feat_map1 = F.normalize(vis_feat_map, p=2, dim=1)
           audio_feat_vec1 = F.normalize(audio_feat_vec, p=2, dim=1)
           
           O = torch.sum(srea.unsqueeze(2) * vis_feat_map1.unsqueeze(1), dim=(-2, -1))
           
           #O = F.softmax(O, dim=1)
           #O = F.normalize(O, p=2, dim=1)
           score = torch.einsum('bnc,bc->bn',[O, audio_feat_vec1])
           
           
           score1 = score
           
           score = (score>score.mean(-1).unsqueeze(-1).repeat(1, 64)).float()
           '''
           score_P = score.unsqueeze(-1).repeat(1, 1, H*W).view(B, 64, H, W)
           

           srea_P = (score_P*srea1).view(B, 64, H*W)
           srea_P, _ = torch.sort(srea_P, dim=1, descending=True)
           srea_P = srea_P[:,:4,:].mean(1)
           
           srea_N = (srea1*(1-score_P)).view(B, 64, H*W)
           srea_N, _ = torch.sort(srea_N, dim=1, descending=True)
           srea_N = srea_N[:,:4,:].mean(1)
           
           
           '''
           s_i2t, index = torch.sort(score1, dim=1, descending=True)
           maskp = torch.zeros_like(score1).to(device)
           y = index[:,0].squeeze().to(torch.long).view(-1, 1)
           maskp.scatter_(1, y, 1)
           maskp = maskp.unsqueeze(-1).repeat(1, 1, H*W).view(B, C/2, H, W)
           srea_P = (srea*maskp).mean(1).view(B, H*W)
           
           maskn = torch.zeros_like(score1).to(device)
           y = index[:,-1].squeeze().to(torch.long).view(-1, 1)
           maskn.scatter_(1, y, 1)
           maskn = maskn.unsqueeze(-1).repeat(1, 1, H*W).view(B, C/2, H, W)
           srea_N = (srea*maskn).mean(1).view(B, H*W)

           sp = (score1 * score).mean(-1)
           sq = (score1 * (1-score)).mean(-1)
           
        
        #audio_feat_vec = self.linear(audio_feat_vec) 
        
        B, C, H, W = vis_feat_map.size()
        att_map2 = self.GAttention2(vis_feat_map, audio_feat_vec)
        
        #att_map1 = self.GAttention1(vis_feat_map, audio_feat_vec)
        return att_map1, att_map2, srea_P, srea_N, sp, sq
        
    def GAttention(self, visual, text):
        B, C, H, W = visual.size()
        asm = torch.einsum("ncqa,nchw->nqa",[visual, text.unsqueeze(2).unsqueeze(3)]).unsqueeze(1)
        asm = sigmoid(asm, 0.001)
        asm = torch.einsum("niqa,nchi->ncqa", [asm,text.unsqueeze(2).unsqueeze(3)]).view(B, C, H * W).mean(1)
        return asm
    
    def GAttention1(self, visual, text):
        B, C, H, W = visual.size()
        visual = F.normalize(visual, p=2, dim=1)
        visual = visual.view(B, C, H * W)
        visual = visual.permute(0, 2, 1)  # B x (HW) x C

        # normalize audio feature
        text = F.normalize(text, p=2, dim=1)
        text = text.unsqueeze(2)  # B x C x 1

        
        visual = visual.unsqueeze(1).repeat(1, B, 1, 1)
        text = text.unsqueeze(0).repeat(B, 1, 1, 1)
        
        att_map_orig = torch.matmul(visual, text)  # B x (HW) x 1
        att_map = self.maxpool1(torch.squeeze(att_map_orig))  # B x B x (HW)
        
        return att_map
           
    def GAttention2(self, vis_feat_map, txt_feat_vec):
        
        B, C, H, W = vis_feat_map.size()
        '''
        vis_feat_map_trans = F.normalize(vis_feat_map, p=2, dim=1)
        vis_feat_map_trans = vis_feat_map_trans.view(B, C, H * W)
        vis_feat_map_trans = vis_feat_map_trans.permute(0, 2, 1)  # B x (HW) x C

        # normalize audio feature
        txt_feat_vec = F.normalize(txt_feat_vec, p=2, dim=1)
        txt_feat_vec = txt_feat_vec.unsqueeze(2)  # B x C x 1

        # similarity/attention map
        att_map_orig = torch.matmul(vis_feat_map_trans, txt_feat_vec)  # B x (HW) x 1

        # min-max normalization on similarity map
        att_map = torch.squeeze(att_map_orig)  # B x (HW)
        '''
        t = torch.nn.functional.normalize(txt_feat_vec, dim=1)
        t = torch.unsqueeze(torch.unsqueeze(a, -1), -1)
        
        v = torch.nn.functional.normalize(vis_feat_map, dim=1)
        att_map = torch.sum(torch.mul(v, t), 1, keepdim=True).squeeze() 
                                                                           
        return att_map.view(B, H*W)
        
    def forward(self, img, txt, lengths, flag=False, val=False):
        
        v_feature, t_feature = self.av_model(img, txt, lengths)# 
        
        B,C,H,W = v_feature.shape
        
        mvsa_feature = self.bn(v_feature)    
        
        if flag==False:
           return cosine_sim(self.fc1(self.avgpool(mvsa_feature).squeeze()), t_feature)
        
        srea = self.convN(mvsa_feature)

        sim_map_view1, sim_map_view2, srea_P, srea_N, sp, sq = self.att_map_weight(mvsa_feature,t_feature,srea,flag)
        
        if flag==True:
           loss2 = js_div(sim_map_view2, srea_P.detach())
           
        
        mvsa_feature = self.fc1(self.avgpool(mvsa_feature).squeeze())
        
          
        if flag==True:
           loss1 = -(torch.log(torch.exp(srea_P.mean(-1))/(torch.exp(srea_P.mean(-1)) + torch.exp(srea_N.mean(-1))))).sum()/(srea_P.shape[0])
           loss3 = -(torch.log(torch.exp(sp)/(torch.exp(sp) + torch.exp(sq)))).sum()/(sp.shape[0])
           #loss1 = (F.cosine_similarity(srea_P, srea_N, dim=-1).mean())
        
        
        dual_sim = cosine_sim(mvsa_feature, t_feature)
        #dual_sim = cosine_similarity(mvsa_feature1, t_feature)
        
        
        return dual_sim, 0.1*loss3+0.1*loss2
        
        


def factory(opt, vocab_words, cuda=True, data_parallel=True):
    opt = copy.copy(opt)

    model = BaseModel(opt, vocab_words)

    if data_parallel:
        model = nn.DataParallel(model).to(device)
        if not cuda:
            raise ValueError

    if cuda:
        model.to(device)

    return model
    

        
def js_div(p_logits, q_logits, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    
    if get_softmax:
        p_output = F.softmax(p_logits)
        q_output = F.softmax(q_logits)
    else:
        p_output = p_logits
        q_output = q_logits
    log_mean_output = ((p_output + q_output )/2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2




