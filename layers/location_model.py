import torch
import torch.nn as nn
import torch.nn.functional as F

def sigmoid(tensor, temp=1.0):
    """ temperature controlled sigmoid

    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    
    
def MM(srea):
    srea = (srea - torch.min(srea, dim=2, keepdim=True).values) / \
              (torch.max(srea, dim=2, keepdim=True).values - torch.min(srea, dim=2,
                                                                          keepdim=True).values + 1e-10)
    return srea
    
class Location_Net_stage_one(nn.Module):
    def __init__(self, visual_net, txt_net):
        super(Location_Net_stage_one, self).__init__()

        # backbone net
        self.visual_net = visual_net
        self.txt_net = txt_net



    def forward(self, v_input, t_input, lengths):
        batch_v = v_input.shape[0]
        batch_t = t_input.shape[0]

        # visual pathway
        v_fea = self.visual_net(v_input)
        

        # audio pathway
        t_fea = self.txt_net(t_input)

        return v_fea, t_fea


