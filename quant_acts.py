import copy
import torch
import torch.nn as nn

from pwlq import *
from uniform import *

##########################################################################################
####  Quantization of Activations
##########################################################################################

class QuantAct(nn.Module):
    '''
    Quantize actications including:
    (1) the input of conv layer
    (2) the input of linear fc layer
    (3) the input of pooling layer
    '''
    def __init__(self, act_bits, get_stats, minv=None, maxv=None,
        cali_sample_size=512, cali_batch_size=4, topk=10):
        '''
        cali_sample_size: calibration sample size, typically from random training data
        cali_batch_size: calibration sampling batch size
        topk: calibrate topk lower and upper bounds
        '''
        super(QuantAct, self).__init__()
        self.act_bits = act_bits
        self.get_stats = get_stats
        self.index = 0
        self.topk = topk
        self.sample_batches = cali_sample_size // cali_batch_size
        stats_size = (self.sample_batches, self.topk) if self.get_stats else 1
        self.register_buffer('minv', torch.zeros(stats_size))
        self.register_buffer('maxv', torch.zeros(stats_size))

    def forward(self, x):
        if self.get_stats:                #get_states=1时表示位于激活层的采集模式
            y = x.clone()
            y = torch.reshape(y, (-1,))     #s
            y, indices = y.sort()
            topk_mins = y[:self.topk]       #最小的十个
            topk_maxs = y[-self.topk:]      #最大的十个
            if self.index < self.sample_batches:
                self.minv[self.index, :] = topk_mins
                self.maxv[self.index, :] = topk_maxs
                self.index += 1
#                print('y:{}'.format(y))
#                print('minv:{},maxv:{}'.format(self.minv,self.maxv))
#                print('size:{}'.format(self.minv.size()))
        if self.act_bits > 0:
            # uniform quantization
            # if self.minv is not None:
            #     if _minv >= 0.0: # activation after relu
            #         _minv *= 0.0
            #         self.signed = False
            #     else:
            #         _maxv = max(-self.minv, self.maxv)
            #         _minv = - self.maxv
            #         self.signed = True
             x = uniform_symmetric_quantizer(x, bits=self.act_bits,
                     minv=-2, maxv=2, signed=True)
        return x


def quant_model_acts(model, act_bits, get_stats, cali_batch_size=4):   #get_state = 0  act_bits = 8.0
    """
    Add quantization of activations for a pretrained model recursively
    """
    print(type(model))
    if type(model) in [nn.Conv2d, nn.Linear, nn.AdaptiveAvgPool2d]:
        print(1)
        quant_act = QuantAct(act_bits, get_stats, cali_batch_size=cali_batch_size)
        return nn.Sequential(quant_act, model)
    elif type(model) == nn.Sequential:
        print(2)
        modules = []
        for name, module in model.named_children():
            modules.append(quant_model_acts(module, act_bits, get_stats, cali_batch_size=cali_batch_size))
        return nn.Sequential(*modules)
    else:
        print(3)
        quantized_model = copy.deepcopy(model)
        for attribute in dir(model):
            module = getattr(model, attribute)
            if isinstance(module, nn.Module):
                setattr(quantized_model, attribute,
                    quant_model_acts(module, act_bits, get_stats, cali_batch_size=cali_batch_size))
        return quantized_model
