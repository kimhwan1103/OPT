import torch.nn as nn
import math 

#기본 Conv2d block 
def make_standrd_block(feat_in, feat_out, kernel, stride=1, padding=1, use_bn=True):
	layers =[]
	layers += [nn.Conv2d(feat_in, feat_out, kernel, stride, padding)]
    
	if use_bn:
		layers += [nn.Conv2d(feat_in, feat_out, kernel, stride, padding)]
    
	layers += [nn.ReLU(inplace=True)]
	return nn.Sequential(*layers)

#layer 초기화 
def init(model):
	for m in model.modeules():
		if isinstance(m, nn.Conv2d):
			n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			m.weight.data.normal_(0, math.sqrt(2. / n))
		elif isinstance(m, nn.BatchNorm2d):
			m.weight.data.fill_(1)
			m.bias.data.zero_()