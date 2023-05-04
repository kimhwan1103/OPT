import torch 
import torch.nn as nn 
import torch.nn.functional as F

#기본 Conv 레이어 
class Conv(nn.Module):
	def __init__(self, inp, oup, kernel, padding):
		super(Conv, self).__init__()
		self.inp = inp 
		self.oup = oup 
		self.kernel = kernel
		self.padding = padding
        
	def forward(self, x):
		conv = nn.Conv2d(self.inp, self.oup, self.kernel, padding=self.padding).cuda()(x)
		bn = nn.BatchNorm2d(self.oup)(conv)
		relu = nn.ReLU(inplace=True).cuda()(bn)
		return relu
	
#기본 Deconv 레이어 
class Deconv(nn.Module):
	def __init__(self, inp, oup, kernel, stride, groups):
		super(Deconv, self).__init__()
		self.inp = inp 
		self.oup = oup
		self.kernel = kernel 
		self.stride = stride 
		self.groups = groups 

	def forward(self, x):
		deconv = nn.ConvTranspose2d(self.inp, self.oup, self.kernel, self.stride, padding=1, groups=self.groups).cuda()(x)
		bn = nn.BatchNorm2d(self.oup)(deconv)
		relu = nn.ReLU(inplace=True).cuda()(bn)
		return relu 

#ROI후 Stage 2 Model 
class Stage_2_Model(nn.Module):
	def __init__(self, num_joints, only_2d):
		super(Stage_2_Model, self).__init__()
		self.num_joints = num_joints #관절 
		self.only_2d = only_2d #2d 옵션 

		#2d brach setting
		#TODO add the BackBone Network
		#self.backbone = 
		self.conv_2d_1 = Conv(416, 256, 1, 0)
		self.deconv_2d_2 = Deconv(256, 192, 4, 2, 4)
		self.conv_2d_3 = Conv(192, 128, 3, 1)
		self.conv_2d_4 = Conv(128, 96, 1, 0)
		self.conv_2d_5 = Conv(96, self.num_joints * 3, 3, 1)

		#3d brach setting
		self.conv_3d_1 = Conv(416, 256, 1, 0)
		self.deconv_3d_2 = Deconv(256, 192, kernel=4, stride=2, groups=4)
		self.conv_3d_3 = Conv(192, 160, 3, 1)
		self.conv_3d_5 = Conv(160, 160, 1, 0)
		self.conv_3d_6 = Conv(160, 128, 3, 1)
		self.conv_3d_7 = Conv(128, self.num_joints * 3, 3, 1)

	def forward(self, x):

		#2d brach 
		#TODO add the BackBone Network 
		#backbone =
		d_2d_1 = self.conv_2d_1()
		d_2d_2 = self.deconv_2d_2(d_2d_1)
		d_2d_3 = self.conv_2d_3(d_2d_2)
		d_2d_4 = self.conv_2d_4(d_2d_3)
		d_2d_5 = self.conv_3d_5(d_2d_4)
		#feature map 추출 
		d_2d_6 = torch.split(d_2d_5, self.num_joints, 1)
		heatmap_2d_7 = d_2d_6[0]
		paf_2d_8 = torch.cat((d_2d_6[1], d_2d_6[2]), 1)

		if self.only_2d:
			return heatmap_2d_7, paf_2d_8
		
		#3D brach
		d_3d_1 = self.conv_3d_1()
		d_3d_2 = self.deconv_3d_2(d_3d_2)
		d_3d_3 = self.conv_3d_3(d_3d_2)
		d_3d_4 = self.concat_3d_4 = torch.cat((d_2d_2, d_3d_3), 1) #brach cat
		d_3d_5 = self.conv_3d_5(d_3d_4)
		d_3d_6 = self.conv_3d_6(d_3d_5)
		conv_3d_7 = self.conv_3d_7(d_3d_6)

		return heatmap_2d_7, paf_2d_8, conv_3d_7

