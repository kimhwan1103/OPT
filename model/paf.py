#참고 코드 https://github.com/fitexmage/XNect_Implementation/blob/master/model/paf_model.py
import torch.nn as nn
import torch 
from .util import make_standrd_block, init

'''
backend : 이미제에서 특징을 추출하기 위한 백엔드 모델 
backend_outp_feats : 백엔드 모델의 출력 특징 맵의 차원수 
n_joints : 관절의 개수 
n_paf : PAF 필드의 개수 
n_stages : Stage 클래스의 개수 (기본 7)
'''
class PAFModel(nn.Module):
	def __init__(self, backend, backend_outp_feats, n_joints, n_paf, n_stages=7):
		super(PAFModel, self).__init__()
		assert(n_stages > 0)
		self.backend = backend 
		stages = [Stage(backend_outp_feats, n_joints, n_paf, True)]
		for i in range(n_stages -1):
			stages.append(Stage(backend_outp_feats, n_joints, n_paf, False))
		self.stages = nn.ModuleList(stages)

	def forward(self, x):
		img_feats = self.backend(x)
		cur_feats = img_feats 
		heatmap_outs = []
		paf_outs = []
		for i, stage in enumerate(self.stages):
			heatmap_out, paf_out = stage(cur_feats)
			heatmap_outs.append(heatmap_out)
			paf_outs.append(paf_out)
			cur_feats = torch.cat([img_feats, heatmap_out, paf_out], 1)

		return heatmap_outs, paf_outs
	
'''
backend_outp_feats : 백엔드 모델의 출력 특징 맵의 차원수 
n_joints : 관절의 개수 
n_paf : PAF 필드의 개수 
stage2 : Stage 2 여부 
'''
class Stage(nn.Module):
	def __init__(self, backend_outp_feats, n_joints, n_paf, stage2):
		super(Stage, self).__init__()
		inp_feats =backend_outp_feats
		if stage2:
			self.block1 = make_paf_block_stage2(inp_feats, n_joints)
			self.block2 = make_paf_block_stage2(inp_feats, n_paf)
		else:
			inp_feats = backend_outp_feats + n_paf
			self.block1 = make_paf_block_stage3(inp_feats, n_joints)
			self.block2 = make_paf_block_stage2(inp_feats, n_paf)
		init(self.block1)
		init(self.block2)
	
	def forward(self, x):
		y1 = self.block1(x)
		y2 = self.block2(x)
		return y1, y2
		
def make_paf_block_stage2(inp_feats, output_feats):
	layers = [make_standrd_block(inp_feats, 128, 3),
	   make_standrd_block(128, 128, 3),
	   make_standrd_block(128, 128, 3),
	   make_standrd_block(128, 512, 1, 1, 0)]
	layers += [nn.Conv2d(512, output_feats, 1, 1, 0)]
	return nn.Sequential(*layers)

def make_paf_block_stage3(inp_feats, output_feats):
	layers = [make_standrd_block(inp_feats, 128, 7, 1, 3),
	   make_standrd_block(128, 128, 7, 1, 3),
	   make_standrd_block(128, 128, 7, 1, 3),
	   make_standrd_block(128, 128, 7, 1, 3),
	   make_standrd_block(128, 128, 7, 1, 3),
	   make_standrd_block(128, 128, 1, 1, 0)]
	layers += [nn.Conv2d(128, output_feats, 1, 1, 0)]
	return nn.Sequential(*layers)

