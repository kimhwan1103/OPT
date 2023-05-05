#코드 참고 https://github.com/fitexmage/XNect_Implementation/blob/master/data_process/data_loader_provider.py

import torch 
from torch.utils.data import DataLoader
from coco import CocoDataset

def create_data_loader(opt):
    tr_dataset, te_dataset = create_data_loader(opt)
    
    train_loader = DataLoader(
        tr_dataset,
        batch_size = opt.batchSize,
        shuffle=True if opt.DEBUG == 0 else False,
        drop_last = True,
        num_workers=opt.nThreads
	)
    
    test_loader = DataLoader(
        te_dataset,
        batch_size = opt.batchSize, 
        shuffle=False,
        num_workers=opt.nThreads 
	)
    
def create_data_sets(opt):
    if opt.dataset == 'coco':
        tr_dataset = CocoDataset(opt.data, opt, 'train')
        te_dataset = CocoDataset(opt.data, opt, 'val')
    else:
        raise ValueError('Data set ' + opt.dataset + ' Not available')
    return tr_dataset, te_dataset