import torch
from torch.utils.data import DataLoader
from dataset.Dataset import MyDataset, CityDataset
from utils import config, WeightLoss, load_premodel
from train.Trainer import trainer
from model import FCN, UNet, SegNet, DeepLabV3Plus, PSPNet, ENet, LinkNet, ICNet, BiSeNet, DFANet


traindata = CityDataset(split='train')
train_dataloader = DataLoader(dataset=traindata,
                              batch_size=config.train_batch_size,
                              shuffle=True,
                              num_workers=config.num_workers)

evaldata = CityDataset(split='val')
eval_dataloader = DataLoader(dataset=evaldata,
                             batch_size=config.test_batch_size,
                             shuffle=False,
                             num_workers=config.num_workers)

criterion = WeightLoss(use_weight=False)
model = DFANet(num_class=config.num_class).cuda()
# model = load_premodel(model)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
trainer(model, optimizer, criterion, traindata, train_dataloader, evaldata, eval_dataloader, is_aux=False, is_evaluate=True)

torch.save(model, '/home/froven/桌面/semantic-segmentation model/model_parameter/_15_DFANet.pth')
