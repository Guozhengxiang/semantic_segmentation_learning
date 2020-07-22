import time
import torch
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


from utils import config, label_accuracy_score
from model import FCN, UNet, SegNet, PSPNet, ENet, LinkNet, ICNet, BiSeNet, DFANet


def evaluate(model, criterion, eval_dataloader, is_aux=False):
    model.eval()
    eval_loss = 0
    eval_acc = 0
    eval_acc_cls = 0
    eval_mean_iu = 0
    eval_fwavacc = 0

    for (val_img, val_label) in eval_dataloader:
            val_img = val_img.cuda()
            val_label = val_label.cuda()
            # forward
            if is_aux is True:
                aux_out1, aux_out2, main_out = model(val_img)
            else:
                main_out = model(val_img)
            val_loss = criterion(main_out, val_label)
            eval_loss += val_loss.item()

            label_pred = main_out.max(dim=1)[1].data.cpu().numpy()
            label_true = val_label.data.cpu().numpy()
            for lbt, lbp in zip(label_true, label_pred):
                acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, config.num_class)
                eval_acc += acc
                eval_acc_cls += acc_cls
                eval_mean_iu += mean_iu
                eval_fwavacc += fwavacc

    return eval_loss, eval_acc, eval_mean_iu


def show_a_result(model):
    plt.figure()
    eval_img = Image.open('/home/froven/桌面/newdata/test/image/IMG_2047_img.png')
    eval_label = Image.open('/home/froven/桌面/newdata/test/label/IMG_2047_label.png').convert('RGB')
    eval_img = F.crop(eval_img, 200, 500, config.crop_size_h, config.crop_size_w)
    eval_label = F.crop(eval_label, 200, 500, config.crop_size_h, config.crop_size_w)
    eval_img = np.array(eval_img)
    eval_label = np.array(eval_label)
    plt.subplot(1, 3, 1)
    plt.imshow(eval_img)
    plt.subplot(1, 3, 2)
    plt.imshow(eval_label)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    eval_img = transform(eval_img).float()
    with torch.no_grad():
        model.eval()
        if torch.cuda.is_available() and config.use_gpu:
            data = (eval_img.unsqueeze(0)).cuda()
        predict = model(data)[-1]
        predict = predict.max(1)[1].squeeze().cpu().data.numpy()
        predict = config.lb2image[predict]
        plt.subplot(1, 3, 3)
        plt.imshow(predict)
    plt.show()


def trainer(model, optimizer, criterion, traindata, train_dataloader, evaldata, eval_dataloader,
            is_aux=False, is_evaluate=False):
    for epoch in range(config.epoch):
        model.train()
        train_loss = 0
        train_acc = 0
        train_acc_cls = 0
        train_mean_iu = 0
        train_fwavacc = 0
        # scheduler.step()
        epoch_start_time = time.time()

        for step, (image, label) in enumerate(train_dataloader):
            image = image.cuda()
            label = label.cuda()
            if is_aux is True:
                aux_out1, aux_out2, main_out = model(image)
                loss = criterion(aux_out1, label) + criterion(aux_out2, label) + criterion(main_out, label)
            else:
                main_out = model(image)
                loss = criterion(main_out, label)
            # # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            label_pred = main_out.max(dim=1)[1].data.cpu().numpy()
            label_true = label.data.cpu().numpy()
            for lbt, lbp in zip(label_true, label_pred):
                acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, config.num_class)
                train_acc += acc
                train_acc_cls += acc_cls
                train_mean_iu += mean_iu
                train_fwavacc += fwavacc

        # # evaluation
        if is_evaluate:
            eval_loss, eval_acc, eval_mean_iu = evaluate(model, criterion, eval_dataloader, is_aux=is_aux)
            print('Epoch: {}, Train_loss: {:.5f}, Train_acc:{:.5f}, Train_mean_iou:{:.5f}, \
                          Eval_loss: {:.5f}, Eval Acc: {:.5f}, Eval Mean IU: {:.5f}, time:{:.5f}'.format(
                                 epoch, train_loss / len(traindata), train_acc / len(traindata),
                                 train_mean_iu / len(traindata), eval_loss / len(evaldata),
                                 eval_acc / len(evaldata), eval_mean_iu / len(evaldata),
                                 (time.time() - epoch_start_time)))
            a = str(train_loss / len(traindata))
        else:
            print('Epoch: {}, Train_loss: {:.5f}, Train_acc:{:.5f}, Train_mean_iou:{:.5f}, time:{:.5f}'.format(
                                 epoch, train_loss / len(traindata), train_acc / len(traindata),
                                 train_mean_iu / len(traindata), (time.time() - epoch_start_time)))
        # # # show the result
        # if epoch % config.step == 0 and epoch != 0:
        #     show_a_result(model)