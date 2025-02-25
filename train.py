import torchvision.transforms

from utils.data_loader import make_datapath_list, ImageDataset, ImageTransform
from models.UNet_with_PConv import PConvUNet
from models.loss import Losses
from models.metrics import Evaluator
from torchvision import models
from collections import OrderedDict
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import argparse
import time
import torch
import os

torch.manual_seed(44)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def fix_model_state_dict(state_dict):
    """
    remove 'module.' of data parallel
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict


def plot_log(data, save_model_name='model'):
    plt.cla()
    plt.plot(data, label='total_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.savefig('./logs/'+save_model_name+'.png')


def check_dir():
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    if not os.path.exists('./result'):
        os.mkdir('./result')


def get_parser():
    parser = argparse.ArgumentParser(
        prog='Image Inpainting using Partial Convolutions',
        usage='python3 main.py',
        description='This module demonstrates image inpainting using U-Net with partial convolutions.',
        add_help=True)

    parser.add_argument('-e', '--epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=6, help='Batch size')
    parser.add_argument('-s', '--image_size', type=int, default=256)
    parser.add_argument('-f', '--finetune', action='store_true')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr_finetune', type=float, default=5e-5)


    return parser


def train_model(pconv_unet, dataloader, val_dataset, num_epochs, parser, save_model_name='model'):

    check_dir()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pconv_unet.to(device)

    """use GPU in parallel"""
    if device == 'cuda':
        pconv_unet = torch.nn.DataParallel(pconv_unet)
        print("parallel mode")

    print("device:{}".format(device))

    if parser.finetune:
        lr = parser.lr_fine
        pconv_unet.fine_tune = True
    else:
        lr = parser.lr

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, pconv_unet.parameters()), lr=lr)
    criterion = Losses().to(device)
   
    torch.backends.cudnn.benchmark = True

    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    lambda_dict = {'valid': 1.0, 'hole': 6.0, 'perceptual': 0.05, 'style': 120, 'tv': 0.1, 'bfm': 0.1}

    iteration = 1
    losses = []

    for epoch in range(num_epochs):

        pconv_unet.train()
        t_epoch_start = time.time()

        epoch_loss = 0.0

        print('-----------')
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('(train)')

        for images, mask, gt in tqdm(dataloader):

            # if size of minibatch is 1, an error would be occurred.
            if images.size()[0] == 1:
                continue

            images = images.to(device)
            mask = mask.to(device)
            gt = gt.to(device)

            mini_batch_size = images.size()[0]

            output, _, fb = pconv_unet(images, mask, gt)
            loss_dict = criterion(images, mask, output, gt, fb)

            loss = torch.tensor(0.0)
            for key, _lambda in lambda_dict.items():
                loss += _lambda * loss_dict[key]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            iteration += 1

        t_epoch_finish = time.time()
        print('-----------')
        print('epoch {}'.format(epoch+1))
        print('total_loss:{:.4f}'.format(epoch_loss/batch_size))
        print('timer: {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))

        losses.append(epoch_loss/batch_size)
        t_epoch_start = time.time()
        plot_log(losses, save_model_name)

        if epoch % 1 == 0:
            torch.save(pconv_unet.state_dict(), 'checkpoints/'+save_model_name+'_'+str(epoch+1)+'.pth')
            pconv_unet.eval()
            evaluator = Evaluator(pconv_unet)
            metrics = evaluator.evaluate(val_dataset, device, '{:s}/test_{:d}.jpg'.format('result', epoch))
            print('metrics: PSNR = {:.2f}, SSIM = {:.2f}'.format(metrics['PSNR'], metrics['SSIM']))
    return pconv_unet


def main(parser):
    pconv_unet = PConvUNet()

    '''load'''
    # pconv_weights = torch.load('./checkpoints/PConvUNet_PConvUNet_1000.pth')
    # pconv_unet.load_state_dict(fix_model_state_dict(pconv_weights))

    train_img_list, val_img_list = make_datapath_list(iorm='img', path='img', phase='train', dataset_num=10000)
    mask_list = make_datapath_list(iorm='mask', path='')

    mean = (0.5,)
    std = (0.5,)
    size = (parser.image_size, 2 * parser.image_size)
    batch_size = parser.batch_size
    num_epochs = parser.epoch

    train_dataset = ImageDataset(img_list=train_img_list,
                                 img_transform=ImageTransform(size=size, mean=mean, std=std),
                                 mask_width=0.5)
    val_dataset = ImageDataset(img_list=val_img_list,
                               img_transform=ImageTransform(size=size, mean=mean, std=std),
                               mask_width=0.5)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # num_workers=4

    pconv_unet_update = train_model(pconv_unet, dataloader=train_dataloader,
                                    val_dataset=val_dataset, num_epochs=num_epochs,
                                    parser=parser, save_model_name='PConvUNet_Rectangle')


if __name__ == "__main__":
    parser = get_parser().parse_args()
    main(parser)
