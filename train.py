import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import datetime

from torch.optim import lr_scheduler
from helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, add_weight_decay
from helper_functions.treesatai import TreeSatAIDataset
from models import create_model
from loss_functions.losses import AsymmetricLoss
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='PyTorch TreeSatAI Training')
# parser.add_argument('data', metavar='DIR', help='path to dataset', default='/home/MSCOCO_2014/')
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--backbone-name', default='tresnet_l')
parser.add_argument('--model-path', default='/data/shilong/data/pretrained/tresnet_l_448.pth', type=str)
parser.add_argument('--num-classes', default=15)
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=224, type=int,
                    metavar='N', help='input image size (default: 448)')
# parser.add_argument('--thre', default=0.8, type=float,
#                     metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--print-freq', '-p', default=64, type=int,
                    metavar='N', help='print frequency (default: 64)')
parser.add_argument('--multi', action='store_true', help='using dataparallel')
# config data
parser.add_argument('--img_dir', default='', type=str)
parser.add_argument('--csv_path', default='', type=str)
parser.add_argument('--output-path', default='./models/coco/res18/', type=str)
parser.add_argument('--subsampling', action='store_true')
parser.add_argument('--sub-h', default=4, type=int)
parser.add_argument('--sub-w', default=4, type=int)
parser.add_argument('--min-superpixels', default=6, type=int)

def main():
    args = parser.parse_args()
    args.do_bottleneck_head = False

    # Setup model
    print('creating model...')
    model = create_model(args).cuda()

    # COCO Data loading
    normalize = transforms.Normalize(mean=[0, 0, 0],
                                     std=[1, 1, 1])

    train_dataset = TreeSatAIDataset(
        img_dir=args.img_dir,
        csv_path=args.csv_path,
        split='train',
        transform=transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            CutoutPIL(cutout_factor=0.5),
            RandAugment(),
            transforms.ToTensor(),
            normalize,
        ])
    )

    val_dataset = TreeSatAIDataset(
        img_dir=args.img_dir,
        csv_path=args.csv_path,
        split='val',
        transform=transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            normalize,
        ])
    )

    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # Actuall Training
    train_multi_label_treesatai(model, train_loader, val_loader, args.lr, args)


def train_multi_label_treesatai(model, train_loader, val_loader, lr, args):
    # ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    # set optimizer
    Epochs = 80
    weight_decay = 1e-4
    criterion = AsymmetricLoss(gamma_neg=0, gamma_pos=0, clip=0, disable_torch_grad_focal_loss=True)  ## CE loss
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)

    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    highest_epoch = 0
    for epoch in range(Epochs):
        for i, (inputData, target) in enumerate(train_loader):
            # break
            inputData = inputData.cuda()
            target = target.cuda()  # (batch,3,num_classes)

            with autocast():  # mixed precision
                output = model(inputData)  # sigmoid will be done in loss !

            mask = torch.ones_like(output[1])
            for j in range(mask.shape[0]):
                mask[j, torch.argmax(output[1][j, :])] = 0

            act = torch.nn.Softmax(dim=1)
            loss = criterion(output[0].float(), target) + 1 * (
                    1 / (output[1].shape[0] * output[1].shape[1])) * torch.sum(
                (act(output[1]) * mask) ** 2)

            model.zero_grad()

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            if i % 500 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                print('[{}] : Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                      .format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], \
                              loss.item()))

        if epoch >= 0:

            model.eval()
            mAP_score = validate_multi(val_loader, model)

            try:
                if not os.path.exists(args.output_path):
                    os.makedirs(args.output_path)
                torch.save(model.state_dict(), os.path.join(
                    args.output_path, 'model-{}-{:.2f}.ckpt'.format(epoch, mAP_score)))
            except:
                pass

            model.train()
            if mAP_score > highest_mAP:
                highest_epoch = epoch
                highest_mAP = mAP_score
                try:
                    if not os.path.exists(args.output_path):
                        os.makedirs(args.output_path)
                    torch.save(model.state_dict(), os.path.join(
                        args.output_path, 'model-highest.ckpt'))
                except:
                    pass
            print('[{}] : current_mAP = {:.2f}, highest_mAP = {:.2f},  highest_epoch = {}\n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), mAP_score, highest_mAP,
                                                                                             highest_epoch))


def validate_multi(val_loader, model):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    # preds_ema = []
    targets = []
    with torch.no_grad():
        with autocast():
            for i, (input, target) in enumerate(tqdm(val_loader)):
                # target = target
                target = target

                output_regular = Sig(model(input.cuda())).cpu()

                preds_regular.append(output_regular.detach().cpu())
                targets.append(target.detach().cpu())

    # import ipdb; ipdb.set_trace()
    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    return mAP_score_regular


if __name__ == '__main__':
    main()
