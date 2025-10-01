import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random
import pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from collections import defaultdict

from torch.autograd import Variable

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LambdaLR

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=-1)
    return entropy

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsets["source_tr"] = ImageList_idx(txt_src, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True)
    dsets["source_te"] = ImageList_idx(txt_src, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True)
    dsets["test"] = ImageList(txt_src, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*2, shuffle=True, num_workers=args.worker, drop_last=True)


    return dset_loaders

def warmup(epoch, lr):
    if epoch <= 10:  # Warmup for the first 10 epochs
        return lr * (epoch + 1) / 10
    else:
        return lr

# Define the cosine decay function
def cosine_decay(epoch, lr):
    return 0.5 * lr * (1 + np.cos(np.pi * epoch / args.max_epoch))  # Adjust 1000 to the total number of epochs

@torch.no_grad()
def sample_iadb(model, x0, num_step):

    x_alpha = x0
    
    for t in range(num_step):
        alpha_start = (t/num_step)
        alpha_end = ((t+1)/num_step)
        alpha = torch.tensor(alpha_start).repeat(x_alpha.shape[0]).unsqueeze(-1).cuda()
        d = model(x_alpha.unsqueeze(-1), alpha).squeeze()
        x_alpha = x_alpha + (alpha_end-alpha_start)*d

    return x_alpha

def cal_acc(loader, feat_bank, netF, netB, netC, netU, start_test, args):
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            index = data[-1]
            inputs = inputs.cuda()

            features_test = netF(inputs)
            feat_norm = F.normalize(features_test)
            outputs_test = netC(netB(features_test))

            distance = feat_norm.detach().clone().cpu() @ F.normalize(feat_bank).T
            _, idx_near = torch.topk(distance,
                dim=-1,
                largest=True,
                k = args.K
                )
            idx_near = idx_near[:,1:]
            feat_near = feat_bank[idx_near]
            mu = torch.mean(feat_near, 1).cuda()
            std = torch.std(feat_near, 1).cuda()

            z0 = mu + std*torch.randn_like(std)
            bottle_alpha_diffuse = sample_iadb(netU, z0, args.diffusion_step)

            logits_diffuse = netC(netB(bottle_alpha_diffuse))
            outputs = nn.Softmax(dim=1)(logits_diffuse)

            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    return accuracy*100


def train_source_diffusion(args):

    dset_loaders = data_load(args)

    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    netU = network.ConditionalUNet().cuda()

    # load source pre-trained model
    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    
    netU.train()
    netF.eval()
    netB.eval()
    netC.eval()

    optimizer_UNet = optim.Adam(netU.parameters(), lr=args.lr)

    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    num_sample = len(dset_loaders['source_tr'].dataset)
    acc_init = 0

    # initialize feature bank
    feat_bank = torch.randn(num_sample, 2048) # 2048
    score_bank = torch.randn(num_sample, args.class_num).cuda()

    print('Initlizing Bank...')

    with torch.no_grad():
        iter_source = iter(dset_loaders["source_tr"])
        for _ in range(len(dset_loaders["source_tr"])):
            data = next(iter_source)
            inputs = data[0]
            labels = data[1]
            indx = data[2]
            inputs = inputs.cuda()
            feas = netF(inputs)
            feas_extract = netB(feas)
            logits = netC(feas_extract)

            feat_bank[indx] = feas.detach().clone().cpu()
            score_bank[indx] = logits.detach().clone()

    print('Start Training...')

    while iter_num < max_iter:
        try:
            inputs_test, labels_source, src_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["source_tr"])
            inputs_test, labels_source, src_idx = next(iter_test)

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()
        labels_source = labels_source.cuda()

        iter_num += 1

        features_test = netF(inputs_test)
        feat_norm = F.normalize(features_test)

        bottleneck = netB(features_test)
        outputs_test = netC(bottleneck)
        softmax_out = nn.Softmax(dim=1)(outputs_test)

        with torch.no_grad():
            feat_bank[src_idx] = features_test.detach().clone().cpu()
            score_bank[src_idx] = outputs_test.detach().clone()

        distance = feat_norm.detach().clone().cpu() @ F.normalize(feat_bank).T
        _, idx_near = torch.topk(distance,
            dim=-1,
            largest=True,
            k = args.K
            )

        feat_near = feat_bank[idx_near]
        feat_near_mean = torch.mean(feat_near, 1).cuda()
        feat_near_std = torch.std(feat_near, 1).cuda()

        mu = feat_near_mean
        std = feat_near_std
        z = mu + std*torch.randn_like(std)

        bs = z.shape[0]
        alpha = torch.rand(bs).cuda()

        feat_alpha = alpha.view(-1,1) * features_test + (1-alpha).view(-1,1) * z

        # diffusion_loss = torch.mean((netU(feat_alpha.unsqueeze(-1), alpha.unsqueeze(-1)).squeeze() - (features_test - feat_near_mean))**2)
        diffusion_loss  = nn.L1Loss()(netU(feat_alpha.unsqueeze(-1), alpha.unsqueeze(-1)).squeeze(), (features_test - feat_near_mean))

        total_loss = 0
        total_loss += diffusion_loss

        netU.eval()
        z0 = mu + std*torch.randn_like(std)
        bottle_alpha_diffuse = sample_iadb(netU, z0, args.diffusion_step)
        y_alpha_diffuse = netC(netB(bottle_alpha_diffuse))
        softmax_alpha = nn.Softmax(dim=1)(y_alpha_diffuse)
        netU.train()

        total_loss += loss.CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(softmax_alpha, labels_source)

        optimizer_UNet.zero_grad()
        total_loss.backward()
        optimizer_UNet.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:

            netB.eval()
            netC.eval()
            netU.eval()

            if args.dset=='VISDA-C':
                acc_s_te = cal_acc(dset_loaders['source_te'], feat_bank, netF, netB, netC, netU, True, args)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te = cal_acc(dset_loaders['source_te'], feat_bank, netF, netB, netC, netU, True, args)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netU = netU.state_dict()

            netU.train()
                
    torch.save(best_netU, osp.join(args.output_dir_src, "source_U.pt"))

    return netF, netB, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DiffusionSFDA')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=5, help="max iterations")
    parser.add_argument('--interval', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size") # 32
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='visda-2017')
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate") #1e-3
    parser.add_argument('--net', type=str, default='resnet101', help="resnet50, res101")
    parser.add_argument('--seed', type=int, default=2021, help="random seed")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--K', type=int, default=16)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='weight/source/')
    parser.add_argument('--output_src', type=str, default='weight/source/')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])

    # diffusion
    parser.add_argument('--diffusion_step', type=int, default=16, help="number of steps for diffusion")

    args = parser.parse_args()
    args.contrastive = True

    if args.dset == 'visda-2017':
        names = ['train', 'validation']
        args.class_num = 12

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = 'home/datasets'

        args.s_dset_path = folder + '/' + args.dset + '/' + names[args.s] + '_list.txt'
        args.test_dset_path = folder + '/' + args.dset + '/' + names[args.t] + '_list.txt'

        args.output_dir_src = osp.join(args.output, args.da, args.dset, names[args.s].upper())
        args.name_src = names[args.s]
        if not osp.exists(args.output_dir_src):
            os.system('mkdir -p ' + args.output_dir_src)
        if not osp.exists(args.output_dir_src):
            os.mkdir(args.output_dir_src)

        args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()

        print(print_args(args))
        train_source_diffusion(args)
