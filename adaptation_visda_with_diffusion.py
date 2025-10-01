import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random
import pdb, math, copy
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import tqdm

from sklearn.manifold import TSNE
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

def contrastive_loss(query, positive, temp):

    feature = torch.cat([query, positive], dim=0)
    labels = torch.cat([torch.arange(query.shape[0]).repeat_interleave(1) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()
    similarity_matrix = feature @ feature.T

    A = torch.ones(labels.shape[0], 1, 1, dtype=torch.bool)
    mask = torch.block_diag(*A).cuda()

    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    logits = logits/temp

    # return loss.CrossEntropyLabelSmooth(num_classes=args.batch_size, epsilon=args.contrastive_epsilon)(logits, labels)
    return nn.CrossEntropyLoss()(logits, labels)
    ###############################

@torch.no_grad()
def sample_iadb(model, x0, num_step):
    x_alpha = x0
    for t in range(num_step):
        alpha_start = (t/num_step)
        alpha_end =((t+1)/num_step)
        d = model(x_alpha.unsqueeze(-1), torch.tensor(alpha_start).repeat(x_alpha.shape[0]).cuda().unsqueeze(-1)).squeeze()
        x_alpha = x_alpha + (alpha_end-alpha_start)*d

    return x_alpha

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter)**(-power)
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
        transforms.ToTensor(), normalize
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
        transforms.ToTensor(), normalize
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsize = len(txt_src)
    tr_size = int(0.9 * dsize)
    _, te_txt = torch.utils.data.random_split(txt_src,
                                              [tr_size, dsize - tr_size])
    tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"],
                                           batch_size=train_bs,
                                           shuffle=True,
                                           num_workers=args.worker,
                                           drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"],
                                           batch_size=train_bs,
                                           shuffle=True,
                                           num_workers=args.worker,
                                           drop_last=False)
    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"],
                                        batch_size=train_bs,
                                        shuffle=True,
                                        num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"],
                                      batch_size=train_bs,
                                      shuffle=False,
                                      num_workers=args.worker,
                                      drop_last=False)
    return dset_loaders


def cal_acc(loader, netF, netB, netC,args, flag=False):
    start_test = True
    num_sample = len(loader.dataset)
    var_all=[]

    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            index = data[-1]
            inputs = inputs.cuda()
            features = netF(inputs)
            fea = netB(features)
            if args.var:
                var_batch=fea.var()
                var_all.append(var_batch)

            outputs = netC(fea)
            softmax_out=nn.Softmax(dim=1)(outputs)

            if start_test:
                all_output = outputs.float().cpu()
                all_fea = features.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_fea = torch.cat((all_fea, features.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

        df_subset = pd.DataFrame()
        tsne = TSNE(n_components=2, verbose=0, perplexity=5, n_iter=600)
        tsne_results = tsne.fit_transform(all_fea)

        print('t-SNE done!')

        sns.set(font_scale=2)

        df_subset['tsne-2d-one'] = tsne_results[:,0]
        df_subset['tsne-2d-two'] = tsne_results[:,1]
        df_subset['y'] = all_label

        plt.figure(figsize=(20,18))

        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="y",
            palette=sns.color_palette('hls', 12),
            # palette="deep",
            data=df_subset,
            legend=True,
            alpha=0.3
        )
        fig_name = 'visualization/' + 'target_tsne' + str(args.s) + '_epoch' + '.png'
        plt.savefig(fig_name)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(
        torch.squeeze(predict).float() == all_label).item() / float(
            all_label.size()[0])

    import time
    
    splits = 4
    split_idx = num_sample//splits
    idx_near = torch.zeros(num_sample, 4, dtype=torch.long) 
    tic = time.time()
    print('Time consumed:', time.time() - tic)

    return accuracy * 100

def hyper_decay(x,beta=-2,alpha=1):
    weight=(1 + 10 * x)**(-beta) * alpha
    return weight

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=-1)
    return entropy

def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    netF = network.ResBase(res_name='resnet101').cuda()

    netB = network.feat_bottleneck(type=args.classifier,
                                   feature_dim=2048,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer,
                                   class_num=args.class_num,
                                   bottleneck_dim=args.bottleneck).cuda()

    netU = network.ConditionalUNet().cuda()

    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_U.pt'
    netU.load_state_dict(torch.load(modelpath))
    
    netU.eval()

    param_group = []
    param_group_c = []
    for k, v in netU.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * 0.1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * 1.0}]
        else:
            v.requires_grad = False

    for k, v in netC.named_parameters():
        if args.lr_decay2 > 0:
            param_group_c += [{'params': v, 'lr': args.lr * 1.0}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    optimizer_c = optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)

    #building feature bank
    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)
    feat_bank = torch.randn(num_sample, args.feat_bank_dim)
    bottle_bank = torch.randn(num_sample, args.bottleneck)
    score_bank = torch.randn(num_sample, args.class_num).cuda()

    netF.eval()
    netB.eval()
    netC.eval()

    ###########################################
    # for generating source pre-trained t-SNE #
    ###########################################
    # acc= cal_acc(dset_loaders['test'],netF,netB,netC,args,flag=True)

    #initialize
    print("Initialize...")
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
        # for i in range(1):
            data = next(iter_test)
            inputs = data[0]
            indx = data[-1]
            inputs = inputs.cuda()
            feature = netF(inputs)
            bottle = netB(feature)
            logits = netC(bottle)
            outputs = nn.Softmax(-1)(logits)

            feat_bank[indx] = feature.detach().clone().cpu()
            bottle_bank[indx] = bottle.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  #.cpu()

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()
    acc_log = 0
    
    real_max_iter = max_iter

    print("Start training...")

    for iter_num in tqdm.tqdm(range(real_max_iter)):
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = next(iter_test)

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()
        if True:
            alpha = (1 + 10 * iter_num / max_iter)**(-args.beta) * args.alpha
        else:
            alpha = args.alpha

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)

        total_loss = 0

        features_test = netF(inputs_test)
        bottle_test = netB(features_test)
        logits_test = netC(bottle_test)
        softmax_out = nn.Softmax(dim=1)(logits_test)

        feat_norm = F.normalize(features_test)
        bottle_norm = F.normalize(bottle_test)

        with torch.no_grad():
            feat_bank[tar_idx] = features_test.detach().clone().cpu()
            score_bank[tar_idx] = softmax_out.detach().clone()
            bottle_bank[tar_idx] = bottle_test.detach().clone().cpu()

            distance_diffuse = feat_norm.detach().clone().cpu() @ F.normalize(feat_bank).T
            _, idx_diffuse = torch.topk(distance_diffuse,
                dim=-1,
                largest=True,
                k = args.K_diffuse+1
                )
            idx_diffuse = idx_diffuse[:,1:]

            distance_near = bottle_norm.detach().clone().cpu() @ F.normalize(bottle_bank).T
            _, idx_near = torch.topk(distance_near,
                dim=-1,
                largest=True,
                k = args.K+1
                )
            idx_near = idx_near[:,1:]

            feat_diffuse = feat_bank[idx_diffuse]
            score_near = score_bank[idx_near]

        mu = torch.mean(feat_diffuse, 1).cuda()
        std = torch.std(feat_diffuse, 1).cuda()

        ################################################################
        # reparameterize the mean of predictions of kNNs on target model 
        # with the mean of predictions of kNNs on source model
        ################################################################
        z0 = mu + std*torch.randn_like(std)
        feat_alpha_diffuse = sample_iadb(netU, z0, args.diffusion_step)

        feat_agg = (feat_alpha_diffuse + mu)/2
        y_agg = netC(netB(feat_agg))
        softmax_agg = nn.Softmax(dim=1)(y_agg)

        total_loss += contrastive_loss(query=softmax_out, positive=softmax_agg, temp=args.temperature_diffusion)

        ###########################################################
        # k-NN predictions from the score bank
        ###########################################################

        score_near = torch.mean(score_near, 1)
        total_loss += nn.CrossEntropyLoss()(softmax_out, score_near)

        optimizer.zero_grad()
        optimizer_c.zero_grad()
        total_loss.backward()
        optimizer.step()
        optimizer_c.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            print("Calculate accuracy...")
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dset == 'visda-2017':
                acc= cal_acc(dset_loaders['test'],netF,netB,netC,args,flag=True)
                log_str = 'Task: {}, Iter:{}/{}, Epoch:{}/{};  Acc on target: {:.2f}'.format(
                    args.name, iter_num, max_iter, iter_num//len(dset_loaders["target"]), args.max_epoch, acc) + '\n' + 'T: '

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            netF.train()
            netB.train()
            netC.train()

    return netF, netB, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DND')
    parser.add_argument('--gpu_id',
                        type=str,
                        nargs='?',
                        default='0',
                        help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch',
                        type=int,
                        default=30,
                        help="max iterations")
    parser.add_argument('--interval', type=int, default=30) #150
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help="batch_size")
    parser.add_argument('--worker',
                        type=int,
                        default=4,
                        help="number of workers")
    parser.add_argument('--dset', type=str, default='visda-2017')
    parser.add_argument('--lr', type=float, default=3e-3, help="learning rate")
    parser.add_argument('--feat_bank_dim', type=float, default=2048, help="feature bank dimension")
    parser.add_argument('--net', type=str, default='resnet101')
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer',
                        type=str,
                        default="wn",
                        choices=["linear", "wn"])
    parser.add_argument('--classifier',
                        type=str,
                        default="bn",
                        choices=["ori", "bn"])
    parser.add_argument('--output', type=str, default='weight/target/')
    parser.add_argument('--output_src', type=str, default='weight/source/')
    parser.add_argument('--tag', type=str, default='AAD')
    parser.add_argument('--da', type=str, default='uda')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=5.0)
    parser.add_argument('--alpha_decay', default=True)
    parser.add_argument('--var', default=False, action='store_true')

    # neighborhood
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--K_diffuse', type=int, default=8)

    # contrastive 
    parser.add_argument('--temperature_diffusion', default=1.0, type=float,
                        help='softmax temperature for diffusion output (default: 0.13)')
    parser.add_argument('--contrastive_epsilon', default=0.1, type=float,
                        help='epsilon for the smooth CrossEntopy (default: 0.1)')
    # diffusion
    parser.add_argument('--diffusion_step', type=int, default=16, help="number of steps for diffusion")

    args = parser.parse_args()

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
        args.t_dset_path = folder + '/' + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + '/' + args.dset + '/' + names[args.t] + '_list.txt'

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset,
                                       names[args.s].upper())
        args.output_dir = osp.join(
            args.output, args.da, args.dset,
            names[args.s].upper() + '-' + names[args.t].upper())
        args.name = names[args.s].upper() + '-' + names[args.t].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.out_file = open(osp.join(args.output_dir, 'log_{}_K_{}.txt'.format(args.tag, args.K)), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        print(print_args(args))
        train_target(args)