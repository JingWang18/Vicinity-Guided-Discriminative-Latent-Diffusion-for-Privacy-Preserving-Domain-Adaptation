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

import pdb

from torch.autograd import Variable

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LambdaLR

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

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

    # return nn.CrossEntropyLoss()(logits, labels)
    return loss.CrossEntropyLabelSmooth(num_classes=args.batch_size, epsilon=args.contrastive_epsilon)(logits, labels)
	###############################


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
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                      drop_last=False)

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
        alpha_end =((t+1)/num_step)
        d = model(x_alpha.unsqueeze(-1), torch.tensor(alpha_start).repeat(x_alpha.shape[0]).cuda().unsqueeze(-1)).squeeze()
        x_alpha = x_alpha + (alpha_end-alpha_start)*d

    return x_alpha

def cal_acc(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            index = data[-1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))

            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy * 100


def train_target_adapt(args):

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
    modelpath = args.output_dir_src + '/source_U.pt'
    netU.load_state_dict(torch.load(modelpath))
    
    netU.eval()
    for k, v in netU.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    for k, v in netC.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    num_sample = len(dset_loaders['target'].dataset)

    # initialize feature bank
    feat_bank = torch.randn(num_sample, 2048) # 2048
    bottle_bank = torch.randn(num_sample, 256).cuda() # 2048
    score_bank = torch.randn(num_sample, args.class_num).cuda()

    source_update = True
    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = next(iter_test)

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0:
            netF.eval()
            netB.eval()
            netC.eval()
            feat_bank, bottle_bank, score_bank, prob_lookup = obtain_bank(source_update, dset_loaders['test'], feat_bank, bottle_bank, score_bank, netF, netB, netC, args)
            netF.train()
            netB.train()
            netC.train()

        alpha = (1 + 10 * iter_num / max_iter)**(-5.0) * 1.0

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test = netF(inputs_test)
        feat_norm = F.normalize(features_test)

        bottleneck = netB(features_test)
        outputs_test = netC(bottleneck)
        softmax_out = nn.Softmax(dim=1)(outputs_test)

        feat_bank[tar_idx] = features_test.detach().clone().cpu()
        bottle_bank[tar_idx] = bottleneck.detach().clone()
        score_bank[tar_idx] = outputs_test.detach().clone()

        if args.contrastive:

            distance_knn = feat_norm.detach().clone().cpu() @ F.normalize(feat_bank).T
            _, idx_knn = torch.topk(distance_knn,
                dim=-1,
                largest=True,
                k = args.K
                )
            idx_knn = idx_knn[:,1:]

            distance_diffuse = feat_norm.detach().clone().cpu() @ F.normalize(feat_bank).T
            _, idx_diffuse = torch.topk(distance_diffuse,
                dim=-1,
                largest=True,
                k = args.K_diffuse
                )
            idx_diffuse = idx_diffuse[:,1:]

##############################################################################################
            # k-NN features
            feat_knn = feat_bank[idx_knn]
            feat_knn = torch.mean(feat_knn, 1).cuda()
##############################################################################################

            feat_diffuse = feat_bank[idx_diffuse]
            mu = torch.mean(feat_diffuse, 1).cuda()
            std = torch.std(feat_diffuse, 1).cuda()

            ###############################################################
            #reparameterize the mean of predictions of kNNs on target model 
            #with the mean of predictions of kNNs on source model
            ###############################################################

            z0 = mu + std*torch.randn_like(std)

            ####################################
            ### Note that z follows Gaussian ###
            ####################################

            feat_alpha_diffuse = sample_iadb(netU, z0, args.diffusion_step)

            feat_agg = (feat_alpha_diffuse + mu)/2
            y_agg = netC(netB(feat_agg))
            softmax_agg = nn.Softmax(dim=1)(y_agg)

            total_loss = 0

            total_loss += contrastive_loss(query=softmax_out, positive=softmax_agg, temp=args.temperature)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dset == 'VISDA-C':
                feat_bank, _,score_bank,_ = obtain_bank(source_update, dset_loaders['test'], feat_bank, bottle_bank, score_bank, netF, netB, netC, args)
                acc_s_te, feat_bank, score_bank = cal_acc(dset_loaders['test'], feat_bank, score_bank, netF, netB, netC)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter,
                                                                            acc_s_te) + '\n' + acc_list
            else:
                feat_bank, _,score_bank,_ = obtain_bank(source_update, dset_loaders['test'], feat_bank, bottle_bank, score_bank, netF, netB, netC, args)
                acc_s_te, feat_bank, score_bank = cal_acc(dset_loaders['test'], feat_bank, score_bank, netF, netB, netC)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            netF.train()
            netB.train()
            netC.train()

    if args.issave:
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))

    return netF, netB, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def obtain_bank(source_update, loader, feat_bank, bottle_bank, score_bank, netF, netB, netC, args):
    num_sample = len(loader.dataset)
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            indx = data[2]
            inputs = inputs.cuda()
            feas = netF(inputs)
            feas_extract = netB(feas)
            outputs = netC(feas_extract)

            feat_bank[indx] = feas.detach().clone().cpu()
            bottle_bank[indx] = feas_extract.detach().clone()
            score_bank[indx] = outputs.detach().clone()  #.cpu()

                
    source_update = False
    fea_lookup =None
    return feat_bank, bottle_bank, score_bank, fea_lookup


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DiffusionSFDA')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=40, help="max iterations")
    parser.add_argument('--interval', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size") # 32
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office',
                        choices=['office', 'office-home'])
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate") #1e-3
    parser.add_argument('--net', type=str, default='resnet101', help="resnet50, res101")
    parser.add_argument('--seed', type=int, default=2021, help="random seed") # 1

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=0.1)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--K_diffuse', type=int, default=16)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='weight')
    parser.add_argument('--output_src', type=str, default='weight')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)

    # contrastive 
    parser.add_argument('--temperature', default=0.13, type=float,
                        help='contrastive temperature (default: 0.13)')
    parser.add_argument('--contrastive_epsilon', default=0.1, type=float,
                        help='epsilon for the smooth CrossEntopy (default: 0.1)')
    # diffusion
    parser.add_argument('--diffusion_step', type=int, default=16, help="number of steps for diffusion")

    args = parser.parse_args()
    args.contrastive = True

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'Real_World']
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31

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

        folder = '/home/datasets'

        args.s_dset_path = folder +'/'+ args.dset + '/' + names[args.s] + '.txt'
        args.t_dset_path = folder +'/'+ args.dset + '/'+ names[args.t]+ '.txt'
        args.test_dset_path = folder +'/'+ args.dset + '/' + names[args.t] + '.txt'

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper() + names[args.t][0].upper())
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)
        if args.da == 'pda':
            args.gent = ''
            args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        print(print_args(args))

        train_target_adapt(args)
