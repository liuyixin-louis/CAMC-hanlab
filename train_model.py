# Train Model from scratch

import os
import time
import argparse
import shutil
import math

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tensorboardX import SummaryWriter

from lib.utils import accuracy, AverageMeter, progress_bar, get_output_folder
from lib.data import get_dataset
from lib.net_measure import measure_model

# from lib.utils import EarlyStopping

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



def parse_args():
    parser = argparse.ArgumentParser(description='Train Network')
    parser.add_argument('--model', default='mobilenet', type=str, help='name of the model to train')
    parser.add_argument('--dataset', default='imagenet', type=str, help='name of the dataset to train')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--n_gpu', default=1, type=int, help='number of GPUs to use')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--n_worker', default=4, type=int, help='number of data loader worker')
    parser.add_argument('--lr_type', default='exp', type=str, help='lr scheduler (exp/cos/step3/fixed)')
    parser.add_argument('--n_epoch', default=150, type=int, help='number of epochs to train')
    parser.add_argument('--wd', default=4e-5, type=float, help='weight decay')
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')
    parser.add_argument('--data_root', default=None, type=str, help='dataset path')
    # resume
    parser.add_argument('--ckpt_path', default=None, type=str, help='checkpoint path to resume from')
    # run eval
    parser.add_argument('--eval', action='store_true', help='Simply run eval')
    parser.add_argument('--patience', default=5, type=int, help='early stopping patience; how long to wait after last time validation loss improved.')

    return parser.parse_args()



# def _create_mnist_model(arch, pretrained):
#     if pretrained:
#         raise ValueError("Model {} (MNIST) does not have a pretrained model".format(arch))
#     try:
#         model = mnist_models.__dict__[arch]()
#     except KeyError:
#         raise ValueError("Model {} is not supported for dataset MNIST".format(arch))
#     return model



# def _create_imagenet_model(arch, pretrained):
#     dataset = "imagenet"
#     cadene = False
#     model = None
#     if arch in RESNET_SYMS:
#         model = imagenet_extra_models.__dict__[arch](pretrained=pretrained)
#     elif arch in TORCHVISION_MODEL_NAMES:
#         try:
#             if is_inception(arch):
#                 model = getattr(torch_models, arch)(pretrained=pretrained, transform_input=False)
#             else:
#                 model = getattr(torch_models, arch)(pretrained=pretrained)
#             if arch == "mobilenet_v2":
#                 patch_torchvision_mobilenet_v2(model)
#         except NotImplementedError:
#             # In torchvision 0.3, trying to download a model that has no
#             # pretrained image available will raise NotImplementedError
#             if not pretrained:
#                 raise
#     if model is None and (arch in imagenet_extra_models.__dict__) and not pretrained:
#         model = imagenet_extra_models.__dict__[arch]()
#     if model is None and (arch in pretrainedmodels.model_names):
#         cadene = True
#         model = pretrainedmodels.__dict__[arch](
#             num_classes=1000,
#             pretrained=(dataset if pretrained else None))
#     if model is None:
#         error_message = ''
#         if arch not in IMAGENET_MODEL_NAMES:
#             error_message = "Model {} is not supported for dataset ImageNet".format(arch)
#         elif pretrained:
#             error_message = "Model {} (ImageNet) does not have a pretrained model".format(arch)
#         raise ValueError(error_message or 'Failed to find model {}'.format(arch))
#     return model, cadene

def get_model():
    """
    get the corresponding model arch of the specific dataset
    """
    SUPPORTED_DATASETS = ('imagenet', 'cifar10', 'mnist')

    # ensure the dataset is supported
    dataset = args.dataset.lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError('Dataset {} is not supported'.format(dataset))
    net = None
    cadene = None

    if args.dataset == 'cifar10':
        if args.model == "mobilenet":
            from models.mobilenet import MobileNet
            net = MobileNet(n_class=10)
        # else:
        #     net = _create_cifar10_model(arch, pretrained)

    elif args.dataset == 'imagenet':
        if args.model =="mobilenet":
            from models.mobilenet import MobileNet
            net = MobileNet(n_class=1000)
        # else:
        #     net, cadene = _create_imagenet_model(arch, pretrained)

    # elif args.dataset == 'mnist':
    #     net = _create_mnist_model(arch, pretrained)

    if net is None:
        raise NotImplementedError
    
    return net.cuda() if use_cuda else net


def train(epoch, train_loader):
    print('\nEpoch: %d' % epoch)
    net.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))


        # timing
        batch_time.update(time.time() - end)
        end = time.time()

        progress_bar(batch_idx, len(train_loader), 'Loss: {:.3f} | Acc1: {:.3f}% | Acc5: {:.3f}%'
                     .format(losses.avg, top1.avg, top5.avg))

    writer.add_scalar('loss/train', losses.avg, epoch)
    writer.add_scalar('acc/train_top1', top1.avg, epoch)
    writer.add_scalar('acc/train_top5', top5.avg, epoch)
    return losses.avg


def test(epoch, test_loader, save=True):
    global best_acc
    net.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            # timing
            batch_time.update(time.time() - end)
            end = time.time()

            progress_bar(batch_idx, len(test_loader), 'Loss: {:.3f} | Acc1: {:.3f}% | Acc5: {:.3f}%'
                         .format(losses.avg, top1.avg, top5.avg))

    if save:
        writer.add_scalar('loss/test', losses.avg, epoch)
        writer.add_scalar('acc/test_top1', top1.avg, epoch)
        writer.add_scalar('acc/test_top5', top5.avg, epoch)

        is_best = False
        if top1.avg > best_acc:
            best_acc = top1.avg
            is_best = True

        print('Current best acc: {}'.format(best_acc))
        save_checkpoint({
            'epoch': epoch,
            'model': args.model,
            'dataset': args.dataset,
            'state_dict': net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict(),
            'acc': top1.avg,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint_dir=log_dir)
    return losses.avg


def adjust_learning_rate(optimizer, epoch):
    if args.lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.n_epoch))
    elif args.lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = args.lr * (decay ** (epoch // step))
    elif args.lr_type == 'fixed':
        lr = args.lr
    else:
        raise NotImplementedError
    print('=> lr: {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, is_best, checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, 'ckpt.pth.tar')
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('.pth.tar', '.best.pth.tar'))


if __name__ == '__main__':

    # 预处理
    args = parse_args()
    

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    # if torch.cuda.is_available():
    #     torch.set_default_tensor_type('torch.cuda.FloatTensor')


    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    
    # 数据集
    print('=> Preparing data..')
    train_loader, val_loader, n_class = get_dataset(args.dataset, args.batch_size, args.n_worker,
                                                    data_root=args.data_root)


    # 模型

    # net = get_model()  # for measure
    # IMAGE_SIZE = 224 if args.dataset == 'imagenet' else 32
    # n_flops, n_params = measure_model(net, IMAGE_SIZE, IMAGE_SIZE)
    # print('=> Model Parameter: {:.3f} M, FLOPs: {:.3f}M'.format(n_params / 1e6, n_flops / 1e6))
    # del net

    net = get_model()  # real training

    if args.ckpt_path is not None:  # assigned checkpoint path to resume from
        print('=> Resuming from checkpoint..')
        checkpoint = torch.load(args.ckpt_path)
        sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        net.load_state_dict(sd)
    
    if use_cuda and args.n_gpu > 1:
        net = torch.nn.DataParallel(net, list(range(args.n_gpu)))

    criterion = nn.CrossEntropyLoss()
    print('Using SGD...')
    print('weight decay  = {}'.format(args.wd))
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    
    if args.eval:  # just run eval
        print('=> Start evaluation...')
        test(0, val_loader, save=False)
    else:  # train
        print('=> Start training...')
        print('Training {} on {}...'.format(args.model, args.dataset))
        log_dir = get_output_folder('./logs', '{}_{}_trainFromSratch'.format(args.model, args.dataset))
        print('=> Saving logs to {}'.format(log_dir))
        # tf writer
        writer = SummaryWriter(logdir=log_dir)
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        for epoch in range(start_epoch, start_epoch + args.n_epoch):
            """just Training"""
            lr = adjust_learning_rate(optimizer, epoch)
            loss_train = train(epoch, train_loader)
            loss_val = test(epoch, val_loader)
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(loss_val, net)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
        writer.close()
        # print('=> Model Parameter: {:.3f} M, FLOPs: {:.3f}M, best top-1 acc: {}%'.format(n_params / 1e6, n_flops / 1e6, best_acc))
