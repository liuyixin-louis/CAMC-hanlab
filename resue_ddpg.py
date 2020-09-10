# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

import torch
import numpy as np
import argparse
from copy import deepcopy
torch.backends.cudnn.deterministic = True

from env.channel_pruning_env import ChannelPruningEnv
from lib.agent import DDPG
from lib.utils import get_output_folder

from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description='AMC search script')

    parser.add_argument('--job', default='resue_test', type=str, help='support option: resue_test')
    parser.add_argument('--suffix', default=None, type=str, help='suffix to help you remember what experiment you ran')

    # env
    parser.add_argument('--model', default='mobilenet', type=str, help='model to prune')
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset to use (cifar/imagenet)')
    parser.add_argument('--data_root', default=None, type=str, help='dataset path') # 数据集的位置
    
    
    parser.add_argument('--lbound', default=0.2, type=float, help='minimum preserve ratio') # 最低的保留率率
    parser.add_argument('--rbound', default=1., type=float, help='maximum preserve ratio')  # 最大的保留率
    parser.add_argument('--reward', default='acc_reward', type=str, help='Setting the reward') 
    parser.add_argument('--acc_metric', default='acc5', type=str, help='use acc1 or acc5') # 准确率测量方式
    parser.add_argument('--use_real_val', dest='use_real_val', action='store_true') 
    parser.add_argument('--ckpt_path', default=None, type=str, help='manual path of checkpoint')
    parser.add_argument('--n_calibration_batches', default=60, type=int,
                        help='n_calibration_batches')
    parser.add_argument('--n_points_per_layer', default=10, type=int,
                        help='method to prune (fg/cp for fine-grained and channel pruning)') 
    parser.add_argument('--channel_round', default=8, type=int, help='Round channel to multiple of channel_round')
    # ddpg
    parser.add_argument("--resume_ddpg_checkpoint",default=None, type=str, help='resume ddpg checkpoint')
    parser.add_argument('--hidden1', default=300, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--lr_c', default=1e-3, type=float, help='learning rate for actor')
    parser.add_argument('--lr_a', default=1e-4, type=float, help='learning rate for actor')
    parser.add_argument('--warmup', default=100, type=int,
                        help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=1., type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=100, type=int, help='memory size for each layer')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
    # noise (truncated normal distribution)
    parser.add_argument('--init_delta', default=0.5, type=float,
                        help='initial variance of truncated normal distribution')
    parser.add_argument('--delta_decay', default=0.95, type=float,
                        help='delta decay during exploration')
    # training
    parser.add_argument('--max_episode_length', default=1e9, type=int, help='')
    parser.add_argument('--output', default='./logs', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--test_episode', default=800, type=int, help='test iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')
    parser.add_argument('--n_gpu', default=1, type=int, help='number of gpu to use')
    parser.add_argument('--n_worker', default=16, type=int, help='number of data loader worker')
    parser.add_argument('--data_bsize', default=50, type=int, help='number of data batch size')

    
    return parser.parse_args()


def get_model_and_checkpoint(model, dataset, checkpoint_path, n_gpu=1):
    if model == 'mobilenet' and dataset == 'imagenet':
        from models.mobilenet import MobileNet
        net = MobileNet(n_class=1000)
    elif model == 'mobilenetv2' and dataset == 'imagenet':
        from models.mobilenet_v2 import MobileNetV2
        net = MobileNetV2(n_class=1000)
    elif model == "mobilenet" and dataset =="cifar10":
        from models.mobilenet import MobileNet
        net = MobileNet(n_class=10)
    else:
        raise NotImplementedError
    # if model=="mobilenetv2":
    #     try:
    #         from torch.hub import load_state_dict_from_url
    #     except ImportError:
    #         from torch.utils.model_zoo import load_url as load_state_dict_from_url
    #     state_dict = load_state_dict_from_url(
    #         'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', progress=True)
    #     net.load_state_dict(state_dict)
    # else:
    sd = torch.load(checkpoint_path)
    if 'state_dict' in sd:  # a checkpoint but not a state_dict
        sd = sd['state_dict']
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    net.load_state_dict(sd)
    net = net.cuda()
    if n_gpu > 1:
        net = torch.nn.DataParallel(net, range(n_gpu))

    return net, deepcopy(net.state_dict())


def test(num_episode, agent, env, output):
    env.set_output(output)
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    T = []  # trajectory
    while episode < num_episode:  # counting based on episode

        # 第一层
        if observation is None:
            observation = deepcopy(env.reset()) 

        # agent pick action since the ddpg agent has already been trained
        action = agent.select_action(observation, episode=episode)

        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action,episode)
        observation2 = deepcopy(observation2)

        T.append([reward, deepcopy(observation), deepcopy(observation2), action, done])

        # agent 保存模型
        if episode % 10 == 0:
            agent.save_model(output)

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done:  # end of episode
            
            
            print('#{}: episode_reward:{:.4f} acc: {:.4f},acc_:{:.4f}, ratio: {:.4f},TargetRatio: {:.4f},done:{:.4f},strategy:{}'.format(episode, episode_reward,
                                                                                 info['accuracy'],info['accuracy_'],
                                                                                 info['compress_ratio'],env.preserve_ratio,info['compress_ratio']/env.preserve_ratio,info['strategy']))
            final_reward = T[-1][0] # 最后的奖励

            # our random number
            nb = [0.3,0.5,0.7].index(env.preserve_ratio)
            preserve_rate = env.preserve_ratio

            
            tfwriter.add_scalar('target_ratio/now',env.preserve_ratio)
            tfwriter.add_scalar('reward/lastward_{}'.format(preserve_rate), final_reward, episode)
            tfwriter.add_scalar('reward/best_{}'.format(preserve_rate), env.best_reward[nb], episode)
            tfwriter.add_scalar('info/accuracy_{}'.format(preserve_rate), info['accuracy'], episode)
            tfwriter.add_scalar('info/other_accuracy_{}'.format(preserve_rate), info['accuracy_'], episode)
            tfwriter.add_scalar('info/compress_ratio_{}'.format(preserve_rate), info['compress_ratio'], episode)
            tfwriter.add_text('info/best_policy_{}'.format(preserve_rate), str(env.best_strategy[env.curr_prunRatio_index]), episode)

            # record the preserve rate for each layer
            for i, preserve_rate in enumerate(env.strategy[env.curr_prunRatio_index]):
                tfwriter.add_scalar('preserve_rate/{}'.format(i), preserve_rate, episode)

            # change the env preserve ratio
            env.change()

            # 重置
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            T = []


def export_model(env, args):
    assert args.ratios is not None or args.channels is not None, 'Please provide a valid ratio list or pruned channels'
    assert args.export_path is not None, 'Please provide a valid export path'
    env.set_export_path(args.export_path)

    print('=> Original model channels: {}'.format(env.org_channels))
    if args.ratios:
        ratios = args.ratios.split(',')
        ratios = [float(r) for r in ratios]
        assert  len(ratios) == len(env.org_channels)
        channels = [int(r * c) for r, c in zip(ratios, env.org_channels)]
    else:
        channels = args.channels.split(',')
        channels = [int(r) for r in channels]
        ratios = [c2 / c1 for c2, c1 in zip(channels, env.org_channels)]
    print('=> Pruning with ratios: {}'.format(ratios))
    print('=> Channels after pruning: {}'.format(channels))

    for r in ratios:
        env.step(r)

    return

def get_model_and_checkpoint(model, dataset, checkpoint_path, n_gpu=1):
    net = None
    cadene = None

    if args.dataset == 'cifar10':
        if args.model == "mobilenet":
            from models.mobilenet import MobileNet
            net = MobileNet(n_class=10)
        
        elif args.model == "preresnet":
            from models.preresnet import PreResNet
            net = PreResNet(depth=56, num_classes=1000)
        # elif args.model == "alexnet":
        #     from models.alexnet import AlexNet
        #     net = AlexNet(n_class=10)
        # else:
        #     net = _create_cifar10_model(arch, pretrained)

    elif args.dataset == 'imagenet':
        if args.model =="mobilenet":
            from models.mobilenet import MobileNet
            net = MobileNet(n_class=1000)
        
        elif args.model == "resnet50":
            from models.resnet import ResNet
            net = ResNet(depth=50, num_classes=1000)
            
        # else:
        #     net, cadene = _create_imagenet_model(arch, pretrained)

    # elif args.dataset == 'mnist':
    #     net = _create_mnist_model(arch, pretrained)

    if net is None:
        raise NotImplementedError

    # if model=="mobilenetv2":
    #     try:
    #         from torch.hub import load_state_dict_from_url
    #     except ImportError:
    #         from torch.utils.model_zoo import load_url as load_state_dict_from_url
    #     state_dict = load_state_dict_from_url(
    #         'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', progress=True)
    #     net.load_state_dict(state_dict)
    # else:

    sd = torch.load(checkpoint_path)
    if 'state_dict' in sd:  # a checkpoint but not a state_dict
        sd = sd['state_dict']
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    net.load_state_dict(sd)
    net = net.cuda()
    if n_gpu > 1:
        net = torch.nn.DataParallel(net, range(n_gpu))
        net = net.cuda()

    return net, deepcopy(net.state_dict())


if __name__ == "__main__":
    # 获取参数
    args = parse_args()

    # 设置种子
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # 获取模型和检查点
    model, checkpoint = get_model_and_checkpoint(args.model, args.dataset, checkpoint_path=args.ckpt_path,
                                                 n_gpu=args.n_gpu)

    # 通道剪裁环境
    env = ChannelPruningEnv(model, checkpoint, args.dataset,
                            n_data_worker=args.n_worker, batch_size=args.data_bsize,
                            args=args, export_model=args.job == 'export')

    if args.job == 'reuseDDPG_test':
        # build folder and logs
        base_folder_name = '{}_{}_search'.format(args.model, args.dataset)
        if args.suffix is not None:
            base_folder_name = base_folder_name + '_' + args.suffix
        args.output = get_output_folder(args.output, base_folder_name)
        print('=> Saving logs to {}'.format(args.output))
        tfwriter = SummaryWriter(logdir=args.output)
        # text_writer = open(os.path.join(args.output, 'log.txt'), 'a+')
        print('=> Output path: {}...'.format(args.output))

        # 获取状态和动作的数目
        nb_states = env.layer_embedding.shape[1] # 第二个维度的长度
        nb_actions = 1  # just 1 action here

        args.rmsize = args.rmsize * len(env.prunable_idx)  # for each layer 每个可以剪裁的通道环境的记忆库容量
        print('** Actual replay buffer size: {}'.format(args.rmsize))

        # 实例化一个agent
        agent = DDPG(nb_states, nb_actions, args)

        # 加载权重数据
        agent.load_weights(args.resume_ddpg_checkpoint)

        # 测试DDPG可否复用
        test(args.test_episode, agent, env, args.output)
    else:
        raise RuntimeError('Undefined job {}'.format(args.job))
