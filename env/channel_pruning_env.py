
import time
import torch
import torch.nn as nn
from lib.utils import AverageMeter, accuracy, prGreen
from lib.data import get_split_dataset
from env.rewards import *
import math
from lib.basic_hook import *


import numpy as np
import copy

class ChannelPruningEnv:
    """
    Env for channel pruning search；
    """
    def __init__(self, model, checkpoint, data, args, n_data_worker=4,
                 batch_size=256, export_model=False):

        # support pruning_ratio (discrete)
        self.support_prun_ratio = [0.3,0.5,0.7]
        self.curr_prunRatio_index = 0
        self.curr_preserve_ratio = 0.3 # we will start from 0.3

        # default setting
        # self.prunable_layer_types = [torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear] # CNN和线性层，mobilenetv1适用，其他不一样

        #！

        # save options
        self.model = model
        # self.checkpoint = checkpoint
        self.n_data_worker = n_data_worker
        self.batch_size = batch_size
        self.data_type = data # 采用的数据集


        # options from args
        self.args = args
        self.lbound = args.lbound
        self.rbound = args.rbound
        

        
        self.use_real_val = args.use_real_val # 是否采用验证集


        self.n_calibration_batches = args.n_calibration_batches # 每一层收集
        self.n_points_per_layer = args.n_points_per_layer
        self.channel_round = args.channel_round #
        self.acc_metric = args.acc_metric
        self.data_root = args.data_root

        self.export_model = export_model    # bool


        # prepare data
        self._init_data()

        # # 1. 确定哪些要剪裁及保存该层的输入
        # self._build_index() # 
        # self.n_prunable_layer = len(self.prunable_idx) 
        # # 2. 收集那些要剪裁层的一些数据
        # self._extract_layer_information()
        # 
        # 
        # 1、2：确定剪裁层和收集数据
        self._build_index_extract_information()
        # 3. 建立状态嵌入表示（静态部分）
        self._build_state_embedding() 

        self.strategy_dict = [[1.0,1.0]]* len(self.model_list)
        self.n_prunable_layer = len(self.prunable_index)

        # now we save the model checkpoint
        torch.save(self.model.state_dict(), self.args.model_cached_ckp)
        self.checkpoint = self.args.model_cached_ckp


        # reset env for init
        self.reset()  # 清空环境 


        # 模型的一些指标
        self.org_acc ,self.org_acc_another = self._validate(self.val_loader, self.model)
        
        print('=> original {}: {:.3f}%'.format(self.acc_metric,self.org_acc))
        print('=> original another acc: {:.3f}%'.format(self.org_acc_another))
        self.org_model_size = sum(self.wsize_list)
        print('=> original weight size: {:.4f} M param'.format(self.org_model_size * 1. / 1e6))
        self.org_flops = sum(self.flops_list)
        print('=> FLOPs:')
        print([self.layers_info[idx]['flops']/1e6 for idx in sorted(self.layers_info.keys())])
        print('=> original FLOPs: {:.4f} M'.format(self.org_flops * 1. / 1e6))

        self.expected_preserve_computation = self.org_flops * self.curr_preserve_ratio

        self.reward = eval(args.reward) # get the reward function

        self.best_reward = dict(zip(self.support_prun_ratio,[-math.inf]*len(self.support_prun_ratio)))
        self.best_strategy = dict(zip(self.support_prun_ratio,[[]]*len(self.support_prun_ratio)))
        self.best_d_prime_list = dict(zip(self.support_prun_ratio,[[]]*len(self.support_prun_ratio)))
        self.org_w_size = sum(self.wsize_list)


        # log path init
        self.output = None


    def _build_index_extract_information(self):
        """建立索引并且提取每一层的信息"""
        # build index and add hook
        
        model =  self.model
        handler_collection = {}
        types_collection = set()

       
        
        def add_hooks(m):
            if len(list(m.children())) > 0: # 只处理叶子节点
                return
            # if hasattr(m, "total_ops") or hasattr(m, "total_params"):
            #     logging.warning("Either .total_ops or .total_params is already defined in %s. "
            #                     "Be careful, it might change your code's behavior." % str(m))

            m.register_buffer('total_ops', torch.zeros(1, dtype=torch.float64))
            m.register_buffer('total_params', torch.zeros(1, dtype=torch.float64))
            m.register_buffer('prun_weight', torch.zeros_like(m.weight)) # for prun ops convenient
            m.register_buffer('origin_weight', m.weight) # for prun ops convenient


            for p in m.parameters():
                m.total_params += torch.DoubleTensor([p.numel()])

            m_type = type(m)

            fn = None
            verbose = True
            if  m_type in register_hooks:
                fn = register_hooks[m_type]
                if m_type not in types_collection and verbose:
                    print("[INFO] Register %s() for %s." % (fn.__qualname__, m_type))
            else:
                if m_type not in types_collection and verbose:
                    prRed("[WARN] Cannot find rule for %s. Treat it as zero Macs and zero Params." % m_type)

            if fn is not None:
                handler = m.register_forward_hook(fn)
                # handler_collection.append(handler)
                handler_collection[m] = (m.register_forward_hook(fn),m.register_forward_hook(record_xy))
            types_collection.add(m_type)

            # extend the forward fn to record layer info
            # def new_forward(m):
            # def lambda_forward(x):
            #     m.input_feat = x.clone()
            #     y = m.old_forward(x)
            #     m.output_feat = y.clone()
            #     return y
            # return lambda_forward
            
            # m.old_forward = m.forward
            # m.forward = new_forward(m)
        
        
        self.prunable_index = []
        self.prunable_ops = []
        self.layers_info = {}
        self.wsize_list = []
        self.flops_list = []
        self.model_list = []
        self.wsize_prunable_list = []
        self.flops_prunable_list = []
        
        
        
        # let the images flow
        prev_training_status = model.training
        model.eval()
        model.apply(add_hooks)


        # with torch.no_grad():
        #     model(*inputs)
        
        
        
        def get_layer_type(layer):
            layer_str = str(layer)
            return layer_str[:layer_str.find('(')].strip()

        def dfs_count(module: nn.Module, prefix="\t") -> (int, int):
            """深度优先，仅计算最小可算flops的层；遇到约定的剪枝层时加入要列表做记录；"""

            total_ops, total_params = 0, 0
            for m in module.children():
                # if not hasattr(m, "total_ops") and not hasattr(m, "total_params"):  # and len(list(m.children())) > 0:
                #     m_ops, m_params = dfs_count(m, prefix=prefix + "\t")
                # else:
                #     m_ops, m_params = m.total_ops, m.total_params
                if m in handler_collection and not isinstance(m, (nn.Sequential, nn.ModuleList)):
                    m_ops, m_params = m.total_ops.item(), m.total_params.item()
                    self.model_list.append(m)
                    self.wsize_list.append(m_params)
                    self.flops_list.append(m_ops)
                    self.layers_info[m] = {"flops":m_ops,"params":m_params}
                    
                    
                else:
                    if get_layer_type(m) == "Bottleneck":
                        # this is the layer we need to prun in resnet50
                        n = len(self.model_list)
                        self.prunable_index.append(n+3) # conv2
                        self.prunable_index.append(n+6) # conv3
                        self.prunable_ops.append(m.conv2)
                        self.prunable_ops.append(m.conv3)

                        # flops
                        self.flops_prunable_list.append(m.conv2.total_ops.item())
                        self.flops_prunable_list.append(m.bn1.total_ops.item())
                        self.flops_prunable_list.append(m.conv1.total_ops.item())
                        self.flops_prunable_list.append(m.conv3.total_ops.item())
                        self.flops_prunable_list.append(m.bn2.total_ops.item())

                        # wsize
                        self.wsize_prunable_list.append(m.conv2.total_params.item())
                        self.wsize_prunable_list.append(m.bn1.total_params.item())
                        self.wsize_prunable_list.append(m.conv1.total_params.item())
                        self.wsize_prunable_list.append(m.conv3.total_params.item())
                        self.wsize_prunable_list.append(m.bn2.total_params.item())
                        
                    m_ops, m_params = dfs_count(m, prefix=prefix + "\t")
                total_ops += m_ops
                total_params += m_params
            #  print(prefix, module._get_name(), (total_ops.item(), total_params.item()))
            


            return total_ops, total_params
        

        
        # extract information for pruning
        self.data_saver = []
        self.layer_info_dict = {}
        print('=> Extracting information...')
        with torch.no_grad():
            for i_b, (input, target) in enumerate(self.train_loader):  # use image from train set
                if i_b == self.n_calibration_batches:
                    break
                self.data_saver.append((input.clone(), target.clone()))
                input_var = torch.autograd.Variable(input).cuda()

                # inference and collect stats
                _ = self.model(input_var)

                if i_b == 0:
                    # first batch: compute the ops and params
                    total_ops, total_params = dfs_count(model)
                    self.non_prunable_flops = sum(self.flops_list) - sum(self.flops_prunable_list)
                    self.non_prunable_wsize = sum(self.wsize_list) - sum(self.wsize_prunable_list)
                    assert len(self.model_list) == len(self.flops_list) 

                    

                for ops_layer in self.prunable_ops:
                    f_in_np = ops_layer.input_feat.data.cpu().numpy()
                    f_out_np = ops_layer.output_feat.data.cpu().numpy()
                    if len(f_in_np.shape) == 4:  # conv
                        if  ops_layer.weight.size(3) >= 1:  # 卷积
                            f_in2save, f_out2save = f_in_np, f_out_np
                    else:
                        assert False # this will never occur for resnet50
                        
                    if 'input_feat' not in self.layers_info[ops_layer]:
                        self.layers_info[ops_layer]['input_feat'] = f_in2save
                        self.layers_info[ops_layer]['output_feat'] = f_out2save
                    else:
                        self.layers_info[ops_layer]['input_feat'] = np.vstack(
                            (self.layers_info[ops_layer]['input_feat'], f_in2save))
                        self.layers_info[ops_layer]['output_feat'] = np.vstack(
                            (self.layers_info[ops_layer]['output_feat'], f_out2save))
        
        for m, (op_handler, xyrecord_handler) in handler_collection.items():
            op_handler.remove()
            xyrecord_handler.remove()
            m._buffers.pop("total_ops")
            m._buffers.pop("total_params")
        

        
        




    def set_output(self,output):
        import os
        path = os.getcwd()
        path = path + output[1:]
        self.output = path+'/log.txt'

    def step(self, action,epoch):
        """根据输入的动作，对环境进行修改并返回环境的反馈信息"""
        # Pseudo prune and get the corresponding statistics.
        # The real pruning happens till the end of all pseudo pruning

        # 
        # if self.visited[self.cur_ind]: # 被访问过，也就是这一层已经被处理过了
        #     action = self.strategy_dict[self.prunable_index[self.cur_ind]][0] #取出这一层的动作
        # else: # 没有处理过，调用action的截断函数

        action = self._action_wall(action)  # percentage to preserve
        preserve_idx = None # 该层保存的通道索引

        # prune and update action
        action, d_prime, preserve_idx = self.prune_kernel(self.prunable_index[self.cur_ind], action, preserve_idx)

        # if not self.visited[self.cur_ind]:
        #     for group in self.shared_idx:
        #         if self.cur_ind in group:  # set the shared ones
        #             for g_idx in group:
        #                 self.strategy_dict[self.prunable_idx[g_idx]][0] = action
        #                 self.strategy_dict[self.prunable_idx[g_idx - 1]][1] = action
        #                 self.visited[g_idx] = True
        #                 self.index_buffer[g_idx] = preserve_idx.copy()

        # if self.export_model:  # export checkpoint
        #     print('# Pruning {}: ratio: {}, d_prime: {}'.format(self.cur_ind, action, d_prime))


        self.strategy[self.curr_preserve_ratio].append(action)  # save action to strategy；这一层的保留率加进去
        self.d_prime_list[self.curr_preserve_ratio].append(d_prime)

        self.strategy_dict[self.prunable_index[self.cur_ind]][0] = action # conv
        self.strategy_dict[self.prunable_index[self.cur_ind]-2][1] = action # bn output
        self.strategy_dict[self.prunable_index[self.cur_ind]-3][1] = action # conv output

        # if self.cur_ind > 0: # 不是第一层
        #     self.strategy_dict[self.prunable_idx[self.cur_ind - 1]][1] = action # 上一层的输出通道保留率

        # all the actions are made
        if self._is_final_layer():
            # 这是最后一层了
            assert len(self.strategy[self.curr_preserve_ratio]) == len(self.prunable_index)
            current_flops = self._cur_flops()
            acc_t1 = time.time()
            acc,acc_ = self._validate(self.val_loader, self.model) # 验证
            acc_t2 = time.time()
            self.val_time = acc_t2 - acc_t1
            compress_ratio = current_flops * 1. / self.org_flops
            info_set = {'compress_ratio': compress_ratio, 'accuracy': acc, 'strategy': self.strategy[self.curr_prunRatio_index].copy(),"accuracy_":acc_}
            
            # 修改：接口变动
            reward = self.reward(self, acc, current_flops,self.preserve_ratio,compress_ratio)



            loc = self.curr_prunRatio_index
            if reward > self.best_reward[self.curr_preserve_ratio]:
                import os 
                
                self.best_reward[self.curr_preserve_ratio] = reward
                self.best_strategy[self.curr_preserve_ratio] = self.strategy[self.curr_preserve_ratio].copy()
                self.best_d_prime_list[self.curr_preserve_ratio]  = self.d_prime_list.copy()
                prGreen('===Target:{}==='.format(self.curr_preserve_ratio))
                prGreen('New best reward: {:.4f}, acc: {:.4f},acc_:{:.4f} compress: {:.4f},target ratio:{:.4f}'.format(reward, acc,acc_, compress_ratio,self.curr_preserve_ratio))
                prGreen('New best policy: {}'.format(self.best_strategy[self.curr_preserve_ratio]))
                prGreen('New best d primes: {}'.format(self.best_d_prime_list[self.curr_preserve_ratio] ))
                

                # write to txt log
                with open(self.output, 'a') as text_writer:
                    text_writer.write('\n============TargetRatio:{}============\n'.format(self.curr_preserve_ratio))
                    text_writer.write(
                    '#epoch: {}; acc: {:.4f},acc_:{:.4f};TargetRatio: {:.4f};DoneRatio: {:.4f};ArchivePercent:{:.4f};PrunStrategy:{} \n'.format(epoch,
                                                                                    info_set['accuracy'],info_set['accuracy_'],
                                                                                    self.curr_preserve_ratio,info_set['compress_ratio'],info_set['compress_ratio']/self.preserve_ratio,info_set['strategy']))
                
                    text_writer.write('New best reward: {:.4f}, acc: {:.4f},acc_:{:.4f} compress: {:.4f},target ratio:{:.4f}\n' \
                    .format(reward, acc,acc_, compress_ratio,self.curr_preserve_ratio))
                    text_writer.write('New best policy: {}\n'.format(self.best_strategy[self.curr_prunRatio_index]))
                    text_writer.write('New best d primes: {}\n'.format(self.best_d_prime_list))
                    text_writer.write('========================================\n')

                
                

            obs = self.layer_embedding[self.cur_ind, :].copy()  # actually the same as the last state
            done = True
            # if self.export_model:  # export state dict
            #     torch.save(self.model.state_dict(), self.export_path)
            #     return None, None, None, None
            return obs, reward, done, info_set

        info_set = None
        reward = 0
        done = False
        # self.visited[self.cur_ind] = True  # set to visited
        self.cur_ind += 1  # the index of next layer


        # build next state (in-place modify) 修改状态嵌入向量
        # 其他的已经被加过了
        self.layer_embedding[self.cur_ind][-3] = self._cur_reduced() * 1. / self.org_flops  # reduced
        self.layer_embedding[self.cur_ind][-2] = sum(self.flops_list[self.prunable_index[self.cur_ind] + 1:]) * 1. / self.org_flops  # rest
        self.layer_embedding[self.cur_ind][-1] = self.strategy[self.curr_preserve_ratio][-1]  # last action

        obs = self.layer_embedding[self.cur_ind, :].copy()
        return obs, reward, done, info_set

    def reset(self):
        """清空环境 以便进行新一轮的训练"""
        # restore env by loading the checkpoint
        self.model.load_state_dict(self.model_cached_ckp)
        self.cur_ind = 0 
        self.strategy = dict(zip(self.support_prun_ratio,[[]]*len(self.support_prun_ratio)))
        self.d_prime_list = dict(zip(self.support_prun_ratio,[[]]*len(self.support_prun_ratio)))
        # self.strategy_dict = copy.deepcopy(self.min_strategy_dict) 


        # reset all the  layer embeddings
        self.layer_embedding[:, -1] = 1.
        self.layer_embedding[:, -2] = 0.
        self.layer_embedding[:, -3] = 0.


        obs = self.layer_embedding[0].copy()
        obs[-2] = sum(self.wsize_list[self.prunable_index[0]+1:]) * 1. / sum(self.wsize_list)
        self.extract_time = 0
        self.fit_time = 0
        self.val_time = 0

        return obs

    def set_export_path(self, path):
        self.export_path = path

    def prune_kernel(self, op_idx, preserve_ratio, preserve_idx=None):
        '''对op_idx对应的额那层模型计算其最终的保留率、保留数、保留通道索引'''
        m_list = list(self.model.modules())
        op = m_list[op_idx] # 可以剪裁的模型，从列表中取出
        assert (preserve_ratio <= 1.)

        if preserve_ratio == 1:  # do not prune
            return 1., op.weight.size(1), None  # TODO: should be a full index
            # n, c, h, w = op.weight.size()
            # mask = np.ones([c], dtype=bool)

        def format_rank(x):
            rank = int(np.around(x)) # np.around四舍，六入，五凑偶；rank是转类型
            return max(rank, 1) # 至少保留一个通道

        n, c = op.weight.size(0), op.weight.size(1) 
        d_prime = format_rank(c * preserve_ratio)   # 输入通道保留数

        # wtf?
        # d_prime = int(np.ceil(d_prime * 1. / self.channel_round) * self.channel_round) 
        # if d_prime > c:
        #     d_prime = int(np.floor(c * 1. / self.channel_round) * self.channel_round)

        extract_t1 = time.time()

        
        X = self.layers_info[op_idx]['input_feat']  # input after pruning of previous ops
        Y = self.layers_info[op_idx]['output_feat']  # fixed output from original model
        # shape

        # conv [C_out, C_in, ksize, ksize]
        # fc [C_out, C_in]
        weight = op.weight.data.cpu().numpy()

        op_type = 'Conv2D'
        # if len(weight.shape) == 2:
        #     op_type = 'Linear'
        #     weight = weight[:, :, None, None]
        
        extract_t2 = time.time()
        self.extract_time += extract_t2 - extract_t1

        # 
        fit_t1 = time.time()
        if preserve_idx is None:  # not provided, generate new
            # 计算出
            importance = np.abs(weight).sum((0, 2, 3)) # 根据输入通道为组划分权重矩阵
            sorted_idx = np.argsort(-importance)  # sum magnitude along C_in, sort descend
            preserve_idx = sorted_idx[:d_prime]  # to preserve index
        assert len(preserve_idx) == d_prime
        mask = np.zeros(weight.shape[1], bool)
        mask[preserve_idx] = True

        # reconstruct, X, Y <= [N, C]
        masked_X = X[:, mask]
        if weight.shape[2] == 1:  # 1x1 conv or fc
            from lib.utils import least_square_sklearn
            rec_weight = least_square_sklearn(X=masked_X, Y=Y)
            rec_weight = rec_weight.reshape(-1, 1, 1, d_prime)  # (C_out, K_h, K_w, C_in')
            rec_weight = np.transpose(rec_weight, (0, 3, 1, 2))  # (C_out, C_in', K_h, K_w)
        else:
            import torch.optim as optim
            import torch.nn.functional as F
            op.prun_weight.data = op.origin_weight.data[:,mask,:,:]
            op.weight.data = op.prun_weight.data
            optimizer = optim.SGD(op.parameters(), lr=args.lr, momentum=args.momentum)
            for batch_idx in range(masked_X.shape[0]):
                optimizer.zero_grad()
                output = op(masked_X[batch_idx])
                loss = F.MSELoss(output,Y[batch_idx])
                loss.backward()
                optimizer.step()
            rec_weight = op.prun_weight.data
            
            # raise NotImplementedError('Todo')



        # if not self.export_model: 
        #     rec_weight_pad = np.zeros_like(weight)
        #     rec_weight_pad[:, mask, :, :] = rec_weight
        #     rec_weight = rec_weight_pad

        # if op_type == 'Linear':
        #     rec_weight = rec_weight.squeeze() # 压缩降维，去掉第一个为1的维度
        #     assert len(rec_weight.shape) == 2 # 线性层一个输入一个输出
        fit_t2 = time.time()
        self.fit_time += fit_t2 - fit_t1


        # now assign; 重构这个权重矩阵
        op.weight.data = torch.from_numpy(rec_weight).cuda()
        action = np.sum(mask) * 1. / len(mask)  # calculate the ratio


        # if self.export_model:  # prune previous buffer ops
        #     prev_idx = self.prunable_idx[self.prunable_idx.index(op_idx) - 1] # 取出上一个剪裁的索引

        #     for idx in range(prev_idx, op_idx):
        #         m = m_list[idx]

        #         if type(m) == nn.Conv2d:  # depthwise
        #             m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[mask, :, :, :]).cuda()
        #             if m.groups == m.in_channels:
        #                 m.groups = int(np.sum(mask)) # 修正 输入通道到输出通道的阻塞连接数

        #         elif type(m) == nn.BatchNorm2d:
        #             # 取出对应位置的元素
        #             m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[mask]).cuda() 
        #             m.bias.data = torch.from_numpy(m.bias.data.cpu().numpy()[mask]).cuda()
        #             m.running_mean.data = torch.from_numpy(m.running_mean.data.cpu().numpy()[mask]).cuda()
        #             m.running_var.data = torch.from_numpy(m.running_var.data.cpu().numpy()[mask]).cuda()
        
        return action, d_prime, preserve_idx

    def _is_final_layer(self):
        return self.cur_ind == len(self.prunable_index) - 1

    def _action_wall(self, action):
        """Predict the sparsity ratio actiont for layer Lt with constrained 
        model size (number of parameters) using fine-grained pruning"""
        
        assert len(self.strategy[self.curr_preserve_ratio]) == self.cur_ind # 

        action = float(action)
        action = np.clip(action, 0, 1) # 01截断

        other_comp = 0  # 其他层
        this_comp = 0 # 这一层相关的计算量（剪裁这一层输入通道能影响到的范围）


        # first we compute this_comp
        curr_prunable_index = self.prunable_index[self.cur_ind]

        current_layer = self.model_list[curr_prunable_index]
        if current_layer.kernel_size[0] == 3: # conv2
            this_comp  = self.layers_info[current_layer]['flops'] * self.lbound + self.layers_info[self.model_list[curr_prunable_index-2]]['flops'] +\
                 self.layers_info[self.model_list[curr_prunable_index-3]]['flops']
        else: # conv3
            this_comp = self.layers_info[current_layer]['flops']  + self.layers_info[self.model_list[curr_prunable_index-2]]['flops'] +\
                 self.layers_info[self.model_list[curr_prunable_index-3]]['flops'] * self.strategy_dict[curr_prunable_index-3][0]
        
        # then the other_comp
        other_comp += non_prunable_flops # the unrelated part
            
        # before the current layer
        before_prunable_index = self.prunable_index[:self.cur_ind]
        block_center_index = before_prunable_index[::2]
        if current_layer.kernel_size[0] == 3: # conv2
            for i,idx in enumerate(block_center_index):
                    other_comp += \
                        self.layers_info[self.model_list[idx]]['flops'] * self.strategy_dict[idx][0] * self.strategy_dict[idx][1] +\
                        self.layers_info[self.model_list[idx-2]]['flops']  * self.strategy_dict[idx-2][1] + \
                            self.layers_info[self.model_list[idx-3]]['flops'] * self.strategy_dict[idx-3][1] + \
                                self.layers_info[self.model_list[idx+1]]['flops'] * self.strategy_dict[idx+1][0] + \
                                self.layers_info[self.model_list[idx+3]]['flops'] * self.strategy_dict[idx+3][0]
        else:# conv3
            for i,idx in enumerate(block_center_index[:-1]):
                    other_comp += \
                        self.layers_info[self.model_list[idx]]['flops'] * self.strategy_dict[idx][0] * self.strategy_dict[idx][1] +\
                        self.layers_info[self.model_list[idx-2]]['flops']  * self.strategy_dict[idx-2][1] + \
                            self.layers_info[self.model_list[idx-3]]['flops'] * self.strategy_dict[idx-3][1] + \
                                self.layers_info[self.model_list[idx+1]]['flops'] * self.strategy_dict[idx+1][0] + \
                                self.layers_info[self.model_list[idx+3]]['flops'] * self.strategy_dict[idx+3][0]
            # the front part of the same bottleneck block
            _i = block_center_index[-1]
            other_comp += self.layers_info[self.model_list[_i-2]]['flops'] * self.strategy_dict[_i][1] + \
                self.layers_info[self.model_list[_i-3]]['flops'] * self.strategy_dict[_i-3][1]
        
        # after the current layer, we use the most aggressive policy as the paper state
        after_prunable_index = self.prunable_index[self.cur_ind+1:]
        if current_layer.kernel_size[0] == 3: # conv2
            # the front part of the same bottleneck block
            _i = after_prunable_index[0]
            other_comp += self.layers_info[self.model_list[_i]]['flops'] * self.lbound + \
                self.layers_info[self.model_list[_i-2]]['flops'] * self.lbound
            block_center_index = after_prunable_index[1::2]
            for i,idx in enumerate(block_center_index):
                    other_comp += \
                        self.layers_info[self.model_list[idx]]['flops'] * self.lbound * self.lbound +\
                        self.layers_info[self.model_list[idx-2]]['flops']  * self.lbound + \
                            self.layers_info[self.model_list[idx-3]]['flops'] * self.lbound + \
                                self.layers_info[self.model_list[idx+1]]['flops'] * self.lbound + \
                                self.layers_info[self.model_list[idx+3]]['flops'] * self.lbound
        else:# conv3
            block_center_index = after_prunable_index[::2]
            for i,idx in enumerate(block_center_index):
                    other_comp += \
                        self.layers_info[self.model_list[idx]]['flops'] * self.lbound * self.lbound +\
                        self.layers_info[self.model_list[idx-2]]['flops']  * self.lbound + \
                            self.layers_info[self.model_list[idx-3]]['flops'] * self.lbound + \
                                self.layers_info[self.model_list[idx+1]]['flops'] * self.lbound + \
                                self.layers_info[self.model_list[idx+3]]['flops'] * self.lbound
        
        # min except prun ratio that this layer should done , aka max preserve ratio
        max_preserve_ratio = (self.expected_preserve_computation - other_comp) * 1. / this_comp 

        # if the prun ratio is less than the min except prun ratio, that is the preserve ratio is more than the max_preserve_ratio
        # we should truncate the preserve ratio to max_preserve_ratio
        action = np.minimum(action, max_preserve_ratio) 

        # Meanwhile, preserve_ratio should be greater than lbound
        action = np.maximum(action, self.lbound) 
        
        # after the current layer
        
        # curr_prunable_index = self.prunable_index[self.cur_ind]
        # for i , ops in enmuerate(self.model_list):
        #     if i >= curr_prunable_index-3 and i < curr_prunable_index:
        #         this_comp += self.layers_info[ops].flops * self.strategy_dict[i][0]  * self.strategy_dict[i][1] 
        #     elif i==curr_prunable_index:
        #         this_comp += self.layers_info[ops].flops * self.lbound * self.strategy_dict[i][0] 
        #     else:
        #         def prunable_block_after(i,curr_prunable_index):
        #             for prun_idx in curr_prunable_index:
        #                 if prun_idx > curr_prunable_index:
        #                     if i>=prun_idx-3 and i<=prun_idx:
        #                         return True
                
        #         if prunable_block_after(i,curr_prunable_index):
                    
        #         else:
        #             other_comp += self.layers_info[ops].flops * self.strategy_dict[i][0] * self.strategy_dict[i][1]


        # for  idx in (self.prunable_index):

        #     flop = self.layers_info[self.model_list[idx]]['flops']

        #     if i == self.cur_ind - 1:  # TODO: add other member in the set
        #         this_comp += flop * self.strategy_dict[idx][0]
        #         # add buffer (but not influenced by ratio)
        #         other_comp += buffer_flop * self.strategy_dict[idx][0]
        #     elif i == self.cur_ind:
        #         this_comp += flop * self.strategy_dict[idx][1]
        #         # also add buffer here (influenced by ratio)
        #         this_comp += buffer_flop
        #     else:
        #         other_comp += flop * self.strategy_dict[idx][0] * self.strategy_dict[idx][1]
        #         # add buffer
        #         other_comp += buffer_flop * self.strategy_dict[idx][0]  # only consider input reduction

        # self.expected_min_preserve = other_comp + this_comp * action 
        # max_preserve_ratio = (self.expected_preserve_computation - other_comp) * 1. / this_comp 

        # action = np.minimum(action, max_preserve_ratio) 
    
        # action = np.maximum(action, self.strategy_dict[self.prunable_idx[self.cur_ind]][0])  # impossible (should be) 

        return action

    # def _get_buffer_flops(self, idx):
    #     """获取该非缓冲层对应前面的缓冲层的总浮点运算数"""
    #     buffer_idx = self.buffer_dict[idx]
    #     buffer_flop = sum([self.layer_info_dict[_]['flops'] for _ in buffer_idx])
    #     return buffer_flop

    def _cur_flops(self):
        """计算网络中目前的flops"""
        flops = 0
        for idx in self.model_list:
            c, n = self.strategy_dict[idx]  # input, output pruning ratio
            flops += self.layers_info[self.model_list[idx]]['flops'] * c * n
        return flops

    def _cur_reduced(self):
        # return the reduced weight
        reduced = self.org_flops - self._cur_flops()
        return reduced

    def _init_data(self):
        # split the train set into train + val
        # for CIFAR, split 5k for val
        # for ImageNet, split 3k for val
        val_size = 5000 if 'cifar' in self.data_type else 3000
        import os
        
        self.train_loader, self.val_loader, n_class = get_split_dataset(self.data_type, self.batch_size,
                                                                        self.n_data_worker, val_size,
                                                                        data_root=self.data_root,
                                                                        use_real_val=self.use_real_val,
                                                                        shuffle=False)  # same sampling
        

        # if self.use_real_val:  # use the real val set for eval, which is actually wrong
        #     print('*** USE REAL VALIDATION SET!')

      

    def _regenerate_input_feature(self):
        """
        
        """
        # only re-generate the input feature

        m_list = list(self.model.modules())

        # delete old features
        for k, v in self.layer_info_dict.items():
            if 'input_feat' in v:
                v.pop('input_feat')

        # now let the image flow
        print('=> Regenerate features...')

        with torch.no_grad():
            for i_b, (input, target) in enumerate(self.data_saver):
                input_var = torch.autograd.Variable(input).cuda()

                # inference and collect stats
                _ = self.model(input_var)

                for idx in self.prunable_idx:
                    f_in_np = m_list[idx].input_feat.data.cpu().numpy()
                    if len(f_in_np.shape) == 4:  # conv
                        if self.prunable_idx.index(idx) == 0:  # first conv
                            f_in2save = None
                        else:
                            randx = self.layer_info_dict[idx][(i_b, 'randx')]
                            randy = self.layer_info_dict[idx][(i_b, 'randy')]
                            f_in2save = f_in_np[:, :, randx, randy].copy().transpose(0, 2, 1)\
                                .reshape(self.batch_size * self.n_points_per_layer, -1)
                    else:  # fc
                        assert len(f_in_np.shape) == 2
                        f_in2save = f_in_np.copy()
                    if 'input_feat' not in self.layer_info_dict[idx]:
                        self.layer_info_dict[idx]['input_feat'] = f_in2save
                    else:
                        self.layer_info_dict[idx]['input_feat'] = np.vstack(
                            (self.layer_info_dict[idx]['input_feat'], f_in2save))

    def _build_state_embedding(self):
        # build the static part of the state embedding
        layer_embedding = []
        module_list = self.model_list
        
        for ind in self.prunable_index:
            m = module_list[ind]
            this_state = []

            if type(m) == nn.Conv2d:
                this_state.append(ind)  # index
                # this_state.append(m_type)  # layer type, 0 for conv2, 1 for conv3
                this_state.append(m.in_channels)  # in channels
                this_state.append(m.out_channels)  # out channels
                this_state.append(m.stride[0])  # stride
                this_state.append(m.kernel_size[0])  # kernel size
                this_state.append(np.prod(m.weight.size()))  # weight size
            
            # elif type(m) == nn.Linear:
            #     this_state.append(i)  # index
            #     this_state.append(1)  # layer type, 1 for fc
            #     this_state.append(m.in_features)  # in channels
            #     this_state.append(m.out_features)  # out channels
            #     this_state.append(0)  # stride
            #     this_state.append(1)  # kernel size
            #     this_state.append(np.prod(m.weight.size()))  # weight size

            discrete_status = [0]*len(self.support_prun_ratio)
            discrete_status[self.curr_prunRatio_index] = 1
            this_state += discrete_status

            # this 3 features need to be changed later
            this_state.append(0.)  # reduced
            this_state.append(0.)  # rest
            this_state.append(1.)  # a_{t-1}
            layer_embedding.append(np.array(this_state))

        # normalize the state
        layer_embedding = np.array(layer_embedding, 'float')
        print('=> shape of embedding (n_layer * n_dim): {}'.format(layer_embedding.shape))
        assert len(layer_embedding.shape) == 2, layer_embedding.shape

        # 归一化
        for i in range(layer_embedding.shape[1]):
            fmin = min(layer_embedding[:, i])
            fmax = max(layer_embedding[:, i])
            if fmax - fmin > 0:
                layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)

        # 状态嵌入向量
        self.layer_embedding = layer_embedding

    def change(self):
        # 采样本轮剪裁率
        self.curr_prunRatio_index = (self.curr_prunRatio_index+1)%len(self.support_prun_ratio)
        self.curr_preserve_ratio = self.support_prun_ratio[self.curr_prunRatio_index]
        self.layer_embedding[:, -6:-len(self.support_prun_ratio)] = 0 # 重置
        self.layer_embedding[:,-6+self.curr_prunRatio_index] = 1 # 更新one-hot向量
        self.expected_preserve_computation = self.preserve_ratio * sum(self.flops_list)

    def _validate(self, val_loader, model, verbose=False):
        '''
        Validate the performance on validation set
        :param val_loader:
        :param model:
        :param verbose:
        :return:
        '''
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        criterion = nn.CrossEntropyLoss().cuda()
        # switch to evaluate mode
        model.eval()
        end = time.time()

        t1 = time.time()
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda(non_blocking=True)
                input_var = torch.autograd.Variable(input).cuda()
                target_var = torch.autograd.Variable(target).cuda()

                # compute output
                output = model(input_var)
                loss = criterion(output, target_var)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
        t2 = time.time()
        if verbose:
            print('* Test loss: %.3f    top1: %.3f    top5: %.3f    time: %.3f' %
                  (losses.avg, top1.avg, top5.avg, t2 - t1))
        if self.acc_metric == 'acc1':
            return (top1.avg,top5.avg)
        elif self.acc_metric == 'acc5':
            return (top5.avg,top1.avg)
        else:
            raise NotImplementedError



    # def _build_index(self):
    #     """一些索引的构建"""
    #     self.prunable_idx = [] # 可剪裁的索引
    #     self.prunable_ops = [] # 可剪裁的网络
    #     self.layer_type_dict = {} # 类型
    #     self.strategy_dict = {} # 策略dict
    #     self.org_channels = [] # 每层输入特征通道数或者特征维度

    #     if self.args.model =="resnet50":
    #         self.prunable_blocks_idx = []
    #         self.prunable_ops = [] # 可剪裁的网络
    #         self.prunable_idx = [] # 可剪裁的索引
    #         for i, module in enumerate(self.model.modules()):
    #             from models.resnet import Bottleneck
    #             if isinstance(module, Bottleneck):
    #                 self.prunable_blocks_idx.append(i)
    #                 self.prunable_ops.append(module.conv2)
    #                 self.prunable_ops.append(module.conv3)
    #                 self.prunable_idx.append((i,2))
    #                 self.prunable_idx.append((i,3))
    #                 self.strategy_dict[(i,2)] = [self.lbound, self.lbound]  
    #                 self.strategy_dict[(i,3)] = [self.lbound, self.lbound] 
    #                 self.org_channels.append(module.conv2.in_channels)
    #                 self.org_channels.append(module.conv3.in_channels)
                
    #         self.min_strategy_dict = copy.deepcopy(self.strategy_dict)
    #         print('=> Prunable layer idx: {}'.format(self.prunable_idx))
    #         print('=> Initial min strategy dict: {}'.format(self.min_strategy_dict))

    #         # added for supporting residual connections during pruning;
    #         self.visited = [False] * len(self.prunable_idx) # 给每个可以减的层都加以访问标记
    #         self.index_buffer = {}

    #     elif self.args.model =="mobilenet":
        
    #         self.buffer_dict = {} # 装dwconv的dict
    #         this_buffer_list = [] # 装dwconv
    #         # build index and the min strategy dict
    #         for i, m in enumerate(self.model.modules()):
    #             if type(m) in self.prunable_layer_types:
    #                 if type(m) == nn.Conv2d and m.groups == m.in_channels:  # depth-wise conv, buffer；
    #                     # 对应的是全并行的Depthwise Convolution卷积网络，此时存放到缓冲列表中
    #                     this_buffer_list.append(i)
    #                 else:  # really prunable
    #                     self.prunable_idx.append(i)
    #                     self.prunable_ops.append(m)
    #                     self.layer_type_dict[i] = type(m)
    #                     self.buffer_dict[i] = this_buffer_list # 将之前存的buffer放进bufferdict相应位置，索引为buffer结束后面第一个模型的索引位置
    #                     this_buffer_list = []  # empty
    #                     self.org_channels.append(m.in_channels if type(m) == nn.Conv2d else m.in_features) # 原来的通道
    #                     self.strategy_dict[i] = [self.lbound, self.lbound]  # 初始化

    #         # 输入和输出的保留率为1
    #         self.strategy_dict[self.prunable_idx[0]][0] = 1  # modify the input
    #         self.strategy_dict[self.prunable_idx[-1]][1] = 1  # modify the output

    #         # self.shared_idx = []
    #         # if self.args.model == 'mobilenetv2':  # TODO: to be tested! Share index for residual connection
    #         #     connected_idx = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]  # to be partitioned
    #         #     last_ch = -1
    #         #     share_group = None
    #         #     for c_idx in connected_idx:
    #         #         if self.prunable_ops[c_idx].in_channels != last_ch:  # new group
    #         #             last_ch = self.prunable_ops[c_idx].in_channels
    #         #             if share_group is not None:
    #         #                 self.shared_idx.append(share_group)
    #         #             share_group = [c_idx]
    #         #         else:  # same group
    #         #             share_group.append(c_idx)
    #         #     print('=> Conv layers to share channels: {}'.format(self.shared_idx))

    #         self.min_strategy_dict = copy.deepcopy(self.strategy_dict)

    #         # 存放那些Deepwise层的索引列表
    #         self.buffer_idx = []
    #         for k, v in self.buffer_dict.items():
    #             self.buffer_idx += v

    #         print('=> Prunable layer idx: {}'.format(self.prunable_idx))
    #         print('=> Buffer layer idx: {}'.format(self.buffer_idx))
    #         print('=> Initial min strategy dict: {}'.format(self.min_strategy_dict))

    #         # added for supporting residual connections during pruning;
    #         self.visited = [False] * len(self.prunable_idx) # 给每个可以减的层都加以访问标记
    #         self.index_buffer = {}
        

    # def _extract_layer_information(self):
        
    #     if self.args.model == "resnet50":
    #         m_list = list(self.model.children())

    #         self.data_saver = []
    #         self.layer_info_dict = dict()
    #         self.wsize_list = []
    #         self.flops_list = []

    #         from lib.utils import measure_layer_for_pruning

    #         # extend the forward fn to record layer info
    #         def new_forward(m):
    #             def lambda_forward(x):
    #                 m.input_feat = x.clone()
    #                 measure_layer_for_pruning(m, x)
    #                 y = m.old_forward(x)
    #                 m.output_feat = y.clone()
    #                 return y
    #             return lambda_forward

            
    #         for m in m_list:  # get all
    #             m.old_forward = m.forward
    #             m.forward = new_forward(m)

    #         # now let the image flow
    #         print('=> Extracting information...')
    #         with torch.no_grad():
    #             for i_b, (input, target) in enumerate(self.train_loader):  # use image from train set
    #                 if i_b == self.n_calibration_batches:
    #                     break
    #                 self.data_saver.append((input.clone(), target.clone()))
    #                 input_var = torch.autograd.Variable(input).cuda()

    #                 # inference and collect stats
    #                 _ = self.model(input_var)

    #                 if i_b == 0:  # first batch
    #                     for idx in range(len(m_list)):
    #                         self.layer_info_dict[idx] = dict()
    #                         self.layer_info_dict[idx]['params'] = m_list[idx].params
    #                         self.layer_info_dict[idx]['flops'] = m_list[idx].flops
    #                         self.wsize_list.append(m_list[idx].params)
    #                         self.flops_list.append(m_list[idx].flops)

    #                 for idx in self.prunable_idx:
    #                     f_in_np = m_list[idx].input_feat.data.cpu().numpy()
    #                     f_out_np = m_list[idx].output_feat.data.cpu().numpy()
    #                     if len(f_in_np.shape) == 4:  # conv
    #                         if self.prunable_idx.index(idx) == 0:  # first conv
    #                             f_in2save, f_out2save = None, None
    #                         elif m_list[idx].weight.size(3) > 1:  # normal conv
    #                             f_in2save, f_out2save = f_in_np, f_out_np
    #                         else:  # 1x1 conv
    #                             # assert f_out_np.shape[2] == f_in_np.shape[2]  # now support k=3
    #                             randx = np.random.randint(0, f_out_np.shape[2] - 0, self.n_points_per_layer)
    #                             randy = np.random.randint(0, f_out_np.shape[3] - 0, self.n_points_per_layer)
    #                             # input: [N, C, H, W]
    #                             self.layer_info_dict[idx][(i_b, 'randx')] = randx.copy()
    #                             self.layer_info_dict[idx][(i_b, 'randy')] = randy.copy()

    #                             f_in2save = f_in_np[:, :, randx, randy].copy().transpose(0, 2, 1)\
    #                                 .reshape(self.batch_size * self.n_points_per_layer, -1)

    #                             f_out2save = f_out_np[:, :, randx, randy].copy().transpose(0, 2, 1) \
    #                                 .reshape(self.batch_size * self.n_points_per_layer, -1)
    #                     else:
    #                         assert len(f_in_np.shape) == 2
    #                         f_in2save = f_in_np.copy()
    #                         f_out2save = f_out_np.copy()
    #                     if 'input_feat' not in self.layer_info_dict[idx]:
    #                         self.layer_info_dict[idx]['input_feat'] = f_in2save
    #                         self.layer_info_dict[idx]['output_feat'] = f_out2save
    #                     else:
    #                         self.layer_info_dict[idx]['input_feat'] = np.vstack(
    #                             (self.layer_info_dict[idx]['input_feat'], f_in2save))
    #                         self.layer_info_dict[idx]['output_feat'] = np.vstack(
    #                             (self.layer_info_dict[idx]['output_feat'], f_out2save))
    #     elif  self.args.model == "mobilenet":
    #         m_list = list(self.model.modules())

    #         self.data_saver = []
    #         self.layer_info_dict = dict()
    #         self.wsize_list = []
    #         self.flops_list = []

    #         from lib.utils import measure_layer_for_pruning

    #         # extend the forward fn to record layer info
    #         def new_forward(m):
    #             def lambda_forward(x):
    #                 m.input_feat = x.clone()
    #                 measure_layer_for_pruning(m, x)
    #                 y = m.old_forward(x)
    #                 m.output_feat = y.clone()
    #                 return y
    #             return lambda_forward

            
    #         for m in m_list:  # get all
    #             m.old_forward = m.forward
    #             m.forward = new_forward(m)

    #         # now let the image flow
    #         print('=> Extracting information...')
    #         with torch.no_grad():
    #             for i_b, (input, target) in enumerate(self.train_loader):  # use image from train set
    #                 if i_b == self.n_calibration_batches:
    #                     break
    #                 self.data_saver.append((input.clone(), target.clone()))
    #                 input_var = torch.autograd.Variable(input).cuda()

    #                 # inference and collect stats
    #                 _ = self.model(input_var)

    #                 if i_b == 0:  # first batch
    #                     for idx in range(len(m_list)):
    #                         self.layer_info_dict[idx] = dict()
    #                         self.layer_info_dict[idx]['params'] = m_list[idx].params
    #                         self.layer_info_dict[idx]['flops'] = m_list[idx].flops
    #                         self.wsize_list.append(m_list[idx].params)
    #                         self.flops_list.append(m_list[idx].flops)

    #                 for idx in self.prunable_idx:
    #                     f_in_np = m_list[idx].input_feat.data.cpu().numpy()
    #                     f_out_np = m_list[idx].output_feat.data.cpu().numpy()
    #                     if len(f_in_np.shape) == 4:  # conv
    #                         if self.prunable_idx.index(idx) == 0:  # first conv
    #                             f_in2save, f_out2save = None, None
    #                         elif m_list[idx].weight.size(3) > 1:  # normal conv
    #                             f_in2save, f_out2save = f_in_np, f_out_np
    #                         else:  # 1x1 conv
    #                             # assert f_out_np.shape[2] == f_in_np.shape[2]  # now support k=3
    #                             randx = np.random.randint(0, f_out_np.shape[2] - 0, self.n_points_per_layer)
    #                             randy = np.random.randint(0, f_out_np.shape[3] - 0, self.n_points_per_layer)
    #                             # input: [N, C, H, W]
    #                             self.layer_info_dict[idx][(i_b, 'randx')] = randx.copy()
    #                             self.layer_info_dict[idx][(i_b, 'randy')] = randy.copy()

    #                             f_in2save = f_in_np[:, :, randx, randy].copy().transpose(0, 2, 1)\
    #                                 .reshape(self.batch_size * self.n_points_per_layer, -1)

    #                             f_out2save = f_out_np[:, :, randx, randy].copy().transpose(0, 2, 1) \
    #                                 .reshape(self.batch_size * self.n_points_per_layer, -1)
    #                     else:
    #                         assert len(f_in_np.shape) == 2
    #                         f_in2save = f_in_np.copy()
    #                         f_out2save = f_out_np.copy()
    #                     if 'input_feat' not in self.layer_info_dict[idx]:
    #                         self.layer_info_dict[idx]['input_feat'] = f_in2save
    #                         self.layer_info_dict[idx]['output_feat'] = f_out2save
    #                     else:
    #                         self.layer_info_dict[idx]['input_feat'] = np.vstack(
    #                             (self.layer_info_dict[idx]['input_feat'], f_in2save))
    #                         self.layer_info_dict[idx]['output_feat'] = np.vstack(
    #                             (self.layer_info_dict[idx]['output_feat'], f_out2save))
                      