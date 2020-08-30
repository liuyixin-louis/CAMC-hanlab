# Experiments
> Search: acc1,acc5(val and test);

> FT: acc1,acc5(val and test);


## TODOLIST

- [x] add more experiment needed info to the tfboard and txtlogger

- [ ] reuse the DDPG model of mobilenetv1 of imagenet. 

- [ ] getModelNet: input dataset and model_arch,output pytorch model

- [ ] Traing Script for some Network that checkpoints are not available

- [ ] Collect Origin Model Checkpoints;

- [ ] Expand the Amc program to support purning for more model_arch on different dataset

- [ ] The result of mobilenetv1 on imagenet  (AMC vs CAMC)

- [ ] The result of mobilenetv1 on cifar10  (AMC vs CAMC)

- [ ] The result of mobilenetv2 on imagenet  (AMC vs CAMC)

- [ ] The result of mobilenetv2 on cifar10  (AMC vs CAMC)

- [ ] The result of preresnet56 on imagenet  (AMC vs CAMC)

- [ ] The result of preresnet56 on cifar10  (AMC vs CAMC)

- [ ] Other available model_arch performance result...





## Our method-CAMC under constrains [0.3,0.5,0.7]

### 1. mobilenetV1 

#### 1.1 On Imagenet DataSet
- script
  
```shell
python /home/young/liuyixin/8.29/CAMC/train_ddpg.py --job train --model mobilenet --dataset imagenet --lbound 0.2 --rbound 1 --reward acc_normalize --data_root /home/dataset/imagenet --ckpt_path /home/young/liuyixin/8.29/CAMC/checkpoints/mobilenetv1_imagenet.pth.tar --seed 2020 --data_bsize 64 --n_gpu 1 --warmup 100 --train_episode 400 --suffix mobilenetV1_imagenet_trainDDPG
```

- result:   http://222.201.187.249:6006/

![image-20200830141444513](https://i.loli.net/2020/08/30/tTv6SLXcmaWIgRl.png)

![image-20200830142657276](https://i.loli.net/2020/08/30/92n6sMquvePgz4J.png)

- otherTry with argment modified

```shell
python /home/young/liuyixin/8.29/CAMC/train_ddpg.py --job train --model mobilenet --dataset imagenet --lbound 0.2 --rbound 1 --reward acc_normalize --data_root /home/dataset/imagenet --ckpt_path /home/young/liuyixin/8.29/CAMC/checkpoints/mobilenetv1_imagenet.pth.tar --seed 2020 --data_bsize 64 --n_gpu 1 --warmup 100 --train_episode 400 --suffix mobilenetV1_imagenet_trainDDPG_100warmup300train
```



#### 1.2 On Cifar10 DataSet

- script
```shell
python /home/young/liuyixin/8.29/CAMC/train_ddpg.py --job train --model mobilenet --dataset cifar10 --lbound 0.2 --rbound 1 --reward acc_reward --data_root /home/dataset/cifar --ckpt_path /home/young/liuyixin/8.22/amc/checkpoints/mobilenetV1cifar10_origin_acc80.35_earlyStop.tar --seed 2020 --data_bsize 32 --n_gpu 1 --warmup 50 --train_episode 400 --suffix mobilenetV1_cifar10_trainDDPG
```





## Thinking Why

1. Will our training convergent with other **acc_reward** function? What is the **advantages** of our designed rewardfunction?

> Example: Compare to acc_reward=acc * 0.01 . (Using **MobilenetV1** on **Imagenet** DataSet, same hyper-parameter)

```shell
python /home/young/liuyixin/8.29/CAMC/train_ddpg.py --job train --model mobilenet --dataset imagenet --lbound 0.2 --rbound 1 --reward acc_reward --data_root /home/dataset/imagenet --ckpt_path /home/young/liuyixin/8.29/CAMC/checkpoints/mobilenetv1_imagenet.pth.tar --seed 2020 --data_bsize 64 --n_gpu 1 --warmup 300 --train_episode 800 --suffix mobilenetV1_imagenet_trainDDPG_acc_reward --acc_metric acc5
```

- result:

![image-20200830142257181](https://i.loli.net/2020/08/30/AumyHPaVhBxl53C.png)

![image-20200830142725494](https://i.loli.net/2020/08/30/laEFNJRX3Bbs9Ie.png)

- Phenomenon(Compared to our result)
  - the curve of acc5 and reward with 0.3 prun_ratio is obviously steeper than ours
  - Our best-reward curve reaches the highest peak earlier
- Analyse
  - 
- Conlusion
  - Our degsined reward function help with Accelerated training process of DDPG while pruning



2. Can we directly reuse the trained ddpg agent?

- script
```shell
python /home/young/liuyixin/8.29/CAMC/resue_ddpg.py --job reuseDDPG_test --model mobilenet --dataset imagenet --lbound 0.2 --rbound 1 --reward acc_normalize --data_root /home/dataset/imagenet --ckpt_path /home/young/liuyixin/8.29/CAMC/checkpoints/mobilenetv1_imagenet.pth.tar --resume_ddpg_checkpoint /home/young/liuyixin/8.29/CAMC/logs/mobilenet_imagenet_search_mobilenetV1_imagenet_trainDDPG-run22 --seed 2020 --data_bsize 64 --n_gpu 1 --test_episode 10 --suffix mobilenetV1_imagenet_testRuseDDPG
```

- Phenomenon：

```
============TargetRatio:0.3============
#epoch: 0; acc: 7.1667,acc_:2.1667;TargetRatio: 0.3000;DoneRatio: 0.3100;ArchivePercent:1.0333;PrunStrategy:[1.0, 0.75, 0.625, 0.5625, 0.5625, 0.53125, 0.53125, 0.53125, 0.53125, 0.53125, 0.515625, 0.515625, 0.5, 0.5078125, 0.5390625] 
New best reward: 0.7963, acc: 7.1667,acc_:2.1667 compress: 0.3100,target ratio:0.3000
New best policy: [1.0, 0.75, 0.625, 0.5625, 0.5625, 0.53125, 0.53125, 0.53125, 0.53125, 0.53125, 0.515625, 0.515625, 0.5, 0.5078125, 0.5390625]
New best d primes: [3, 24, 40, 72, 72, 136, 136, 272, 272, 272, 264, 264, 256, 520, 552]
========================================

============TargetRatio:0.5============
#epoch: 1; acc: 58.5333,acc_:30.8333;TargetRatio: 0.5000;DoneRatio: 0.4948;ArchivePercent:0.9896;PrunStrategy:[1.0, 0.75, 0.75, 0.75, 0.75, 0.71875, 0.6875, 0.6875, 0.6875, 0.671875, 0.65625, 0.65625, 0.640625, 0.671875, 0.734375] 
New best reward: 0.8018, acc: 58.5333,acc_:30.8333 compress: 0.4948,target ratio:0.5000
New best policy: [1.0, 0.75, 0.75, 0.75, 0.75, 0.71875, 0.6875, 0.6875, 0.6875, 0.671875, 0.65625, 0.65625, 0.640625, 0.671875, 0.734375]
New best d primes: [3, 24, 48, 96, 96, 184, 176, 352, 352, 344, 336, 336, 328, 688, 752]
========================================

============TargetRatio:0.7============
#epoch: 2; acc: 81.5667,acc_:55.2000;TargetRatio: 0.7000;DoneRatio: 0.7004;ArchivePercent:1.0005;PrunStrategy:[1.0, 1.0, 0.875, 0.875, 0.875, 0.84375, 0.84375, 0.84375, 0.828125, 0.828125, 0.8125, 0.8125, 0.796875, 0.78125, 0.6796875] 
New best reward: 0.8963, acc: 81.5667,acc_:55.2000 compress: 0.7004,target ratio:0.7000
New best policy: [1.0, 1.0, 0.875, 0.875, 0.875, 0.84375, 0.84375, 0.84375, 0.828125, 0.828125, 0.8125, 0.8125, 0.796875, 0.78125, 0.6796875]
New best d primes: [3, 32, 56, 112, 112, 216, 216, 432, 424, 424, 416, 416, 408, 800, 696]
========================================
```

- Analyse
  - 
- Conlusion
  - Our degsined reward function help with Accelerated training process of DDPG while pruning