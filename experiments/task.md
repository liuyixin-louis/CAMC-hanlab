# Experiments
> Search: acc1,acc5(val and test);

> FT: acc1,acc5(val and test);


## TODOLIST

- [x] add more experiment needed info to the tfboard and txtlogger

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
python /home/young/liuyixin/8.29/CAMC/train_ddpg.py --job train --model mobilenet --dataset imagenet --lbound 0.2 --rbound 1 --reward acc_normalize --data_root /home/dataset/imagenet --ckpt_path /home/young/liuyixin/8.29/CAMC/checkpoints/mobilenetv1_imagenet.pth.tar --seed 2020 --data_bsize 64 --n_gpu 1 --warmup 300 --train_episode 800 --suffix mobilenetV1_imagenet_trainDDPG
```

- result:   http://222.201.187.249:6006/


#### 1.2 On Cifar10 DataSet

- script
```shell
python /home/young/liuyixin/8.29/CAMC/train_ddpg.py --job train --model mobilenet --dataset cifar10 --lbound 0.2 --rbound 1 --reward acc_reward --data_root /home/dataset/cifar --ckpt_path /home/young/liuyixin/8.22/amc/checkpoints/mobilenetV1cifar10_origin_acc80.35_earlyStop.tar --seed 2020 --data_bsize 32 --n_gpu 1 --warmup 50 --train_episode 400 --suffix mobilenetV1_cifar10_trainDDPG
```


## 


