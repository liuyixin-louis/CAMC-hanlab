# Experiments



## Order by Time



1. **The result of mobilenetv1 on imagenet  (AMC vs CAMC)** 

Note：**Val set**、**Search**

> ```bash
> python amc_search.py \
>  --job=train \
>  --model=mobilenet \
>  --dataset=imagenet \
>  --preserve_ratio=*** \
>  --lbound=0.2 \
>  --rbound=1 \
>  --reward=acc_reward \
>  --data_root=/home/dataset/imagenet \
>  --ckpt_path=./checkpoints/mobilenet_imagenet.pth.tar \
>  --seed=2020 \
>  --warmup=100 \
>  --train_episode=400 \
>  --n_gpu=1
> 
> python /home/young/liuyixin/8.29/CAMC/train_ddpg.py --job train --model mobilenet --dataset imagenet --lbound 0.2 --rbound 1 --reward acc_normalize --data_root /home/dataset/imagenet --ckpt_path /home/young/liuyixin/8.29/CAMC/checkpoints/mobilenetv1_imagenet.pth.tar --seed 2020 --data_bsize 64 --n_gpu 1 --warmup 100 --train_episode 400 --suffix mobilenetV1_imagenet_trainDDPG_100warmup300train
> ```

- 100warmup400train:

  - acc

  |          | AMC                | CAMC      |
  | -------- | ------------------ | --------- |
  | 0.3-ACC5 | 9.333              | 10.2      |
  | 0.3-ACC1 | 2.867              | 2.27      |
  | 0.5-ACC5 | 72.2667            | 73.1      |
  | 0.5-ACC1 | 45.666666666666664 | 46.4      |
  | 0.7-ACC5 | 90.1667            | **75.67** |
  | 0.7-ACC1 | 59.56666666666667  | 50.42     |

  - stategy

|      | AMC                                                          | CAMC                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 0.3  | [1.0, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5, 0.53125, 0.53125, 0.546875, 0.5625, 0.5625, 0.609375, 0.7109375, 0.203125] | [1.0, 0.75, 0.625, 0.5625, 0.5625, 0.5625, 0.53125, 0.53125, 0.53125, 0.53125, 0.53125, 0.53125, 0.515625, 0.5, 0.203125] |
| 0.5  | [1.0, 0.75, 0.75, 0.6875, 0.6875, 0.6875, 0.6875, 0.734375, 0.703125, 0.671875, 0.6875, 0.671875, 0.765625, 0.7890625, 0.5390625] | [1.0, 0.75, 0.75, 0.75, 0.75, 0.71875, 0.71875, 0.703125, 0.703125, 0.6875, 0.6875, 0.671875, 0.6875, 0.6953125, 0.5] |
| 0.7  | [1.0, 0.75, 1.0, 0.875, 0.875, 0.84375, 0.65625, 0.90625, 0.875, 0.90625, 0.625, 0.921875, 0.734375, 0.859375, 0.734375] | [1.0, 0.75, 0.75, 0.75, 0.75, 0.71875, 0.71875, 0.71875, 0.703125, 0.703125, 0.703125, 0.6875, 0.6875, 0.6875, 0.6875] |

**Note: It is strange that CAMC  get worse result with 0.7 prun_ratio(75.67 vs AMC 90) .MayBe The training epcho need to increase? More Experiments are needed.**





- 100warmUp800Train

```bash
python /home/young/liuyixin/8.29/CAMC/train_ddpg.py --job train --model mobilenet --dataset imagenet --lbound 0.2 --rbound 1 --reward acc_normalize --data_root /home/dataset/imagenet --ckpt_path /home/young/liuyixin/8.29/CAMC/checkpoints/mobilenetv1_imagenet.pth.tar --seed 2020 --data_bsize 64 --n_gpu 1 --warmup 100 --train_episode 800 --suffix mobilenetV1_imagenet_trainDDPG_100warmup800train --rmsize 100
```

| prun_ratio | acc5(val) | prune_stategy                                                |                                                              |
| ---------- | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 0.3        | 10.0333   | [1.0, 0.75, 0.625, 0.5625, 0.5625, 0.53125, 0.53125, 0.53125, 0.53125, 0.515625, 0.5, 0.5, 0.5, 0.515625, 0.4140625] | [3, 24, 40, 72, 72, 136, 136, 272, 272, 264, 256, 256, 256, 528, 424] |
| 0.5        | 72.9667   | [1.0, 0.75, 0.75, 0.75, 0.75, 0.71875, 0.71875, 0.703125, 0.703125, 0.703125, 0.6875, 0.6875, 0.6875, 0.671875, 0.4609375] | [3, 24, 48, 96, 96, 184, 184, 360, 360, 360, 352, 352, 352, 688, 472] |
| 0.7        | 90.2667   | [1.0, 1.0, 0.875, 0.875, 0.875, 0.84375, 0.84375, 0.828125, 0.8125, 0.8125, 0.8125, 0.796875, 0.8125, 0.828125, 0.7109375] | [3, 32, 56, 112, 112, 216, 216, 424, 416, 416, 416, 408, 416, 848, 728] |

**Note : the result 0f 0.7 now is close to 90. All the acc5 is slightly outperform orginal AMC. May the problem mentioned above is related to the number of  traing epoch .**



- 300warmUp800train:

![image-20200831082445427](https://i.loli.net/2020/08/31/b8W4SutDZ3JHaNp.png)

```bash
python /home/young/liuyixin/8.29/CAMC/train_ddpg.py --job train --model mobilenet --dataset imagenet --lbound 0.2 --rbound 1 --reward acc_normalize --data_root /home/dataset/imagenet --ckpt_path /home/young/liuyixin/8.29/CAMC/checkpoints/mobilenetv1_imagenet.pth.tar --seed 2020 --data_bsize 64 --n_gpu 1 --warmup 300 --train_episode 800 --suffix mobilenetV1_imagenet_trainDDPG_300warmup800train --rmsize 300
```

****

| prun_ratio | acc5(val) | prune_stategy                                                |                                                              |
| ---------- | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 0.3        | 9.9333    | [1.0, 0.75, 0.625, 0.5625, 0.5625, 0.53125, 0.53125, 0.53125, 0.53125, 0.53125, 0.515625, 0.515625, 0.5, 0.5234375, 0.3125] | [3, 24, 40, 72, 72, 136, 136, 272, 272, 272, 264, 264, 256, 536, 320] |
| 0.5        | 72.2667   | [1.0, 0.75, 0.75, 0.75, 0.75, 0.71875, 0.71875, 0.703125, 0.6875, 0.6875, 0.671875, 0.671875, 0.671875, 0.703125, 0.5625] | [3, 24, 48, 96, 96, 184, 184, 360, 352, 352, 344, 344, 344, 720, 576] |
| 0.7        | 89.9667   | [1.0, 1.0, 0.875, 0.875, 0.875, 0.875, 0.875, 0.859375, 0.84375, 0.828125, 0.796875, 0.78125, 0.78125, 0.7734375, 0.6484375] | [3, 32, 56, 112, 112, 224, 224, 440, 432, 424, 408, 400, 400, 792, 664] |

**Note : Increasing on warmup epoch and memory size seem leads to worse result .**





2. The result of mobilenetv1 on cifar10 (AMC vs CAMC)

- Our:

```bash
python /home/young/liuyixin/8.29/CAMC/train_ddpg.py --job train --model mobilenet --dataset cifar10 --lbound 0.2 --rbound 1 --reward acc_reward --data_root /home/dataset/cifar --ckpt_path /home/young/liuyixin/8.22/amc/checkpoints/mobilenetV1cifar10_origin_acc80.35_earlyStop.tar --seed 2020 --data_bsize 32 --n_gpu 1 --warmup 100 --train_episode 800 --suffix mobilenetV1_cifar10_trainDDPG
```



3. The result of mobilenetv2 on imagenet  (AMC vs CAMC)



4. The result of mobilenetv2 on cifar10  (AMC vs CAMC)



5. The result of preresnet56 on imagenet  (AMC vs CAMC)



6. The result of preresnet56 on cifar10  (AMC vs CAMC)



7. Other available model_arch performance result...







## Thinking Why

1. **Will our training convergent with other acc_reward function? What is the advantages of our designed rewardfunction?**

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



2. **Can we directly reuse the trained ddpg agent?**

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
  - **It seems that there are slightly drop on performance**
- Conlusion
  - 







## Checkpoint Training

### Download

| model_arch              | downloadURL                                                  |
| ----------------------- | ------------------------------------------------------------ |
| mobilenetv1 on Imagenet | http://www.lyx6178.cn:5000/download/mobilenetv1_imagenet.pth.tar |
| mobilenetv1 on cifar    | http://www.lyx6178.cn:5000/download/mobilenetv1_cifar10_top1-acc82.07.tar |

### Script

```bash
    python -W ignore train_model.py \
        --model=mobilenet \
        --dataset=cifar10 \
        --lr=0.05 \
        --n_gpu=4 \
        --batch_size=256 \
        --n_worker=32 \
        --lr_type=cos \
        --n_epoch=100 \
        --wd=4e-5 \
        --seed=2020 \
        --data_root=/home/dataset/cifar \
        --patience=20
 
```

### Performance 

- Our pretrained MobileNet on cifar10

|      | val   | train |
| ---- | ----- | ----- |
| acc1 | 82.07 | 90.91 |
| acc5 | 99.04 | 99.91 |



