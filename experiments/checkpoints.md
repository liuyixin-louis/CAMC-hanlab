


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



