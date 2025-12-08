#!/bin/bash

python train.py --dataroot ./datasets/photo2monet_moredata --name photo2monet_cyclegan_md --model cycle_gan --display_id -1 --gpu_ids 0 & 

python train.py --dataroot ./datasets/photo2monet_moredata --name photo2monet_cyclegan_md_et --model cycle_gan --early_stopping --display_id -1 --gpu_ids 1 & 


python train.py --dataroot ./datasets/photo2monet_moredata --name photo2monet_cyclegan_md_early_decay --model cycle_gan --early_stopping --lambda_cyc_decay --display_id -1 --gpu_ids 0


## cycleGAN
# lambda decay
python train.py --dataroot ./datasets/photo2monet_moredata --name photo2monet_cyclegan_lambda_decay --model cycle_gan  --lambda_cyc_decay --display_id -1 --gpu_ids 1
# cyclegan pretrain!
tmux new -s train_cyclegan_pretrain
python train.py --dataroot ./datasets/pretrain --name pretrain_cyclegan --model cycle_gan --n_epochs 50 --n_epochs_decay 50 --display_id -1 --gpu_ids 1


# python train.py --dataroot ./datasets/photo2monet_moredata --name photo2monet_cyclegan_md__256 --model cycle_gan --netG unet_256 --n_epochs 200 --n_epochs_decay 200 --display_id -1 --gpu_ids 0

python train.py --dataroot ./datasets/photo2monet_moredata --name photo2monet_cyclegan_md_unet_256 --model cycle_gan --netG unet_256 --n_epochs 200 --n_epochs_decay 200 --display_id -1 --gpu_ids 0 & 
