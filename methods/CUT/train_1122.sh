#!/bin/bash

# /home/xiaojinsong/桌面/sdb1_xiaojinsong/big_data/painter/CUT

tmux new -s train_cyclegan
conda init
source ~/.bashrc
conda activate painter
cd /home/xiaojinsong/桌面/sdb1_xiaojinsong/big_data/painter/CycleGAN

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

tmux new -s train_cut



# 随时通过以下命令查看会话：
tmux list-sessions
# 重新连接到某个会话：
tmux attach -t train_cut
# 离开会话
ctrl+b d




## CUT
# basic
python train.py --dataroot ./datasets/photo2monet_moredata --name photo2monet_cut_md --model cut --display_id -1 --gpu_ids 1 &
# cut_epoch600
python train.py --dataroot ./datasets/photo2monet_moredata --name photo2monet_cut_md_epoch_600 --model cut --n_epochs 300 --n_epochs_decay 300 --display_id -1 --gpu_ids 1 &
# cut unet
python train.py --dataroot ./datasets/photo2monet_moredata --name photo2monet_cut_md_unet --model cut --netG unet_256  --display_id -1 --gpu_ids 1 &
# cut_pretrain
python train.py --dataroot ./datasets/pretrain --name pretrain_cut --model cut --n_epochs 100 --n_epochs_decay 200 --display_id -1 --gpu_ids 0,1
# restart from best
python train.py --dataroot ./datasets/photo2monet_moredata --name photo2monet_cut_restart --model cut --lr 0.00005 --n_epochs 10 --n_epochs_decay 90 --lr_policy cosine --display_id -1 --gpu_ids 0 --continue_train --epoch_count 1 --beta1 0.9 
# restart from pretrain
python train.py --dataroot ./datasets/photo2monet_moredata --name photo2monet_cut_pretrain_restart --model cut --lr 0.00005 --n_epochs 50 --n_epochs_decay 200 --lr_policy cosine --display_id -1 --gpu_ids 0 --continue_train --epoch_count 1 --beta1 0.9 

## test
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
