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
