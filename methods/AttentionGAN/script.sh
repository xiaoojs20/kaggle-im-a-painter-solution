python train.py --dataroot ./datasets/photo2monet_moredata/ --name photo2monet_attentiongan_epoch_200 --model attention_gan --dataset_mode unaligned --no_dropout --norm instance --lambda_A 10 --lambda_B 10 --lambda_identity 0.5 --niter 100 --niter_decay 100 --gpu_ids 0 --display_id -1  --print_freq 100


python train.py --dataroot ./datasets/photo2monet_moredata/ --name photo2monet_attentiongan_epoch_400 --model attention_gan --dataset_mode unaligned --no_dropout --norm instance --lambda_A 10 --lambda_B 10 --lambda_identity 0.5 --niter 200 --niter_decay 200 --gpu_ids 1 --display_id -1  --print_freq 100

python train.py --dataroot ./datasets/photo2monet_moredata/ --name photo2monet_attentiongan_unet256_epoch_400 --model attention_gan --dataset_mode unaligned --no_dropout --norm instance --lambda_A 10 --lambda_B 10 --lambda_identity 0.5 --niter 200 --niter_decay 200 --gpu_ids 1 --display_id -1  --print_freq 100 --netG unet_256

python train.py --dataroot ./datasets/photo2monet_moredata/ --name photo2monet_attentiongan_pretrain --model attention_gan --no_dropout --norm instance --niter 100 --niter_decay 100 --gpu_ids 0 --display_id -1  --print_freq 100 


tmux new -s attn_gan
conda init
source ~/.bashrc
conda activate painter
cd /home/xiaojinsong/桌面/sdb1_xiaojinsong/big_data/painter//AttentionGAN