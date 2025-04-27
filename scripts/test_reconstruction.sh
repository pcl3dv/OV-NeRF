expname=room0
CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/reconstruction/replica_room0.txt \
--ckpt log/replica/$expname/$expname.th \
--render_only 1 \
--render_train 1