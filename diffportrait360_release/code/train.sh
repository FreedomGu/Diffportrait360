#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python -m torch.distributed.run --master_port 12355 --master_addr 'localhost' --nproc_per_node 1 train.py \
--model_config ./model_lib/ControlNet/models/cldm_v15_reference_only_pose_enable_PC.yaml \
--output_dir  ./control \
--train_batch_size 1 \
--num_workers 4 \
--img_bin_limit 10 \
--control_mode controlnet_important \
--control_type GAN_Generated \
--train_dataset full_head_clean_real_data_single_frame \
--with_text \
--local_image_dir ./image_log \
--local_log_dir ./log_board \
--finetune_attn \
--finetune_control \
--save_steps 5000 \
--resume_dir /mnt/turtle/yuming/diffportrait360/back_head-230000.th --dataset_root_path /mnt/turtle/yuming/Sphere_sample \ 
# change to your ckpt path and your datapath 
$@


# train mm


