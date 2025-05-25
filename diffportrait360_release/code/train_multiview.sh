python -m torch.distributed.run --master_port 12360 --master_addr 'localhost' --nproc_per_node 1 train_multiview_consistency.py \
--model_config ./model_lib/ControlNet/models/cldm_v15_reference_only_temporal_pose.yaml \
--output_dir  /mnt/turtle/yuming/diffportrait3D_body/image_log_twoapperance/mm_supp \
--train_batch_size 1 \
--num_workers 8 \
--img_bin_limit 10 \
--control_mode controlnet_important \
--control_type GAN_Generated \
--train_dataset full_head_clean_real_data_temporal \
--with_text \
--wonoise \
--mask_bg \
--local_image_dir /mnt/turtle/yuming/diffportrait3D_body/image_log_twoapperance/mm_supp  \
--local_log_dir /mnt/turtle/yuming/diffportrait3D_body/image_log_twoapperance/mm_supp \
--dataset_root_path /mnt/turtle/yuming/Data_Supp/data_inference_v2 \
--more_image_control \
--finetune_mmonly \
--save_steps 5000 \
--logging_gen_steps 500 \

$@