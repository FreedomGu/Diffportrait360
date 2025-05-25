#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
# Step 0.0 :Put the you own image to test under folder sample_data/input_image

# Step 0.1: Using 3DDFA_V2_cropping to get camera pose and crop the image to the right format

model="easy-khair-180-gpc0.8-trans10-025000.pkl"
out="../../sample_data/3DNoise"
target_img="../../sample_data/input_image"

# change the path to your own
PANO_HEAD_MODEL="/mnt/turtle/yuming/diffportrait3D_body/CVPR2024/code_reform/dataset/PanoHead/easy-khair-180-gpc0.8-trans10-025000.pkl"
Head_Back_MODEL="/mnt/turtle/yuming/diffportrait360/back_head-230000.th"
Diff360_MODEL="/mnt/turtle/yuming/diffportrait360/model_state-340000.th"

# Step1: PanoHead 3D aware noise generation
# cd to 3DNOise Generation folder in order to get the 3D aware noise from PanoHead PTI
# cd 3DNoise
# python projector_withseg.py \
# --outdir=${out} \
# --num_steps 200 \
# --target_img=${target_img} \
# --network ${model} \
# --camera_json ../../sample_data/input_image/dataset.json \
# --network ${PANO_HEAD_MODEL}

# cd ..

# # Step2: Genertate Head_Back

torchrun --master_port 14020 inference.py \
--model_config ./model_lib/ControlNet/models/cldm_v15_reference_only_pose_enable_PC.yaml \
--test_dataset back_head_generation \
--control_mode controlnet_important \
--local_image_dir ../sample_data/Back_Head \
--resume_dir ${Head_Back_MODEL} \
--control_type GAN_Generated \
--inference_image_path ../sample_data/input_image \
--nSample 1 \
--condition_path ../sample_data/cam_condition/sphere32 \



# # check sample_data/Back_Head folder to see if the result is correct
# # # Step3: Generate Video

# torchrun --master_port 14031 inference.py \
# --model_config ./model_lib/ControlNet/models/cldm_v15_reference_only_temporal_pose.yaml \
# --test_dataset full_head_clean_inference_final_face \
# --control_mode controlnet_important \
# --local_image_dir ../sample_data/result \
# --resume_dir ${Diff360_MODEL} \
# --control_type GAN_Generated \
# --inference_image_path ../sample_data \
# --nSample 8 \
# --condition_path ../sample_data/cam_condition/sphere32 \
# --denoise_from_guidance \
# --initial_image_path ../sample_data/3DNoise \

$@

