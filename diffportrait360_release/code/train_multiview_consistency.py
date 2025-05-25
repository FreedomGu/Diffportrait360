"""
Train a control net
"""
import os
import sys
sys.path.append('/mnt/turtle/yuming/diffportrait3D_body/CVPR2024/code_reform/dataset/PanoHead')

import argparse
import datetime
import numpy as np
import pdb
from parser import get_parser
import cv2
from tqdm import tqdm
import pdb
# torch
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torchvision import transforms as T
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
#from ema_pytorch import EMA

# distributed 
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP 

# data
from dataset import zju_mocap, smpl_render, full_head_clean
#from dataset.PanoHead import full_head
# utils
from utils.checkpoint import load_from_pretrain, save_checkpoint_ema
from utils.utils import set_seed, count_param, print_peak_memory, anal_tensor, merge_lists_by_index
from utils.lr_scheduler import LambdaLinearScheduler
from dataset.hdfs_io import hexists
#from langdetect import detect
# model
from model_lib.ControlNet.cldm.model import create_model
import copy
TORCH_VERSION = torch.__version__.split(".")[0]
FP16_DTYPE = torch.float16 if TORCH_VERSION == "1" else torch.bfloat16
print(f"TORCH_VERSION={TORCH_VERSION} FP16_DTYPE={FP16_DTYPE}")




def delete_zero_conv_in_controlmodel(state_dict):
    keys_to_delete = [key for key in state_dict.keys() if (key.startswith('control_model.zero_convs') or key.startswith('control_model.middle_block_out')) ]

    for key in keys_to_delete:
        del state_dict[key]
    return state_dict

# You can now use the modified state_dict without the deleted keys

def copy_diffusion_outputblocks(state_dict):

    new_state_dict = copy.deepcopy(state_dict)
    for key, value in state_dict.items():
        if key.startswith('model.diffusion_model.output_blocks'):
            new_key = key.replace('model.diffusion_model.output_blocks', 'control_model.output_blocks')
            new_state_dict[new_key] = value
    return new_state_dict
def copy_diffusion_outputblocks_pose(state_dict):

    new_state_dict = copy.deepcopy(state_dict)
    for key, value in state_dict.items():
        if key.startswith('model.diffusion_model.output_blocks'):
            new_key = key.replace('model.diffusion_model.output_blocks', 'control_model.output_blocks')
            new_state_dict[new_key] = value
    return new_state_dict
def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]
def copy_diffusion_input_blocks_to_pose_control(state_dict):
    
    new_state_dict = copy.deepcopy(state_dict)
    for key, value in state_dict.items():
        is_control, name = get_node_name(key, 'pose_control_model.')
        if is_control:
            try:
                copy_k = 'model.diffusion_model.' + name
                new_state_dict[key] = state_dict[copy_k].clone() 
                print("trasfer {} to {}".format(copy_k, key))
            except:
                print("new layer {}".format(key))
            #import pdb;pdb.set_trace()
    del state_dict
    return new_state_dict 

def copy_appearance_model_input_blocks_to_pose_control(state_dict):
    
    new_state_dict = copy.deepcopy(state_dict)
    for key, value in state_dict.items():
        is_control, name = get_node_name(key, 'control_model.')
        if is_control:
            try:
                copy_k = 'pose_control_model.' + name
                new_state_dict[key] = state_dict[copy_k].clone() 
                print("trasfer {} to {}".format(copy_k, key))
            except:
                print("new layer {}".format(key))
            #import pdb;pdb.set_trace()
    del state_dict
    return new_state_dict 
def copy_diffusion_parameters(state_dict):

    new_state_dict = {}#copy.deepcopy(state_dict)
    for key, value in state_dict.items():
        if key.startswith('control_model'):
            new_key = key.replace('control_model', 'appearance_control_model')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    #import pdb;pdb.set_trace()
    return new_state_dict
def replace_appearance_value_from_pretrained(model, appearance_path, motion_resume_path, strict=True, map_location='cpu'):
    print(f'Loading Appearance model state dict from {appearance_path} ...')
    print(f'Loading resume model state dict from {motion_resume_path} ...')
    state_dict = load_from_pretrain(appearance_path, map_location=map_location)
    state_dict = state_dict.get('state_dict', state_dict)
    mm_state_dict = load_from_pretrain(motion_resume_path, map_location=map_location)
    mm_state_dict = mm_state_dict.get('state_dict', mm_state_dict)
    motion_resume_dir = merge_state_dict_appearance(state_dict, mm_state_dict)
    model.load_state_dict(motion_resume_dir, strict=True)
    del state_dict, mm_state_dict, motion_resume_dir

    #new_state_dict = merge_state_dict_apppose_mm(state_dict, mm_state_dict)
def load_state_dict_apperancepose_motion(model, apppose_path, motion_path, reinit_hint_block=False, strict=True, map_location="cpu"):
    print(f"Loading Appearance + Control model state dict from {apppose_path} ...")
    print(f"Loading Motion model state dict from {motion_path} ...")
    state_dict = load_from_pretrain(apppose_path, map_location=map_location)
    state_dict = state_dict.get('state_dict', state_dict)
    #state_dict = delete_zero_conv_in_controlmodel(state_dict)
    #new_state_dict = copy_diffusion_outputblocks(state_dict)
    mm_state_dict = load_from_pretrain(motion_path, map_location=map_location)
    mm_state_dict = mm_state_dict.get('state_dict', mm_state_dict)
    
    if reinit_hint_block:
        print("Ignoring hint block parameters from checkpoint!")
        for k in list(state_dict.keys()):
            if k.startswith("appearance_control_model.input_hint_block"):
                state_dict.pop(k)
    new_state_dict = merge_state_dict_apppose_mm(state_dict, mm_state_dict)
    
    #pdb.set_trace()
    model.load_state_dict(new_state_dict, strict=False)
    
    #pdb.set_trace()
    
    del state_dict 
def load_state_dict_motion_appearance(model, ckpt_path, motion_path, reinit_hint_block=False, strict=True, map_location="cpu"):
    print(f"Loading Appearance + Control model state dict from {ckpt_path} ...")
    print(f"Loading XPortrait Motion model state dict from {motion_path} ...")
    state_dict = load_from_pretrain(ckpt_path, map_location=map_location)
    state_dict = state_dict.get('state_dict', state_dict)
    mm_state_dict = load_from_pretrain(motion_path, map_location=map_location)
    mm_state_dict = mm_state_dict.get('state_dict', mm_state_dict)
    if reinit_hint_block:
        print("Ignoring hint block parameters from checkpoint!")
        for k in list(state_dict.keys()):
            if k.startswith("control_model.input_hint_block"):
                state_dict.pop(k)
    new_state_dict = merge_appearance_mm(state_dict, mm_state_dict)
    model.load_state_dict(new_state_dict, strict=strict)
    #import pdb;pdb.set_trace()
    del state_dict, new_state_dict
def load_state_dict(model, ckpt_path, reinit_hint_block=False, strict=True, map_location="cpu"):
    print(f"Loading model state dict from {ckpt_path} ...")
    state_dict = load_from_pretrain(ckpt_path, map_location=map_location)
    state_dict = state_dict.get('state_dict', state_dict)
    if args.resume_dir is not None and hexists(os.path.join(args.resume_dir, "optimizer_state_latest.th")):
        new_state_dict = state_dict
    else:      
        new_state_dict = copy_diffusion_outputblocks(state_dict)
    if reinit_hint_block:
        print("Ignoring hint block parameters from checkpoint!")
        for k in list(state_dict.keys()):
            if k.startswith("control_model.input_hint_block"):
                state_dict.pop(k)
    model.load_state_dict(new_state_dict, strict=strict)
    del state_dict, new_state_dict
def merge_state_dict_apppose_mm(apppose_state_dit, mm_state_dict):
    #import pdb;pdb.set_trace()
    
    for key, value in mm_state_dict.items():
        if "motion_modules" in key:
            if "down_blocks.0.motion_modules.0" in key:
                apppose_state_dit[key.replace("down_blocks.0.motion_modules.0","model.diffusion_model.input_blocks_motion_module.0.0")] = value
            elif "down_blocks.0.motion_modules.1" in key:
                apppose_state_dit[key.replace("down_blocks.0.motion_modules.1","model.diffusion_model.input_blocks_motion_module.1.0")] = value
            elif "down_blocks.1.motion_modules.0" in key:
                apppose_state_dit[key.replace("down_blocks.1.motion_modules.0","model.diffusion_model.input_blocks_motion_module.2.0")] = value
            elif "down_blocks.1.motion_modules.1" in key:
                apppose_state_dit[key.replace("down_blocks.1.motion_modules.1","model.diffusion_model.input_blocks_motion_module.3.0")] = value
            elif "down_blocks.2.motion_modules.0" in key:
                apppose_state_dit[key.replace("down_blocks.2.motion_modules.0","model.diffusion_model.input_blocks_motion_module.4.0")] = value
            elif "down_blocks.2.motion_modules.1" in key:
                apppose_state_dit[key.replace("down_blocks.2.motion_modules.1","model.diffusion_model.input_blocks_motion_module.5.0")] = value
            elif "down_blocks.3.motion_modules.0" in key:
                apppose_state_dit[key.replace("down_blocks.3.motion_modules.0","model.diffusion_model.input_blocks_motion_module.6.0")] = value
            elif "down_blocks.3.motion_modules.1" in key:
                apppose_state_dit[key.replace("down_blocks.3.motion_modules.1","model.diffusion_model.input_blocks_motion_module.7.0")] = value
            # elif "down_blocks.3.motion_modules.2" in key:
            #     apppose_state_dit[key.replace("down_blocks.3.motion_modules.2","model.diffusion_model.input_blocks_motion_module.8.0")] = value
            elif "up_blocks.0.motion_modules.0" in key:
                apppose_state_dit[key.replace("up_blocks.0.motion_modules.0","model.diffusion_model.output_blocks_motion_module.0.0")] = value
            elif "up_blocks.0.motion_modules.1" in key:
                apppose_state_dit[key.replace("up_blocks.0.motion_modules.1","model.diffusion_model.output_blocks_motion_module.1.0")] = value
            elif "up_blocks.0.motion_modules.2" in key:
                apppose_state_dit[key.replace("up_blocks.0.motion_modules.2","model.diffusion_model.output_blocks_motion_module.2.0")] = value
            elif "up_blocks.1.motion_modules.0" in key:
                apppose_state_dit[key.replace("up_blocks.1.motion_modules.0","model.diffusion_model.output_blocks_motion_module.3.0")] = value
            elif "up_blocks.1.motion_modules.1" in key:
                apppose_state_dit[key.replace("up_blocks.1.motion_modules.1","model.diffusion_model.output_blocks_motion_module.4.0")] = value
            elif "up_blocks.1.motion_modules.2" in key:
                apppose_state_dit[key.replace("up_blocks.1.motion_modules.2","model.diffusion_model.output_blocks_motion_module.5.0")] = value
            elif "up_blocks.2.motion_modules.0" in key:
                apppose_state_dit[key.replace("up_blocks.2.motion_modules.0","model.diffusion_model.output_blocks_motion_module.6.0")] = value
            elif "up_blocks.2.motion_modules.1" in key:
                apppose_state_dit[key.replace("up_blocks.2.motion_modules.1","model.diffusion_model.output_blocks_motion_module.7.0")] = value
            elif "up_blocks.2.motion_modules.2" in key:
                apppose_state_dit[key.replace("up_blocks.2.motion_modules.2","model.diffusion_model.output_blocks_motion_module.8.0")] = value
            elif "up_blocks.3.motion_modules.0" in key:
                apppose_state_dit[key.replace("up_blocks.3.motion_modules.0","model.diffusion_model.output_blocks_motion_module.9.0")] = value
            elif "up_blocks.3.motion_modules.1" in key:
                apppose_state_dit[key.replace("up_blocks.3.motion_modules.1","model.diffusion_model.output_blocks_motion_module.10.0")] = value
            elif "up_blocks.3.motion_modules.2" in key:
                apppose_state_dit[key.replace("up_blocks.3.motion_modules.2","model.diffusion_model.output_blocks_motion_module.11.0")] = value
        else:
            continue
    #import pdb;pdb.set_trace()
    #del mm_state_dict
    return apppose_state_dit

def merge_appearance_mm(appearance_dict, mm_state_dict):
    '''mm_state_dict from X portrit, appearance_dict from pretrained'''
    #import pdb;pdb.set_trace()
    for key, value in mm_state_dict.items():
        if key.startswith('model.diffusion_model.input_blocks_motion_module') or key.startswith('model.diffusion_model.output_blocks_motion_module'):
                #print('debugging:trasnfering {} layer'.format(key))
                appearance_dict[key] = value
    return appearance_dict
def replace_string(input_str):
    # Define the block mappings
    block_mappings = {
        "down_blocks": "input_blocks",
        "mid_block": "middle_block",
        "up_blocks": "output_blocks" }

    # Using a regular expression to capture the block name, x and y values
    pattern = re.compile(r'(down_blocks|mid_block|up_blocks)\.(\d+)\.motion_modules\.(\d+)\.')

    # Define a substitution function to format the replacement string
    def substitution(match):
        block_name = match.group(1)
        x = int(match.group(2))
        y = int(match.group(3))
        
        # Use the block mapping to get the correct block name
        new_block_name = block_mappings[block_name]
        
        return f'model.diffusion_model.{new_block_name}_motion_module.{x*3 + y}.0.'

    # Perform the replacement
    return pattern.sub(substitution, input_str)    
def load_state_dict_hack(model, ckpt_path, reinit_hint_block=False, strict=True, map_location="cpu"):
    print(f"Loading model state dict from {ckpt_path} ...")
    state_dict = load_from_pretrain(ckpt_path, map_location=map_location)
    state_dict = state_dict.get('state_dict', state_dict)
    if reinit_hint_block:
        print("Ignoring hint block parameters from checkpoint!")
        for k in list(state_dict.keys()):
            if k.startswith("control_model.input_hint_block"):
                state_dict.pop(k)
    model.load_state_dict(state_dict, strict=strict)
    del state_dict   

def replace_keys_in_state_dict(state_dict,str1,str2):
    new_state_dict = {}
    for key, value in state_dict.items():
        if str1 in key:
            new_key = key.replace(str1, str2)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def merge_state_dict(image_state_dict, pose_state_dict_new):
    for key, value in pose_state_dict_new.items():
        if "pose_control_model" in key:
            image_state_dict[key] = value
        else:
            continue
    return image_state_dict
def merge_state_dict_appearance(appearance_state_dict, motion_resume_dir):
    for key,value in appearance_state_dict.items():
        if "control_model" in key:
            motion_resume_dir[key] = value
        elif "pose_control_model" in key:
            motion_resume_dir[key] = value
        else:
            continue
    return motion_resume_dir

def load_state_dict_reference_only(model, ckpt_path, reinit_hint_block=False, strict=True, map_location="cpu"):
    print(f"Loading model state dict from {ckpt_path} ...")
    state_dict = load_from_pretrain(ckpt_path, map_location=map_location)
    state_dict = state_dict.get('state_dict', state_dict)
    state_dict = delete_zero_conv_in_controlmodel(state_dict)
    new_state_dict = copy_diffusion_outputblocks(state_dict)
    if reinit_hint_block:
        print("Ignoring hint block parameters from checkpoint!")
        for k in list(state_dict.keys()):
            if k.startswith("appearance_control_model.input_hint_block"):
                state_dict.pop(k)
    
    model.load_state_dict(new_state_dict, strict=strict)
    del state_dict   

def load_state_dict_image_pose(model, image_ckpt_path, pose_ckpt_path, strict=False, map_location="cpu"):
    print(f"Loading appearance model state dict from {image_ckpt_path} ...")
    
    state_dict = load_from_pretrain(image_ckpt_path, map_location=map_location)
    state_dict = state_dict.get('state_dict', state_dict)
    image_state_dict = replace_keys_in_state_dict(state_dict,"control_model","appearance_control_model")
    # pdb.set_trace()
    #$hack for pose_state_dict
    pose_ckpt_path = '/mnt/turtle/yuming/diffportrait3D_body/CVPR2024/code_reform/model_lib/ControlNet/pretrained_weights/control_sd15_depth.pth'
    print(f"Loading pose model state dict from {pose_ckpt_path} ...")
    #print(f"Loading model state dict from {pose_ckpt_path} ...")
    
    pose_state_dict = load_from_pretrain(pose_ckpt_path, map_location=map_location)
    # pdb.set_trace()
    pose_state_dict = pose_state_dict.get('state_dict', pose_state_dict)
    # pdb.set_trace()
    pose_state_dict_new = replace_keys_in_state_dict(pose_state_dict,"control_model","pose_control_model")
    # pose will use the control net it self as the model.
    # pdb.set_trace()
    state_dict_final = merge_state_dict(image_state_dict,pose_state_dict_new)
    #pdb.set_trace()
    model.load_state_dict(state_dict_final, strict=strict)
    del state_dict, image_state_dict, pose_state_dict, pose_state_dict_new, state_dict_final
        
    #import pdb;pdb.set_trace()
    #del state_dict#, new_state_dict
def load_state_dict_image(model, ckpt_path, reinit_hint_block=False, strict=True, map_location='cpu'):
    print(f"Loading model state dict from {ckpt_path} ...")
    state_dict = load_from_pretrain(ckpt_path, map_location=map_location)
    if reinit_hint_block:
        print("Ignoring hint block parameters from checkpoint!")
        for k in list(state_dict.keys()):
            if k.startswith("control_model.input_hint_block"):
                state_dict.pop(k)
    
    model.load_state_dict(state_dict, strict = False)
        
    #import pdb;pdb.set_trace()
    del state_dict#, new_state_dict
 
def get_cond_control(args, batch_data, control_type, device, model=None, batch_size=None, train=True):
    # Single-control
    control_type = copy.deepcopy(control_type)[0]
    if control_type == "GAN_Generated" :
        if train:
            #assert "pose_map" in batch_data
            condition = batch_data["condition"].cuda() # 1 , B, C , H , W
            c_cat = condition
            cond_image = batch_data["condition_image"].cuda() # 1, C, H , W
            #import pdb;pdb.set_trace()
            
            cond_image = model.get_first_stage_encoding(model.encode_first_stage(cond_image))
            cond_img_cat = cond_image
            cond_img_cat = cond_img_cat.repeat(condition.shape[1],1,1,1)
        else:
            #assert "pose_map" in batch_data
            condition = batch_data["condition"].cuda() # 1 , B, C , H , W
            c_cat = condition
            cond_image = batch_data["condition_image"].cuda() # 1, C, H , W
            #import pdb;pdb.set_trace()
            
            cond_image = model.get_first_stage_encoding(model.encode_first_stage(cond_image))
            cond_img_cat = cond_image
            cond_img_cat = cond_img_cat.repeat(condition.shape[1],1,1,1)
        more_cond_imgs = None
        if 'extra_appearance' in batch_data:
            more_cond_imgs = []
            m_cond_img = batch_data['extra_appearance'] # assume only one batch per inference.
            m_cond_img = model.get_first_stage_encoding(model.encode_first_stage(m_cond_img.cuda()))
            m_cond_img = m_cond_img.repeat(condition.shape[1], 1, 1, 1)
            more_cond_imgs = m_cond_img#.append([m_cond_img])
    else:
        raise NotImplementedError(f"cond_type={control_type} not supported!")

    if train:
       # pdb.set_trace()
        return [ c_cat[:batch_size].squeeze().to(device) ], [cond_img_cat[:batch_size].squeeze().to(device) ], more_cond_imgs 
    else:

        # for c_cat in c_cat_list:
        #     if args.control_dropout > 0:
        #         mask = torch.rand((c_cat.shape[0],1,1,1)) > args.control_dropout
        #         c_cat = c_cat * mask.type(torch.float32).to(device)
        #     # pdb.set_trace()
        #    return_list.append([c_cat[:batch_size].to(device)])
        return [ c_cat[:batch_size].squeeze().to(device) ], [cond_img_cat[:batch_size].squeeze().to(device) ], more_cond_imgs 

def visualize(args, name, batch_data, tb_writer, infer_model, global_step, nSample, nTest=1,seg_model=None):
    
    infer_model.eval()
    nSample = max(nSample, batch_data["image"].squeeze().shape[0])
    target_imgs = batch_data["image"].squeeze().cuda()
    more_image_list = [more_img.cuda() for more_img in batch_data["extra_appearance"]]
    text = batch_data["text_blip"]
    c_cat_list, cond_img_cat, more_cond_imgs = get_cond_control(args, batch_data, args.control_type, args.device, model=infer_model, batch_size=nSample, train=False)
    c_cross = infer_model.get_learned_conditioning(text)
    c_cross = c_cross.repeat(nSample, 1, 1)
    uc_cross = infer_model.get_unconditional_conditioning(nSample)
    gene_img_list = []
    rec_image_list = []
    #for img_num in range(len(target_imgs)):
    print("Generate Image {} in {} images".format(target_imgs.shape[0], target_imgs.shape[0])) 
    #import pdb;pdb.set_trace()
    c = {"c_concat": c_cat_list, "c_crossattn": [c_cross], "image_control": cond_img_cat}
    if args.control_mode == "controlnet_important":
        uc = {"c_concat": c_cat_list, "c_crossattn": [uc_cross]}
    else:
        uc = {"c_concat": c_cat_list, "c_crossattn": [uc_cross], "image_control":cond_img_cat}
    if args.wonoise:
        c['wonoise'] = True
        uc['wonoise'] = True
    else:
        c['wonoise'] = False
        uc['wonoise'] = False
    if 'more_image_control' in batch_data:
        c['more_image_control'] = [more_cond_imgs]
    inpaint = None
    #import pdb; pdb.set_trace()
    # generate images
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        infer_model.to(args.device)
        infer_model.eval()
        gene_img, _ = infer_model.sample_log(cond=c,
                                batch_size=nSample, ddim=True,
                                ddim_steps=50, eta=0.5,
                                unconditional_guidance_scale=3,
                                unconditional_conditioning=uc,
                                inpaint=inpaint
                                )
        
        gene_img = infer_model.decode_first_stage( gene_img )
        gene_img_list.append(gene_img.clamp(-1, 1).cpu())
    image = target_imgs.squeeze()
    #import pdb;pdb.set_trace()
    latent = infer_model.get_first_stage_encoding(infer_model.encode_first_stage(image))
    rec_image = infer_model.decode_first_stage(latent)

    rec_image_list.append(rec_image)
    #for i in batch_data['more_image_control']:
    #    rec_image_list.append(i)

    #import pdb;pdb.set_trace()

    cated = torch.cat([rec_img.cpu() for rec_img in rec_image_list] + [c_cat_list[0].cpu()] + gene_img_list + [batch_data['condition_image'].cpu()] +[more_img.cpu()[None,...] for more_img in more_image_list])
    
    cated = (cated.clamp(-1,1) + 1) * 0.5
    save_image(cated, f"{args.local_image_dir}/{args.control_type[0]}_{name}_test_gan_{global_step}.jpg")
    #import pdb;pdb.set_trace()
    #.clamp(-1,1).add(1).mul(0.5)

    tb_writer.add_text(f'{name}_{args.control_type}_caption',f"{str(text)}", global_step )


def main(args):
    torch.manual_seed(42)
    # ******************************
    # initialize training
    # ******************************
    # assign rank
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12360'
    
    
    args.world_size = int(os.environ['WORLD_SIZE'])
    
    args.local_rank = int(os.environ['LOCAL_RANK'])
    print("local_rank", args.local_rank)
    #import pdb;pdb.set_trace()
    args.rank = int(os.environ['RANK'])
    args.device = torch.device("cuda", args.local_rank)
    args.num_gpu = torch.cuda.device_count()
    args.use_gpu = torch.cuda.is_available() and args.num_gpu > 0
    #seg_model = load_model(args, args.model_path, True, False)
    os.makedirs(args.local_image_dir,exist_ok=True)
    os.makedirs(args.local_log_dir,exist_ok=True)
    if args.rank == 0:
        print(args)

    # initial distribution comminucation
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    torch.backends.cuda.matmul.allow_tf32 = False # it doenst work once not trained in A100 64G gpu
    torch.backends.cudnn.benchmark = True

    # set seed for reproducibility
    set_seed(args.seed)

    # visdom / tensorboard
    if args.rank == 0:
        tb_writer = SummaryWriter(log_dir=args.local_log_dir)
    else:
        tb_writer = None
    
    # ******************************
    # create model
    # ******************************
    #import pdb;pdb.set_trace()
    model = create_model(args.model_config).cpu()

    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control
    model.to(args.local_rank)

    #seg_model.to(args.local_rank)
    if args.local_rank == 0:
        print('Total base  parameters {:.02f}M'.format(count_param([model])))
    model_ema = None
    # # initialize the whole model with zero weights
    # for param in model.parameters():
    #     param.data.fill_(0.0)
    # print("Model initialized with zero weights.")
    # ******************************
    # load pre-trained models
    # ******************************
    optimizer_state_dict = None
    global_step = args.global_step
    args.resume_dir = args.resume_dir or args.output_dir
    # load model state dict without motion module initilization
    if args.mm_path_dir is None and hexists(os.path.join(args.resume_dir, "optimizer_state_latest.th")):
        print('Loding optimizer state dict from {} ...'.format(os.path.join(args.resume_dir, "optimizer_state_latest.th")))
        optimizer_state_dict = load_from_pretrain(os.path.join(args.resume_dir, "optimizer_state_latest.th"), map_location="cpu")
        if global_step == 0:
            global_step = optimizer_state_dict["step"]
        #if args.local_rank == 0:
            #assert hexists(os.path.join(args.resume_dir, f"model_state-{global_step}.th"))
        # replace the appearance module with 
        if args.appearance_dir is not None:
            if args.local_rank == 0:
                #assert hexists(args.appearance_dir)
                replace_appearance_value_from_pretrained(model, args.appearance_dir, os.path.join(args.resume_dir, f"model_state-{global_step}.th"))
        else:
            load_state_dict(model, os.path.join(args.resume_dir, f"model_state-{global_step}.th"), strict=False)
    else:
        
        # only consider appearance model , M motion controlnet model as well as pose control model
        
        if args.resume_dir is not None and hexists(os.path.join(args.resume_dir, "optimizer_state_latest.th")):
            print('Loading optimizer state dict from {} ...'.format(os.path.join(args.resume_dir, "optimizer_state_latest.th")))
            optimizer_state_dict = load_from_pretrain(os.path.join(args.resume_dir, "optimizer_state_latest.th"), map_location = "cpu")
            if global_step == 0 :
                global_step = 320000#optimizer_state_dict["step"]
            # loading Motion Module model
            if args.mm_path_dir is not None :
                if args.local_rank == 0:
                    assert hexists(args.mm_path_dir)
                    if args.mm_path_dir.endswith('mm_sd_v15.ckpt'):
                        load_state_dict_apperancepose_motion(model, os.path.join(args.resume_dir, f"model_state-{global_step}.th"), args.mm_path_dir,  reinit_hint_block=True)
                    else:
                        load_state_dict_motion_appearance(model, os.path.join(args.resume_dir, f"model_state-{global_step}.th"), args.mm_path_dir,  reinit_hint_block=True)
            #pdb.set_trace()

    

    # ******************************
    # create optimizer
    # ******************************
    if args.finetune_all:
        params = list(model.control_model.parameters())
        params += list(model.model.diffusion_model.parameters())
        try:
            params += list(model.image_control_model.parameters())
        except:
            pass
        print("Finetune All.")
    elif args.finetune_imagecond_unet:
        params = list(model.model.diffusion_model.parameters())
        try:
            params += list(model.image_control_model.parameters())
        except:
            pass
        for p in model.control_model.parameters():
            p.requires_grad_(False)
        print("Finetune Unet and image controlnet, freeze pose controlnet")
    elif args.finetune_attn:
        
        for b in list(model.model.diffusion_model.input_blocks) + [model.model.diffusion_model.middle_block] + list(model.model.diffusion_model.output_blocks) + list(model.model.diffusion_model.out):
            for p in b.parameters():
                p.requires_grad_(False)
        self_attn_parameters = []
        for name, param in model.model.diffusion_model.named_parameters():
            if "attn1" in name:
                param.requires_grad_(True)
                self_attn_parameters.append(param)
        params = list(model.control_model.parameters())
        print("Train controlnet and attention layers in UNet")
        params += self_attn_parameters
        if not args.sd_locked:
            params += list(model.model.diffusion_model.output_blocks.parameters())
            params += list(model.model.diffusion_model.out.parameters())
    elif args.finetune_control:
        for b in list(model.model.diffusion_model.input_blocks) + [model.model.diffusion_model.middle_block] + list(model.model.diffusion_model.output_blocks) + list(model.model.diffusion_model.out):
            for p in b.parameters():
                p.requires_grad_(False)
        for pd in model.pose_control_model.parameters():
            pd.requires_grad_(True)
        params = list(model.pose_control_model.parameters())
        print("Train controlnet layers in UNet")
        #params += list(model.control_model.parameters())
        if not args.sd_locked:
            params += list(model.model.diffusion_model.output_blocks.parameters())
            params += list(model.model.diffusion_model.out.parameters())  
    elif args.finetune_mmonly:
        for b in list(model.pose_control_model.input_blocks) + [model.pose_control_model.middle_block] + list(model.pose_control_model.input_hint_block) + list(model.pose_control_model.middle_block_out)  + list(model.pose_control_model.zero_convs):
            for p in b.parameters():
                p.requires_grad_(False)
        for b in list(model.model.diffusion_model.input_blocks) + [model.model.diffusion_model.middle_block] + list(model.model.diffusion_model.output_blocks) + list(model.model.diffusion_model.out):
            for p in b.parameters():
                p.requires_grad_(False)
        for p in model.control_model.parameters():
            p.requires_grad_(False)
        print("Freeze pose + appearance + UNet only motion module")
        #params = list(model.model.diffusion_model.input_blocks_motion_module.parameters())+ list(model.model.diffusion_model.middle_block_motion_module.parameters()) + list(model.model.diffusion_model.output_blocks_motion_module.parameters())
        params = list(model.model.diffusion_model.input_blocks_motion_module.parameters()) + list(model.model.diffusion_model.output_blocks_motion_module.parameters())
        if args.finetune_mmonlyattn:
            self_attn_parameters = []
            for name, param in model.model.diffusion_model.named_parameters():
                if "attn1" in name:
                    param.requires_grad_(True)
                    self_attn_parameters.append(param)
            for p in model.control_model.parameters():
                p.requires_grad_(True)
            params_attn = list(model.control_model.parameters())
            print("Train appearance attention layers in UNet")
            params = params + self_attn_parameters + params_attn
        if args.finetune_mmonlypose:
            for pd in model.pose_control_model.parameters():
                pd.requires_grad_(True)
            params_pose = list(model.pose_control_model.parameters())
            print("Train pose_control layers in UNet")
            params += params_pose
        print("Train Motion Module") 
        #import pdb;pdb.set_trace()
    else:
        for b in list(model.model.diffusion_model.input_blocks) + [model.model.diffusion_model.middle_block] + list(model.model.diffusion_model.output_blocks) + list(model.model.diffusion_model.out):
            for p in b.parameters():
                p.requires_grad_(False)
        params = list(model.control_model.parameters())
        try:
            params += list(model.image_control_model.parameters())
        except:
            pass
        if not args.sd_locked:
            params += list(model.model.diffusion_model.output_blocks.parameters())
            params += list(model.model.diffusion_model.out.parameters())
        print("Train controlnet")
    
    optimizer = ZeroRedundancyOptimizer(
        params = params,
        lr=args.lr,
        optimizer_class=torch.optim.AdamW,
        weight_decay=args.weight_decay,
    )
    #using adam
    #optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.use_fp16 and FP16_DTYPE==torch.float16))
    
    # load optimizer state dict
    if optimizer_state_dict is not None and args.load_optimizer_state:
        try:
            optimizer.load_state_dict(optimizer_state_dict["state_dict"])
            if 'scaler_state_dict' in optimizer_state_dict: 
                scaler.load_state_dict(optimizer_state_dict["scaler_state_dict"])
            print("load optimizer done")
        except:
            print("WARNING: load optimizer failed")
        del optimizer_state_dict
    #import pdb;pdb.set_trace()
    #import pdb;pdb.set_trace()
    # create lr scheduler
    lr_scheduler = LambdaLinearScheduler(warm_up_steps=[ args.lr_anneal_steps ], cycle_lengths=[ 10000000000000 ], # incredibly large number to prevent corner cases
                                            f_start=[ 1.e-6 ], f_max=[ 1. ], f_min=[ 1. ])
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler.schedule)
    lr_scheduler.step(global_step)

    # ******************************
    # create DDP model
    # ******************************
    if args.compile and TORCH_VERSION == "2":
        model = torch.compile(model)
    torch.cuda.set_device(args.local_rank)
    model = DDP(model, 
                device_ids=[args.local_rank], 
                output_device=args.local_rank,
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=True,
                gradient_as_bucket_view=True # will save memory
    )
    print_peak_memory("Max memory allocated after creating DDP", args.local_rank)
    
    # ******************************
    # create dataset and dataloader
    # ******************************
    image_transform = T.Compose([
            #RemoveWhite(),
            #T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.ToTensor(), 
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    if args.train_dataset == "full_head_clean_real_data_temporal":
        dataset_cls = getattr(full_head_clean, args.train_dataset)
        test_dataset_cls = getattr(full_head_clean, args.train_dataset)
        root_path = args.dataset_root_path
        test_root_path = args.dataset_root_path #+ '_test'       
        image_dataset = dataset_cls(
                                image_transform=image_transform,
                                root_path= root_path,
                                #mask_condition = args.mask_condition, 
                                )
        
        test_image_dataset = test_dataset_cls(
                                image_transform=image_transform,
                                root_path= root_path,
                                #mask_condition = args.mask_condition, 
                                )
    image_dataloader = DataLoader(image_dataset, 
                                    batch_size=args.train_batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=True,
                                    pin_memory=True)
    image_dataloader_iter = iter(image_dataloader)

    test_image_dataloader = DataLoader(test_image_dataset, 
                                  batch_size=args.val_batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True,
                                  pin_memory=True)
    test_image_dataloader_iter = iter(test_image_dataloader)

    if args.local_rank == 0:
        print(f"image dataloader created: dataset={args.train_dataset} batch_size={args.train_batch_size} control_type={args.control_type}")


    dist.barrier()
    local_step = 0
    loss_list = []
    first_print = True
    infer_model = model.module if hasattr(model, "module") else model
    
    print(f"[rank{args.rank}] start training loop!")
    #estimate_deviation(args, infer_model, tb_writer, global_step)
    
    for itr in range(global_step, args.num_train_steps):

        # Get input
        #print("global_step:", global_step)
        batch_data = next(image_dataloader_iter)
        with torch.no_grad():
            image = batch_data["image"].to(args.device) #256, -1~1
            if args.random_mask:
                text_blip = batch_data["text_blip"]
                text_bg = batch_data["text_bg"]
                text = merge_lists_by_index(text_bg, text_blip)
                # pdb.set_trace()
            elif args.mask_bg or False:
                text = batch_data["text_bg"]
            else:
                text = batch_data["text_blip"]
            for i in range(len(text)):
                if not args.with_text:
                    text[i] = ""
                else:
                    if np.random.random() <= args.empty_text_prob:
                        text[i] = ""
            #import pdb;pdb.set_trace()
            # from 1 B C H W to B C H W
            x = infer_model.get_first_stage_encoding(infer_model.encode_first_stage(image.squeeze()))
            #pdb.set_trace()
            c_cat_list, cond_img_cat, more_image_cond = get_cond_control(args, batch_data, args.control_type, args.device, model=infer_model, train=True) # list o f[b,C,512,512]
            #
            c_cross = infer_model.get_learned_conditioning(text)
            c_cross = c_cross.repeat(x.shape[0], 1, 1)
            cond = {"c_concat": c_cat_list, "c_crossattn": [c_cross], "image_control": cond_img_cat, "more_image_control": [more_image_cond]}
            
            
            #import pdb;pdb.set_trace()
            if args.wonoise:
                cond['wonoise'] = True
            else:
                cond['wonoise'] = False
            #import pdb;pdb.set_trace()
        # Foward
        model.train()
        with torch.cuda.amp.autocast(enabled=args.use_fp16, dtype=FP16_DTYPE):
            #import pdb;pdb.set_trace()
            loss, loss_dict = model(x, cond)
            #import pdb;pdb.set_trace()
            loss_list.append(loss.item())
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
        if first_print:
            print_peak_memory("Max memory allocated after FOWARD first step", args.local_rank)
        
        # Backward
        scaler.scale(loss).backward()
        local_step += 1
        #dist.destroy_process_group()
        if first_print:
            print_peak_memory("Max memory allocated after BACKWARD first step", args.local_rank)
        
        # Optimize
        if local_step % args.gradient_accumulation_steps == 0:
            if not scaler.is_enabled():
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)   
            local_step = 0
            global_step += 1

        # log losses
        if args.local_rank == 0 and args.logging_steps > 0 and (global_step % args.logging_steps == 0 or global_step < 1000000):
            lr = optimizer.param_groups[0]['lr']
            logging_loss = np.mean(loss_list) if len(loss_list) > 0 else 0
            memory_usage = torch.cuda.max_memory_allocated(args.local_rank) // 1e6
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + 
                f" Step {global_step:06}: loss {logging_loss:-8.6f} lr {lr:-.9f} memory {memory_usage:.1f}")
            if args.rank == 0:
                tb_writer.add_scalar("Loss", loss_list[-1], global_step)
            loss_list = []

        # Log images
        if args.rank == 0 and args.logging_gen_steps > 0 and (global_step % args.logging_gen_steps == 0 or global_step in [1,100]):
            #import pdb;pdb.set_trace()
            test_batch_data=next(test_image_dataloader_iter)
            #pdb.set_trace()
            with torch.no_grad():
                #nSample = min(args.train_batch_size, args.val_batch_size)
                nSample= test_batch_data["condition"].shape[1]
                # pdb.set_trace()
                visualize(args, "imgs", test_batch_data, tb_writer, infer_model, global_step, nSample=nSample, nTest=1)

        # Save models
        if args.save_steps > 0 and global_step % args.save_steps == 0:
            optimizer.consolidate_state_dict(to=0)
            if args.rank==0:
                print("Saving model checkpoint to %s", args.output_dir)
                models_state = dict()
                models_state[0] = infer_model.state_dict()
                if model_ema is not None:
                    models_state[args.ema_rate] = model_ema.ema_model.state_dict()
                save_checkpoint_ema(output_dir=args.output_dir,
                                models_state=models_state,
                                epoch=global_step,
                                optimizer_state=optimizer.state_dict(),
                                scaler_state=scaler.state_dict())
                print(f"model saved to {args.output_dir}")

        # Clear cache
        if first_print or global_step % 200 == 0:
            torch.cuda.empty_cache()
            print_peak_memory("Max memory allocated After running {} steps:".format(global_step), args.local_rank)

        first_print = False


if __name__ == "__main__":
    args = get_parser()
    main(args)