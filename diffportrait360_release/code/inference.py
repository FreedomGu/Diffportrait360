''' Inference script for DiffPortrait360'''
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from utils.checkpoint import load_from_pretrain
from utils.utils import set_seed, count_param, print_peak_memory
# model
from model_lib.ControlNet.cldm.model import create_model
import imageio
# utils
from dataset import full_head_clean
import copy
from utils.checkpoint import load_from_pretrain
from torchvision.utils import save_image

def load_state_dict(model, ckpt_path, strict=True, map_location="cpu"):
    print(f"Loading model state dict from {ckpt_path} ...")
    state_dict = load_from_pretrain(ckpt_path, map_location=map_location)
    state_dict = state_dict.get('state_dict', state_dict)
    model.load_state_dict(state_dict, strict=strict)
    del state_dict  
def get_cond_control(args, batch_data, control_type, device, model=None,batch_size=None, train=True, seg_model=None):
    # Single-control
    control_type = copy.deepcopy(control_type)[0]
    if control_type == "GAN_Generated" :
        if train:
            #assert "pose_map" in batch_data
            c_cat = batch_data["condition"].cuda()
            cond_image = batch_data["condition_image"].cuda()
            #import pdb;pdb.set_trace()
            
            cond_image = model.get_first_stage_encoding(model.encode_first_stage(cond_image))
            cond_img_cat = cond_image
        else:
            #assert "pose_map_list" in batch_data
            pose_map_list = batch_data["condition"]
            #print("Get Inference Control Type")
            c_cat_list = pose_map_list
            cond_image = batch_data["condition_image"].cuda()
            cond_image = model.get_first_stage_encoding(model.encode_first_stage(cond_image))
            cond_img_cat = cond_image.cuda()
    else:
        raise NotImplementedError(f"cond_type={control_type} not supported!")
    if train:
        return [ c_cat[:batch_size].to(device) ], [cond_img_cat[:batch_size].to(device) ]
    else:
        return pose_map_list, [cond_img_cat]

def visualize(args, name, batch_data, infer_model, case, nSample):
    infer_model.eval()
    # video length #max(nSample, batch_data["image"].squeeze().shape[0])
    cond_imgs = batch_data["condition_image"].cuda()
    #gt = torch.stack(batch_data['image']).squeeze()
    if nSample == 1 :
        conditions = batch_data['condition'].cuda()
        if args.denoise_from_guidance:
            fea_condtion = batch_data['fea_condition'].cuda()#.squeeze()
    else:
        try:
            conditions = torch.stack(batch_data["condition"]).squeeze().cuda()
        except:
            conditions = batch_data["condition"].cuda()
        if args.denoise_from_guidance:
            fea_condtion = batch_data['fea_condition'].cuda().squeeze()
        #import pdb;pdb.pdb.set_trace()
    #print("text_blip:", batch_data["text_blip"])
    text = batch_data["text_blip"]
    c_cross = infer_model.get_learned_conditioning(text)
    c_cross = c_cross.repeat(nSample, 1, 1)
    #import pdb;pdb.set_trace()
    uc_cross = infer_model.get_unconditional_conditioning(nSample)
    gene_img_list = []
    generated_imgs = []
    cond_img = infer_model.get_first_stage_encoding(infer_model.encode_first_stage(cond_imgs))
    cond_img = cond_img.repeat(nSample, 1, 1, 1)
    cond_img_cat = [cond_img]
    #more_cond_imgs = []
    #import pdb;pdb.set_trace()
    if 'extra_appearance' in batch_data:
        #more_cond_imgs = []
        m_cond_img = batch_data['extra_appearance'] # assume only one batch per inference.
        m_cond_img = infer_model.get_first_stage_encoding(infer_model.encode_first_stage(m_cond_img.cuda()))
        m_cond_img = m_cond_img.repeat(nSample, 1, 1, 1)
        more_cond_imgs = m_cond_img#.append([m_cond_img])   
    for i in range(conditions.shape[0] // nSample):
        print("Generate Image {} in {} images".format(nSample * i, conditions.shape[0])) 
        inpaint = None
        if args.denoise_from_guidance:
            #import pdb;pdb.set_trace()
            fea_map_enc = infer_model.get_first_stage_encoding(infer_model.encode_first_stage(fea_condtion[i*nSample: i*nSample+nSample]))
            c = {"c_concat": [conditions[i*nSample: i*nSample+nSample]], "c_crossattn": [c_cross], "image_control": cond_img_cat, 'feature_control':fea_map_enc}
        else:
            c = {"c_concat": [conditions[i*nSample: i*nSample+nSample]], "c_crossattn": [c_cross], "image_control": cond_img_cat}
        if args.control_mode == "controlnet_important":
            uc = {"c_concat": [conditions[i*nSample: i*nSample+nSample]], "c_crossattn": [uc_cross]}
        else:
            uc = {"c_concat": [conditions[i*nSample: i*nSample+nSample]], "c_crossattn": [uc_cross], "image_control": cond_img_cat}
        
        c['wonoise'] = True
        uc['wonoise'] = True
        # generate images
        if 'extra_appearance' in batch_data:
            c['more_image_control'] = [more_cond_imgs]
        # check if ti has alreayd exist:
        # if os.path.isfile((f"{args.local_image_dir}/{batch_data['image_name'][0]}.mp4")):
        #     return
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
            gene_img = infer_model.decode_first_stage(gene_img)
            for j in range(nSample):
                if 'extra_appearance' in batch_data:
                    if 'fea_condition' in batch_data and args.denoise_from_guidance:
                        cated = torch.cat([fea_condtion[i*nSample+j].cpu().squeeze(),gene_img[j].cpu().squeeze(), conditions[i*nSample+j].cpu().squeeze(), cond_imgs.cpu().squeeze(), batch_data['extra_appearance'].cpu().squeeze()],axis=2)
                    else:
                        cated = torch.cat([gene_img[j].cpu().squeeze(), conditions[i*nSample+j].cpu().squeeze(), cond_imgs.cpu().squeeze(), batch_data['extra_appearance'].cpu().squeeze()],axis=2)
                else:
                    cated = torch.cat([gene_img[j].cpu().squeeze(), conditions[i*nSample+j].cpu().squeeze(), cond_imgs.cpu().squeeze()],axis=2)
                cated = cated.clamp(-1, 1).add(1).mul(0.5).permute(1,2,0).cpu().numpy()
                #import pdb;pdb.set_trace()
                gene_img_list.append(cated)
                generated_imgs.append(gene_img[j].squeeze().clamp(-1, 1).add(1).mul(0.5).permute(1,2,0).cpu().numpy())
    if nSample == 1:
        save_image(gene_img.clamp(-1,1).add(1).mul(0.5), args.local_image_dir+"/"+batch_data['image_name'][0]) 
    else:             
        writer_gen = imageio.get_writer(f"{args.local_image_dir}/{batch_data['image_name'][0]}.mp4", fps=10)
        for idm, tensor in enumerate(gene_img_list): 
                writer_gen.append_data(generated_imgs[idm])   
def main(args):
    # ******************************
    # initializing
    # ******************************
    args.device = torch.device("cuda")
    args.num_gpu = torch.cuda.device_count()
    args.use_gpu = torch.cuda.is_available() and args.num_gpu > 0
    #seg_model = load_model(args, args.model_path, True, False)
    os.makedirs(args.local_image_dir,exist_ok=True)
    print(args)
    set_seed(args.seed)
    # ******************************
    # create model
    # ******************************
    model = create_model(args.model_config).cpu()
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control
    model.to(args.device)
    print('Total base  parameters {:.02f}M'.format(count_param([model])))
    # ******************************
    # load pre-trained models
    # ******************************
    ckpt_path = args.resume_dir
    print('loading state dict from {} ...'.format(ckpt_path))
    load_state_dict(model, ckpt_path, strict=True) 
    torch.cuda.empty_cache()
    # ******************************
    # create dataset and dataloader
    # ******************************
    image_transform = T.Compose([
            T.ToTensor(), 
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    if args.test_dataset == 'back_head_generation':
        test_dataset_cls = getattr(full_head_clean, args.test_dataset)
        test_image_dataset = test_dataset_cls(
                                image_transform = image_transform,
                                inference_image_dataset = args.inference_image_path,
                                condition_path = args.condition_path
                            )
    elif args.test_dataset == "full_head_clean_inference_final_face":
       test_dataset_cls = getattr(full_head_clean, args.test_dataset)
       test_image_dataset = test_dataset_cls(
                                image_transform=image_transform,
                                condition_path = args.condition_path,
                                inference_image_dataset = args.inference_image_path,
                                initial_image_path = args.initial_image_path,
                                #extra_appearance_num = args.extra_appearance_num,
                                #mask_condition = args.mask_condition ,
                            )
    else:
       print("find the appropriate dataset class!")
       return
    test_image_dataloader = DataLoader(test_image_dataset, 
                                  batch_size=1,
                                  num_workers=0,
                                  #pin_memory=True,
                                  shuffle=False)
    test_image_dataloader_iter = iter(test_image_dataloader)
    #if args.local_rank == 0:
    print(f"image dataloader created: dataset={args.test_dataset} batch_size={1} ")
    #dist.barrier()
    first_print = True
    infer_model = model.module if hasattr(model, "module") else model
    print(f"start inference loop!") 
    #import pdb;pdb.set_trace()
    for itr in range(0, len(test_image_dataloader)):
        test_batch_data = next(test_image_dataloader_iter)
        with torch.no_grad():
            nSample = int(args.nSample) # video length 
            visualize(args, "val_images", test_batch_data, infer_model, nSample=nSample, case=str(itr))
        # Clear cache
        if first_print or itr % 200 == 0:
            torch.cuda.empty_cache()
            print_peak_memory("Max memory allocated After running {} steps:".format(itr), 0)
        first_print = False
if __name__ == "__main__":
    str2bool = lambda arg: bool(int(arg))
    parser = argparse.ArgumentParser(description='Control Net training')
    ## Model
    parser.add_argument('--model_config', type=str, default="model_lib/ControlNet/models/cldm_v15_video.yaml",
                        help="The path of model config file")
    parser.add_argument('--sd_locked', type =str2bool, default=True,
                        help='Freeze parameters in original stable-diffusion decoder')
    parser.add_argument('--only_mid_control', type =str2bool, default=False,
                        help='Only control middle blocks')
    parser.add_argument("--control_mode", type=str, default="balance",
                        help="Set controlnet is more important or balance.")
    ## Training
    parser.add_argument('--seed', type=int, default=42, 
                        help='random seed for initialization')
    parser.add_argument('--global_step', type=int, default=0,
                        help='initial global step to start with (use with --init_path)')
    ## Data
    parser.add_argument("--local_image_dir", type=str, default=None, required=True,
                        help="The local output directory where generated images will be written.")
    parser.add_argument("--resume_dir", type=str, default=None,
                        help="The resume directory where the model checkpoints will be loaded.")
    parser.add_argument('--control_type', type=str, nargs="+", default=["pose"],
                        help='The type of conditioning')
    parser.add_argument("--test_dataset", type=str, default=None)
    #parser.add_argument('--dataset_root_path', type=str, default=None, help='The path of the dataset')
    parser.add_argument('--inference_image_path', type=str, default=None, help='The path of the inference image')
    parser.add_argument('--denoise_from_guidance', action='store_true', default=False, help='Denoise from guidance')
    parser.add_argument('--initial_image_path', type=str, default=None, help='The path of the initial image')
    parser.add_argument("--condition_path", type=str, default=None, help="The resume directory where the condtiion path will be loaded.")
    parser.add_argument("--nSample", type=int, default=None, help="The resume directory where the condtiion path will be loaded.")
    args = parser.parse_args()
    main(args)