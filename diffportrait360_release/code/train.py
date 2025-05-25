import os
import re
import datetime
import numpy as np
from parser import get_parser
# torch
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms as T
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
#from ema_pytorch import EMA

# distributed 
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP 

# data
from dataset import full_head_clean
from utils.checkpoint import load_from_pretrain, save_checkpoint_ema
from utils.utils import set_seed, count_param, print_peak_memory, merge_lists_by_index
from utils.lr_scheduler import LambdaLinearScheduler
from dataset.hdfs_io import hexists
from model_lib.ControlNet.cldm.model import create_model
import copy
TORCH_VERSION = torch.__version__.split(".")[0]
FP16_DTYPE = torch.float16 #if TORCH_VERSION == "1" else torch.bfloat16
print(f"TORCH_VERSION={TORCH_VERSION} FP16_DTYPE={FP16_DTYPE}")
torch.set_default_tensor_type(torch.FloatTensor)
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
            condition = batch_data["condition"].cuda()
            c_cat = condition
            cond_image = batch_data["condition_image"].cuda()
            
            
            cond_image = model.get_first_stage_encoding(model.encode_first_stage(cond_image))
            cond_img_cat = cond_image
        else:
            #assert "pose_map_list" in batch_data
            pose_map_list = [batch_data["condition"]]
            #print("Get Inference Control Type")
            c_cat_list = pose_map_list
            cond_image = batch_data["condition_image"].cuda()
            cond_image = model.get_first_stage_encoding(model.encode_first_stage(cond_image))
            cond_img_cat = cond_image
        more_cond_imgs = None
        #import pdb;pdb.set_trace()
    else:
        raise NotImplementedError(f"cond_type={control_type} not supported!")

    if train:
        if args.control_dropout > 0:
                mask = torch.rand((c_cat.shape[0],1,1,1)) > args.control_dropout
                c_cat = c_cat * mask.type(torch.float16).to(device)
        return [ c_cat[:batch_size].to(device) ], [cond_img_cat[:batch_size].to(device) ], more_cond_imgs
    else:
        return_list = []

        for c_cat in c_cat_list:
            if args.control_dropout > 0:
                mask = torch.rand((c_cat.shape[0],1,1,1)) > args.control_dropout
                c_cat = c_cat * mask.type(torch.float16).to(device)
            # pdb.set_trace()
            return_list.append([c_cat[:batch_size].to(device)])
        return return_list, [cond_img_cat[:batch_size].to(device) ], more_cond_imgs

def visualize(args, name, batch_data, tb_writer, infer_model, global_step, nSample, nTest=1,seg_model=None):
    
    infer_model.eval()

    # get inputs
    # real_image = batch_data["image_list"].to(args.device)[:nSample] #256, -1~1
    real_image_list = [real_image.cuda() for real_image in batch_data["image"]]

    
    if args.random_mask:
        text_blip = batch_data["text_blip"][:nSample]
        text_bg = batch_data["text_bg"][:nSample]
        text = merge_lists_by_index(text_bg, text_blip)

    elif args.mask_bg:
        text = batch_data["text_bg"][:nSample]
    else:
        text = batch_data["text_blip"][:nSample]
    if not args.with_text:
        for i in range(len(text)):
            text[i] = ""
    if args.text_prompt is not None:
        for i in range(len(text)):
            text[i] = args.text_prompt


    c_cat_list, cond_img_cat, more_cond_imgs = get_cond_control(args, batch_data, args.control_type, args.device, model=infer_model,batch_size=nSample, train=False, seg_model=seg_model)
    c_cross = infer_model.get_learned_conditioning(text)[:nSample]
    uc_cross = infer_model.get_unconditional_conditioning(nSample)
    gene_img_list = []
    pose_map_list = []
    rec_image_list = []
    for img_num in range(len(real_image_list)):
        print("Generate Image {} in {} images".format(img_num,len(real_image_list))) 
        # pdb.set_trace()
        c = {"c_concat": c_cat_list[img_num], "c_crossattn": [c_cross], "image_control":cond_img_cat}
        if args.control_mode == "controlnet_important":
            uc = {"c_concat": c_cat_list[img_num], "c_crossattn": [uc_cross]}
        else:
            uc = {"c_concat": c_cat_list[img_num], "c_crossattn": [uc_cross], "image_control":cond_img_cat}
        pose_map_list.append(c_cat_list[img_num][0][:,:3].cpu())
        if args.wonoise:
            c['wonoise'] = True
            uc['wonoise'] = True
        else:
            c['wonoise'] = False
            uc['wonoise'] = False
        if 'more_image_control' in batch_data:
            c['more_image_control'] = [more_cond_imgs]
        inpaint = None
        inpaint_list = []

        # generate images
        with torch.cuda.amp.autocast(enabled=False, dtype=torch.float16):
            # for _ in range(nTest):
            infer_model.to(args.device)
            infer_model.eval()
            #import pdb;pdb.set_trace()
            gene_img, _ = infer_model.sample_log(cond=c,
                                    batch_size=nSample, ddim=True,
                                    ddim_steps=50, eta=0.5,
                                    unconditional_guidance_scale=3,
                                    unconditional_conditioning=uc,
                                    inpaint=inpaint
                                    )
            gene_img = infer_model.decode_first_stage( gene_img )
            gene_img_list.append(gene_img.clamp(-1, 1).cpu())
        image = real_image_list[img_num][None,...]
        #more_image = batch_data['extra_appearance'][img_num]
        #import pdb;pdb.set_trace()
        latent = infer_model.get_first_stage_encoding(infer_model.encode_first_stage(image))
        rec_image = infer_model.decode_first_stage(latent)
        rec_image_list.append(rec_image)
    if args.random_mask:
        cond_image = batch_data["condition_image"].cpu() * (1 - batch_data["randommask"].cpu())
    else:
        cond_image = batch_data["condition_image"].cpu()
        cated = torch.cat([rec_img.cpu() for rec_img in rec_image_list] + pose_map_list + gene_img_list +[cond_image.cpu()] )
    # rec_img is the target ,,,, pose_map is the camera pose control map ,,,, gene_img is the generated image ,,,, cond_image is the condition image
    cated = cated.clamp(-1,1).add(1).mul(0.5)
    save_image(cated, f"{args.local_image_dir}/{args.control_type[0]}_{name}_test_gan_{global_step}.jpg")
    #import pdb;pdb.set_trace()
    #.clamp(-1,1).add(1).mul(0.5)
    
    tb_writer.add_text(f'{name}_{args.control_type}_caption',f"{str(text)}", global_step )



def main(args):
    
    # ******************************
    # initialize training
    # ******************************
    # assign rank    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    
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
    dist.init_process_group("nccl",  rank=args.rank, world_size=args.world_size)
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
    
    # ******************************

    # ******************************
    # load pre-trained models
    # ******************************
    optimizer_state_dict = None
    global_step = args.global_step
    #args.resume_dir = args.resume_dir 
    if args.resume_dir is not None:
        optimizer_state_dict = load_state_dict(model, args.resume_dir) 
        global_step = 0
        torch.cuda.empty_cache()


    # ******************************
    # create optimizer
    # ******************************
    if args.finetune_attn:
        for b in list(model.model.diffusion_model.input_blocks) + [model.model.diffusion_model.middle_block] + list(model.model.diffusion_model.output_blocks) + list(model.model.diffusion_model.out):
            for p in b.parameters():
                p.requires_grad_(False)
        params = list(model.control_model.parameters())
        print("Train controlnet attention layers in UNet")
        if hasattr(model, 'dual_control_model'):
            params += list(model.dual_control_model.parameters())
            print("!!!!!!!!!!Train dual appearance in UNet")
            #import pdb;pdb.set_trace()
        if args.finetune_control:
            for pd in model.pose_control_model.parameters():
                pd.requires_grad_(True)
            params += list(model.pose_control_model.parameters())
            print("Train controlnet layers in UNet")
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
    #if args.resume_dir is not None:
        ### regot the parameters information from the pretrained appearance model 
        
    optimizer = ZeroRedundancyOptimizer(
        params = params,
        lr=args.lr,
        optimizer_class=torch.optim.AdamW,
        weight_decay=args.weight_decay,
    )
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
            T.ToTensor(), 
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    if args.train_dataset == "full_head_clean_real_data_single_frame":
        dataset_cls = getattr(full_head_clean, args.train_dataset)
        test_dataset_cls = getattr(full_head_clean, args.train_dataset)
        root_path = args.dataset_root_path
        test_root_path = args.dataset_root_path + '_test'       
        image_dataset = dataset_cls(
                                image_transform=image_transform,
                                root_path= root_path,
                                more_image_control = args.more_image_control,
                                #mask_condition = args.mask_condition, 
                                )
        
        test_image_dataset = test_dataset_cls(
                                image_transform = image_transform,
                                root_path= root_path,
                                more_image_control = args.more_image_control,
                                #mask_condition = args.mask_condition, 
                                )
    #import pdb;pdb.set_trace()
    image_dataloader = DataLoader(image_dataset, 
                                    batch_size=args.train_batch_size,
                                    num_workers=args.num_workers,
                                    #shuffle=True,
                                    pin_memory=False)
    image_dataloader_iter = iter(image_dataloader)

    test_image_dataloader = DataLoader(test_image_dataset, 
                                  batch_size=args.val_batch_size,
                                  num_workers=args.num_workers,
                                  #shuffle=True,
                                  pin_memory=False)
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
        try:
            batch_data = next(image_dataloader_iter)
        except:
            image_dataloader_iter = iter(image_dataloader)
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

            x = infer_model.get_first_stage_encoding(infer_model.encode_first_stage(image))
            #pdb.set_trace()
            c_cat_list, cond_img_cat, more_cond_imgs  = get_cond_control(args, batch_data, args.control_type, args.device, model=infer_model, train=True, seg_model=None) # list o f[b,C,512,512]
            c_cross = infer_model.get_learned_conditioning(text)

            cond = {"c_concat": c_cat_list, "c_crossattn": [c_cross], "image_control": cond_img_cat}
            if more_cond_imgs is not None:
                cond.update({"more_image_control": [more_cond_imgs]})
            #import pdb;pdb.set_trace()
            if args.wonoise:
                cond['wonoise'] = True
            else:
                cond['wonoise'] = False

        # Foward
        model.train()
        with torch.cuda.amp.autocast(enabled=args.use_fp16, dtype=FP16_DTYPE):
            
            loss, loss_dict = model(x, cond)
            
            loss_list.append(loss.item())
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
        if first_print:
            print_peak_memory("Max memory allocated after FOWARD first step", args.local_rank)
        
        # Backward
        scaler.scale(loss).backward()
        local_step += 1
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
                nSample = min(args.train_batch_size, args.val_batch_size)
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