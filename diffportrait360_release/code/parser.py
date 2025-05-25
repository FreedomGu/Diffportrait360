import argparse
str2bool = lambda arg: bool(int(arg))

def get_parser():
    ## Model
    parser = argparse.ArgumentParser(description='Control Net training')
    parser.add_argument('--model_config', type=str, default="model_lib/ControlNet/models/cldm_v15_video_appearance.yaml",
                        help="The path of model config file")
    parser.add_argument('--reinit_hint_block', action='store_true', default=False,
                        help="Re-initialize hint blocks for channel mis-match")
    parser.add_argument('--image_size',  type =int, default=64)
    parser.add_argument('--empty_text_prob', type=float, default=0.1,
                    help="For cfg, probablity of replacing text to empty seq ")
    parser.add_argument('--sd_locked', type =str2bool, default=True,
                        help='Freeze parameters in original stable-diffusion decoder')
    parser.add_argument('--only_mid_control', type =str2bool, default=False,
                        help='Only control middle blocks')
    parser.add_argument('--finetune_attn', action='store_true', default=False,
                        help='Fine-tune all self-attention layers in UNet')
    parser.add_argument('--finetune_all', action='store_true', default=False,
                        help='Fine-tune all UNet and ControlNet parameters')
    parser.add_argument('--finetune_imagecond_unet', action='store_true', default=False,
                        help='Fine-tune all UNet and image ControlNet parameters')
    parser.add_argument('--finetune_control', action='store_true', default=False,
                        help='Fine-tune Control')
    parser.add_argument("--finetune_dual_attn", action='store_true', default=False, help='Fine-tune DualControl')
    parser.add_argument("--finetune_mmonly", action='store_true', default=False,
                        help = "help agumentation sequence control")
    parser.add_argument('--control_type', type=str, nargs="+", default=["pose"],
                        help='The type of conditioning')
    parser.add_argument('--control_dropout', type=float, default=0.0,
                        help='The probability of dropping out control inputs, only applied for multi-control')
    parser.add_argument('--depth_bg_threshold', type=float, default=0.0,
                        help='The threshold of cutting off depth')
    parser.add_argument("--control_mode", type=str, default="balance",
                        help="Set controlnet is more important or balance.")
    parser.add_argument('--wonoise', action='store_true', default=False,
                        help='Use with referenceonly, remove adding noise on reference image')
    ## Training
    parser.add_argument('--num_workers', type = int, default = 4,
                        help='total number of workers for dataloaders')
    parser.add_argument('--train_batch_size', type = int, default = 1,
                        help='batch size for each gpu in distributed training')
    parser.add_argument('--val_batch_size', type = int, default = 1,
                        help='batch size for each gpu during inference(must be set to 1)')
    parser.add_argument('--lr', type = float, default = 1e-5,
                        help='learning rate of new params in control net')
    parser.add_argument('--lr_sd', type = float, default = 1e-5,
                        help='learning rate of params from original stable diffusion modules')
    parser.add_argument('--weight_decay', type = float, default = 0,
                        help='weight decay (L2) regularization')
    parser.add_argument('--lr_anneal_steps', type = float, default = 0,
                        help='steps for learning rate annealing')
    parser.add_argument('--ema_rate', type = float, default = 0,
                        help='rate for ema')
    parser.add_argument('--num_train_steps', type = int, default = 1000000,
                        help='number of train steps')
    parser.add_argument('--grad_clip_norm', type = float, default = 0.5,
                        help='grad_clip_norm')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--seed', type=int, default=42, 
                        help='random seed for initialization')
    parser.add_argument("--logging_steps", type=int, default=10000, help="Log every X updates steps.")
    parser.add_argument("--logging_gen_steps", type=int, default=1000, help="Log Generated Image every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=10000, help=" 1000 Save checkpoint every X updates steps.")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default")
    parser.add_argument('--use_fp16', action='store_true', default=False,
                        help='Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit')
    parser.add_argument('--global_step', type=int, default=0,
                        help='initial global step to start with (use with --init_path)')
    parser.add_argument('--load_optimizer_state', action='store_true', default=False,
                        help='Whether to restore optimizer when resuming')
    parser.add_argument('--compile', type=str2bool, default=False,
                        help='compile model (for torch 2)')
    parser.add_argument('--with_text', action='store_false', default=True,
                        help='Feed text_blip into the model')
    parser.add_argument('--text_prompt', type=str, default=None,
                        help='Feed text_prompt into the model')
    ## Data
    parser.add_argument('--dataset_root_path', type=str, default=None, help='datapath to ')
    
    parser.add_argument('--mask_bg', action='store_true', default=False,
                        help='Mask the background of image to pure black')
    parser.add_argument('--random_mask', action='store_true', default=False,
                        help='Randomly mask reference image. Similar to inpaint.')
    parser.add_argument('--mask_condition', action='store_true', default=False,
                        help='mask condition for debugging appearance_control_model only')
    parser.add_argument("--train_dataset", type=str, default="pano_head",
                        help="The dataset class for training.")
    parser.add_argument("--img_bin_limit", type = int, default = 29,
                        help="The upper limit while loading image from a sequence.")
    parser.add_argument("--output_dir", type=str, default=None, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--local_log_dir", type=str, default=None, required=True,
                        help="The local output directory where tensorboard will be written.")
    parser.add_argument("--local_image_dir", type=str, default=None, required=True,
                        help="The local output directory where generated images will be written.")
    parser.add_argument("--resume_dir", type=str, default=None,
                        help="The resume directory where the model checkpoints will be loaded.")
    parser.add_argument("--image_pretrain_dir", type=str, default=None,
                        help="The resume directory where the appearance control model checkpoints will be loaded.")
    parser.add_argument("--pose_pretrain_dir", type=str, default=None,
                        help="The resume directory where the pose control model checkpoints will be loaded.")
    parser.add_argument("--mm_path_dir", type=str, default=None,
                        help="The resume directory where the motion model checkpoints will be loaded.")
    parser.add_argument("--init_path", type=str, default="//",
                        help="The resume directory where the model checkpoints will be loaded.")
    parser.add_argument("--pose_augmentation", action='store_true', default=False,
                        help = "help agumentation appearance_control_model pose control")
    parser.add_argument("--cond_by_camera", action='store_true', default=False,
                        help = "help agumentation appearance_control_model pose control")
    parser.add_argument("--appearance_dir", default=None, type=str, help="Resume of the appearance and replace it ")
    ## Segment Model ar
    parser.add_argument("--more_image_control", action='store_true', default=False, help='if is a moreImageControl Pipeline')
    parser.add_argument('--finetune_mmonlyattn', action='store_true', default=False,
                        help = "help agumentation sequence control")
    parser.add_argument('--finetune_mmonlypose', action='store_true', default=False,)
    args = parser.parse_args()
    ### Segment Model args
    
    return args

