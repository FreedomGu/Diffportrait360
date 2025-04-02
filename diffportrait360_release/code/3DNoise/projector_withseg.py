""" Projecting input images into latent spaces. """

import copy
import os
from time import perf_counter
import shutil
import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import mrcfile
import json
import dnnlib
import legacy
import scipy.interpolate
from camera_utils import LookAtPoseSampler
def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img
def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size


def project(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    c: torch.Tensor,
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    optimize_noise             = False,
    verbose                    = False,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    camera_lookat_point = torch.tensor([0, 0, 0.0], device=device)
    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=2.7, device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c_samples = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), c_samples.repeat(w_avg_samples,1))  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # fix delta_c

    delta_c = G.t_mapping(torch.from_numpy(np.mean(z_samples, axis=0, keepdims=True)).to(device), c[:1], truncation_psi=1.0, truncation_cutoff=None, update_emas=False)
    delta_c = torch.squeeze(delta_c, 1)
    c[:,3] += delta_c[:,0]
    c[:,7] += delta_c[:,1]
    c[:,11] += delta_c[:,2]

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32) / 255.0 * 2 - 1
    target_images_perc = (target_images + 1) * (255/2)
    if target_images_perc.shape[2] > 256:
        target_images_perc = F.interpolate(target_images_perc, size=(256, 256), mode='area')
    target_features = vgg16(target_images_perc, resize_images=False, return_lpips=True)

    w_avg = torch.tensor(w_avg, dtype=torch.float32, device=device).repeat(1, G.backbone.mapping.num_ws, 1)
    w_opt = w_avg.detach().clone()
    w_opt.requires_grad = True
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device="cpu")
    if optimize_noise:
        optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
    else:
        optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    if optimize_noise:
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = w_opt + w_noise
        synth_images = G.synthesis(ws, c=c, noise_mode='const')['image']

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images_perc = (synth_images + 1) * (255/2)
        if synth_images_perc.shape[2] > 256:
            synth_images_perc = F.interpolate(synth_images_perc, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images_perc, resize_images=False, return_lpips=True)
        perc_loss = (target_features - synth_features).square().sum(1).mean()

        mse_loss = (target_images - synth_images).square().mean()

        w_norm_loss = (w_opt-w_avg).square().mean()

        # Noise regularization.
        reg_loss = 0.0
        if optimize_noise:
            for v in noise_bufs.values():
                noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
        loss = 0.1 * mse_loss + perc_loss + 1.0 * w_norm_loss +  reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step: {step+1:>4d}/{num_steps} mse: {mse_loss:<4.2f} perc: {perc_loss:<4.2f} w_norm: {w_norm_loss:<4.2f}  noise: {float(reg_loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach().cpu()[0]

        # Normalize noise.
        if optimize_noise:
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

    if w_out.shape[1] == 1:
        w_out = w_out.repeat([1, G.mapping.num_ws, 1])

    return w_out, c


def project_pti(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    w_pivot: torch.Tensor,
    c: torch.Tensor,
    *,
    num_steps                  = 1000,
    initial_learning_rate      = 3e-4,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    verbose                    = False,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).train().requires_grad_(True).to(device) # type: ignore

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32) / 255.0 * 2 - 1
    target_images_perc = (target_images + 1) * (255/2)
    if target_images_perc.shape[2] > 256:
        target_images_perc = F.interpolate(target_images_perc, size=(256, 256), mode='area')
    target_features = vgg16(target_images_perc, resize_images=False, return_lpips=True)

    w_pivot = w_pivot.to(device).detach()
    optimizer = torch.optim.Adam(G.parameters(), betas=(0.9, 0.999), lr=initial_learning_rate)

    out_params = []

    for step in range(num_steps):
        # Learning rate schedule.
        # t = step / num_steps
        # lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        # lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        # lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        # lr = initial_learning_rate * lr_ramp
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr

        # Synth images from opt_w.
        synth_images = G.synthesis(w_pivot, c=c, noise_mode='const')['image']

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images_perc = (synth_images + 1) * (255/2)
        if synth_images_perc.shape[2] > 256:
            synth_images_perc = F.interpolate(synth_images_perc, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images_perc, resize_images=False, return_lpips=True)
        perc_loss = (target_features - synth_features).square().sum(1).mean()

        mse_loss = (target_images - synth_images).square().mean()

        loss = 0.1 * mse_loss + perc_loss

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step: {step+1:>4d}/{num_steps} mse: {mse_loss:<4.2f} perc: {perc_loss:<4.2f}')

        if step == num_steps - 1 or step % 10 == 0:
            out_params.append(copy.deepcopy(G).eval().requires_grad_(False).cpu())

    return out_params

#----------------------------------------------------------------------------

def run_projection(
    network_pkl: str,
    # target_fname: str,
    target_img:str,
    outdir: str,
    seed: int,
    num_steps: int,
    num_steps_pti: int,
    fps: int,
    target_c : list
):
    """Project given image to the latent space of pretrained network pickle.

    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Render debug output: optional video and projected image and W vector.
    #import pdb;pdb.set_trace()
    outdir = os.path.join(outdir, str(target_img.split("/")[-1].split(".")[0]))
    os.makedirs(outdir, exist_ok=True)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        network_data = legacy.load_network_pkl(fp)
        G = network_data['G_ema'].requires_grad_(False).to(device) # type: ignore
    
    G.rendering_kwargs["ray_start"] = 2.35
    if target_img is not None:
        target_fname = target_img
        c = target_c
        #torch.Tensor(c).to(device)

    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps, c = project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        c=c,
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')
    G_steps = project_pti(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        w_pivot=projected_w_steps[-1:],
        c=c,
        num_steps=num_steps_pti,
        device=device,
        verbose=True
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    G = G_steps[-1].to(device)
    # delete G steps
    #import pdb;pdb.set_trace()
    del G_steps
    #if save_video:
        # Interpolation.
    # Interpolation.
    grid = []
    grid_h = 1
    grid_w = 1
    wraps = 2 
    psi = 1.0
    #ws  = projected_w
    ws = projected_w_steps[-1]
    if len(ws) % (grid_w*grid_h) != 0:
        raise ValueError('Number of input seeds must be divisible by grid W*H')
    num_keyframes = len(ws) // (grid_w*grid_h)
    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(- wraps, (wraps + 1))
            y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
            interp = scipy.interpolate.interp1d(x, y, kind='cubic', axis=0)
            row.append(interp)
        grid.append(row)

    



    video = imageio.get_writer(f'{outdir}/condition32.mp4', mode='I', fps=fps, codec='libx264', bitrate='16M')
    pitch_range = 0.8
    w_frames= 32
    
    camera_lookat_point = torch.tensor([0, 0, 0], device=device)
    all_poses = []
    imgs = []
    for frame_idx in tqdm(range(w_frames)):
        imgs = []
        for yi in range(grid_h):
            for xi in range(grid_w):
                cam2world_pose = LookAtPoseSampler.sample(3.14/2 + 2 * 3.14 * frame_idx / w_frames,  3.14/2,
                                                                    camera_lookat_point, radius=2.7, device=device) 
                all_poses.append(cam2world_pose.squeeze().cpu().numpy())
                intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
                c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                interp = grid[yi][xi]
                wcd = torch.from_numpy(interp(frame_idx / w_frames)).to(device)
                #import pdb;pdb.set_trace()
                img = G.synthesis(ws=wcd.unsqueeze(0), c=c[0:1], noise_mode='const')['image'][0]  
                imgs.append(img)  
        video.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
    video.close()

    # Save final projected frame and W vector.
    # target_pil.save(f'{outdir}/target.png')
    # projected_w = projected_w_steps[-1]
    # G_final = G_steps[-1].to(device)
    # synth_image = G_final.synthesis(projected_w.unsqueeze(0).to(device), c=c, noise_mode='const')['image']
    # synth_image = (synth_image + 1) * (255/2)
    # synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    # PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    # np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

    # with open(f'{outdir}/fintuned_generator.pkl', 'wb') as f:
    #     network_data["G_ema"] = G_final.eval().requires_grad_(False).cpu()
    #     pickle.dump(network_data, f)
#----------------------------------------------------------------------------
import argparse
if __name__ == "__main__":

        parser = argparse.ArgumentParser(description='Project image to latent space')
        parser.add_argument('--network', type=str, required=True, help='Network pickle filename')
        parser.add_argument('--target_img_path', type=str, required=True, help='Target image folder')
        parser.add_argument('--target_seg', type=str, required=False, help='Target segmentation folder')
        parser.add_argument('--num_steps', type=int, default=50, help='Number of optimization steps')
        parser.add_argument('--num_steps_pti', type=int, default=5, help='Number of optimization steps for pivot tuning')
        parser.add_argument('--seed', type=int, default=666, help='Random seed')
        parser.add_argument('--outdir', type=str, required=True, help='Where to save the output images')
        parser.add_argument('--fps', type=int, default=30, help='Frames per second of final video')
        parser.add_argument('--camera_json', type=str, default=None, help='Camera json file')
        args = parser.parse_args()
        id_list = []
        print(args.target_img_path)
        target_path = args.target_img_path
        f = open(args.camera_json)
        camera_data = json.load(f)['labels']
        for ids in sorted(os.listdir(target_path)):
            if ids.endswith('.png') or ids.endswith('.jpg'):
                id_list.append(ids)
        dics = []
        #image_path = '/mnt/turtle/yuming/RenrderME/Final_Inference/Final_inthewild_data/Front_Face'
        #save_path = '/mnt/turtle/yuming/RenrderME/Final_Inference/Final_inthewild_PH_format_aligned/Generated_Face'
        #if not os.path.exists(save_path):
        #    os.makedirs(save_path, exist_ok=True)
        camera_dict = {item[0]: item[1] for item in camera_data}
        #@import pdb;pdb.set_trace()
        for i in range (0, len(id_list)):
            target_img_path = os.path.join(target_path, id_list[i])
            #camera_dict
            c = camera_dict[id_list[i].split('.')[0]+'.png']    
            c = torch.Tensor(np.array(c,dtype='float32')).view(1,-1).cuda()
            run_projection(network_pkl=args.network,
                           target_img = target_img_path,
                           outdir=args.outdir,
                           num_steps=args.num_steps,
                           num_steps_pti=args.num_steps_pti,
                           fps=args.fps,
                           seed = args.seed,
                           target_c = c) # pylint: disable=no-value-for-parameter
        # write done this dic list
#----------------------------------------------------------------------------