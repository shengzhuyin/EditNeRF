import random

import torch
import os
import numpy as np

from rendering import render_path
from dataset import load_data
from inputs import config_parser
from model import create_nerf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)

def interpolate():
    INTERPO_COLOR = True
    
    parser = config_parser()
    args = parser.parse_args()

    images, poses, style, i_test, i_train, bds_dict, dataset, hwfs, near_fars, _ = load_data(args, num_images = 4)
    images_test, poses_test, style_test, hwfs_test, nf_test = images[i_test], poses[i_test], style[i_test], hwfs[i_test], near_fars[i_test]
    images_train, poses_train, style_train, hwfs_train, nf_train = images[i_train], poses[i_train], style[i_train], hwfs[i_train], near_fars[i_train]
    print(f"[DEBUG] {i_test = }, {style_test.shape = }, {style_test = }")
    print(f"[DEBUG] {style.shape = }, {style = }")

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

    np.save(os.path.join(basedir, expname, 'poses.npy'), poses_train.cpu())
    np.save(os.path.join(basedir, expname, 'hwfs.npy'), hwfs_train.cpu())

    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    
    print(f"[DEBUG] {poses_test.shape = }, {style_test.shape = }, {hwfs_test.shape = }, {nf_test.shape = }")
    
    # pick one pose
    n = poses_test.shape[0]
    # index = random.randint(0, n - 1)
    # color_idx = random.randint(0, n - 1)
    index, color_idx = 1, 0
    print(f"[DEBUG] {n = }, {index = }, {color_idx = }")
    N = 128
    poses_test = torch.stack([poses_test[index]] * N, dim = 0)
    hwfs_test = torch.stack([hwfs_test[index]] * N, dim = 0)
    nf_test = torch.stack([nf_test[index]] * N, dim = 0)
    
    def interop(style, second_style, t):
        assert t>=0 and t<=1
        if t>0.5:
            t = 1-t
        t = t*2
        return torch.lerp(input = style, end = second_style, weight = t)
    
    def interpolate_color(base_style, second_style, n):
        assert (base_style - second_style).norm() > 1e-5, f"Styles are equal"
        style_sigma = base_style[:args.style_dim//2].clone() # shape is fixed
        base_style_color = base_style[args.style_dim//2:].clone()
        base_second_style = second_style[args.style_dim//2:].clone()
        color_interpolations = [interop(base_style_color, base_second_style, i/n) for i in range(n)]
        color_interpolations = [torch.cat([style_sigma, i], dim = 0) for i in color_interpolations]
        color_interpolations = torch.stack(color_interpolations)
        return color_interpolations

    def interpolate_shape(base_style, second_style, n):
        assert (base_style - second_style).norm() > 1e-5, f"Styles are equal"
        style_color = base_style[args.style_dim//2:].clone() # color is fixed
        base_style_sigma = base_style[:args.style_dim//2].clone()
        base_second_style = second_style[:args.style_dim//2].clone()
        shape_interpolations = [interop(base_style_sigma, base_second_style, i/n) for i in range(n)]
        shape_interpolations = [torch.cat([i, style_color], dim = 0) for i in shape_interpolations]
        shape_interpolations = torch.stack(shape_interpolations)
        return shape_interpolations
    
    base_style = style_test[index]
    second_style = style_test[color_idx]
    assert base_style.shape == (args.style_dim, )

    if INTERPO_COLOR:
        style_test = interpolate_color(base_style, second_style, N)
        print(f"[DEBUG] {style_test[0] - style_test[1]}")
    else :
        style_test = interpolate_shape(base_style, second_style, N)
        print(f"[DEBUG] {style_test[0] - style_test[1]}")
    
    print(f"[DEBUG] {poses_test.shape = }, {style_test.shape = }, {hwfs_test.shape = }, {nf_test.shape = }")
        
    with torch.no_grad():
        # pick one poses_test, hwfs_test, nf_test
        # and interpolate between two styles (style_test), keep the shape same and interpolate colors.
        
        testsavedir = os.path.join(basedir, expname, 'interpolation_INTERPO_COLOR' if INTERPO_COLOR else "interpolation_INTERPO_SHAPE")
        os.makedirs(testsavedir, exist_ok=True)
        _, _, psnr = render_path(poses_test.to(device), style_test, hwfs_test, args.chunk, render_kwargs_test, nfs=nf_test, gt_imgs=None, savedir=testsavedir)
        print('Saved interpolation set w/ psnr', psnr)

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    interpolate()
