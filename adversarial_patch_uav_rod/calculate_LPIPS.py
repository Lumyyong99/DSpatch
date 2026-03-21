import cv2
import glob
import numpy as np
import os.path as osp
import argparse
import torch
import pyiqa


def main(args):
    folder_gt =args.gt
    folder_restored =args.sr
    img_format=args.format

    prefix=None
    lpips_score = []
    ssim_score = []
    psnr_score = []

    lpips_score = []

    img_list = sorted(glob.glob(osp.join(folder_gt, '*')))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    lpips_metric = pyiqa.create_metric('lpips', device=device)
    psnr_metric = pyiqa.create_metric('psnr', test_y_channel=True, crop_border=4, color_space='ycbcr')
    ssim_metric = pyiqa.create_metric('ssim', test_y_channel=True, crop_border=4, color_space='ycbcr')

    fr_iqa_dict={}

    for i, img_path in enumerate(img_list):
        basename, ext = osp.splitext(osp.basename(img_path))
        if prefix is None:
            img_sr = osp.join(folder_restored, basename + img_format)
            ssim_score.append(ssim_metric(img_sr,img_path))
            psnr_score.append(psnr_metric(img_sr,img_path))
            lpips_score.append(lpips_metric(img_sr,img_path))
    fr_iqa_dict['psnr']=sum(psnr_score).item() / len(psnr_score)
    fr_iqa_dict['ssim']=sum(ssim_score).item() / len(ssim_score)
    fr_iqa_dict['lpips']=sum(lpips_score).item() / len(lpips_score)

    print(fr_iqa_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='/home/SR_iSAID/val-full/HR/', help='Path to gt (Ground-Truth)')
    parser.add_argument('--sr', type=str, default='results/Set14', help='Path to restored images')
    parser.add_argument('--crop_border', type=int, default=4, help='Crop border for each side')
    parser.add_argument('--prefix', type=str, default=None, help='Suffix for restored images')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for restored images')
    parser.add_argument('--format', type=str, default='.png', help='Suffix for restored images')
    parser.add_argument(
        '--test_y_channel',
        action='store_true',
        help='If True, test Y channel (In MatLab YCbCr format). If False, test RGB channels.')
    parser.add_argument('--correct_mean_var', action='store_true', help='Correct the mean and var of restored images.')
    args = parser.parse_args()
    
    main(args)