import json
import argparse
import torch

import torch.nn.functional as F
from torchvision.transforms import Normalize

import nibabel as nib

from sam2ct.build_sam2ct import build_sam2_volume_predictor
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_module

def parse_args():
    parser = argparse.ArgumentParser(description='SAM2CT Volume Inference')
    
    parser.add_argument('--checkpoint', type=str, default='sam2ct/checkpoints/SAM2CT.pt', help='Path to model checkpoint')
    parser.add_argument('--model_config', type=str, default='SAM2CT', help='Path to model config file')
    parser.add_argument('--image_path', type=str, default='examples/example_image.nii.gz', help='Path to input image')
    parser.add_argument('--gt_path', type=str, default='examples/example_label.nii.gz', help='Path to ground truth mask')
    parser.add_argument('--prompt_path', type=str, default='examples/line_prompt_example.json', help='Path to prompt info')
    parser.add_argument('--input_size', type=int, default=1024, help='Input size of images')
    parser.add_argument('--save_path', type=str, default='data/output/prediction.nii.gz', help='Path to save prediction')
    return parser.parse_args()

def dice_score(pred, target, eps=1e-6):
    # Ensure binary (0 or 1)
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    # Flatten full 3D volume
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    dice = (2. * intersection + eps) / (pred.sum() + target.sum() + eps)

    return dice.item()

def load_and_preprocess_image(image_path, image_size=1024, min_val=-500, max_val=500, label=False):
    image_data = nib.load(image_path).get_fdata()
    image_data = torch.tensor(image_data).float()
    if not label:
        image_data = torch.clamp(image_data, min_val, max_val)
        image_data = image_data - torch.min(image_data)
        image_data = image_data / torch.max(image_data)
    image_data = image_data.unsqueeze(0).permute(3, 0, 1, 2)
    image_data = F.interpolate(image_data, size=(image_size, image_size), mode='bilinear', align_corners=False)
    image_data = image_data.float()

    if not label:
        image_data = image_data.repeat(1, 3, 1, 1)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = Normalize(mean, std)
        image_data = normalize(image_data)

    return image_data




def process_image(model, image_path, gt_path, prompt_data_path, image_size=1024, scale=2):
    image_data = load_and_preprocess_image(image_path, image_size=image_size)
    gt_data = load_and_preprocess_image(gt_path, image_size=image_size, label=True)

    image_data = image_data.to(model.device)
    with open(prompt_data_path) as f:
        prompt_info = json.load(f)
    prompt_slice = prompt_info['slice']
    prompt_type = prompt_info['category']
    points = prompt_info['points']

    if prompt_type == 'line':
        point_coords = scale*torch.tensor(points).unsqueeze(0)
        point_coords = point_coords.to('cuda')
        labels = torch.tensor([4, 4], dtype=torch.int, device=point_coords.device).unsqueeze(0)
        point_prompt = {"point_coords": point_coords, "point_labels": labels}
    elif prompt_type == 'major_minor':
        # point_coords = torch.tensor([[points[0], points[1]], [points[2], points[3]],
        #                            [points[6], points[7]], [points[8], points[9]]]).unsqueeze(0)
        point_coords = torch.tensor([[points[1], points[0]], [points[3], points[2]],
                                    [points[7], points[6]], [points[9], points[8]]]).unsqueeze(0)
        point_coords = point_coords.to('cuda')
        labels = torch.tensor([4, 4, 4, 4], dtype=torch.int, device=point_coords.device).unsqueeze(0)
        point_prompt = {"point_coords": point_coords, "point_labels": labels}
    elif prompt_type == 'arrow':
        arrow_tail = points[:2]
        arrow_head = points[2:4]
        
        new_point = point_away_from_head(arrow_head[0], arrow_head[1], arrow_tail[0], arrow_tail[1])
        #point_coords = torch.tensor([[arrow_head[0], arrow_head[1]], [new_point[0], new_point[1]]]).unsqueeze(0)

        point_coords = torch.tensor([[arrow_head[1], arrow_head[0]], [new_point[1], new_point[0]]]).unsqueeze(0)
        point_coords = point_coords.to('cuda')
        labels = torch.tensor([6, 5], dtype=torch.int, device=point_coords.device).unsqueeze(0)
        point_prompt = {"point_coords": point_coords, "point_labels": labels}

    pred_volume = model.create_prediction(prompt_slice=prompt_slice,
        image_data=image_data, device=image_data.device,
        mask_prompt=None, point_prompt=point_prompt)
    print(torch.sum(pred_volume))
    print(torch.sum(gt_data))
    dice = dice_score(pred_volume, gt_data)
    print(dice)
    pass

if __name__ == '__main__':
    args = parse_args()

    GlobalHydra.instance().clear()
    initialize_config_module("sam2ct/config", version_base="1.2")

    device = torch.device('cuda')
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    model = build_sam2_volume_predictor(args.model_config, args.checkpoint, device=device)
    process_image(model, args.image_path, args.gt_path, args.prompt_path)