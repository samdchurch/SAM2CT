import torch
from sam2ct.build_sam2ct import build_sam2_volume_predictor
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_module

def process_image(image_path, prompt_data_path):
    pass


def load_model(log_dir, model_config, checkpoint=None):
    if checkpoint is None:
        ckpt_path = f'{log_dir}/checkpoints/checkpoint.pt'
    else:
        ckpt_path = f'{log_dir}/checkpoints/checkpoint_{checkpoint}.pt'
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    return build_sam2_volume_predictor(model_config, ckpt_path, device=device)

if __name__ == '__main__':
    GlobalHydra.instance().clear()
    initialize_config_module("sam2ct/config", version_base="1.2")
    model_config = 'SAM2CT'
    ckpt_path = 'sam2ct/checkpoints/SAM2CT.pt'
    device = torch.device('cuda')
    model = build_sam2_volume_predictor(model_config, ckpt_path, device=device)

