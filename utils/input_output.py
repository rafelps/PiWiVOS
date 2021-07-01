import os

import torch
from torchvision.transforms import ToPILImage


def save_mask_test(masks, seq_name, frame, palette, checkpoint_dir, image_set):
    base_dir = os.path.dirname(checkpoint_dir)
    res_dir = os.path.join(base_dir, 'results', image_set, seq_name)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    to_pil = ToPILImage()

    for mask in masks:
        mask = mask.type(torch.ByteTensor).cpu()
        mask_img = to_pil(mask)
        mask_img.putpalette(palette)
        mask_img.save(os.path.join(res_dir, "{:05d}.png".format(frame)))
        frame += 1


def save_model(model_state_dict, job_name):
    file = f'{job_name}.pth'
    path = os.path.join('logs', job_name)
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model_state_dict, os.path.join(path, file))
    return


def load_model(checkpoint_path):
    assert os.path.isfile(checkpoint_path), 'Checkpoint not existing'
    model_state_dict = torch.load(checkpoint_path)
    return model_state_dict
