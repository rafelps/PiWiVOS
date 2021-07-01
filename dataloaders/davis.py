import os

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class DAVISCurrentFirstAndPrevious(Dataset):

    """
    Loads triplets of frame_0, frame_t-1, frame_t;
    Resizes frames so they all have same dimensions
    Used for training as shuffles all frames
    Returns frames FloatTensor(n_frames=3, ch=3, h, w)
    Returns masks LongTensor(n_frames=3, ch=1, h, w)
    Returns info dict
    """

    def __init__(self,
                 davis_root_dir,
                 image_set,
                 resize=(480, 864),
                 multi_object=True,
                 davis_year=2017,
                 resolution='480p',
                 transform=None):

        assert resolution == '480p', 'Other resolutions than 480p not implemented yet in dataloader'
        self.MO = multi_object
        self.image_dir = os.path.join(davis_root_dir, 'JPEGImages', resolution)
        self.mask_dir = os.path.join(davis_root_dir, 'Annotations', resolution)
        image_set_file = os.path.join(davis_root_dir, 'ImageSets', str(davis_year), f'{image_set}.txt')
        self.transform = transform
        self.resize = resize

        self.frames = []  # Store all the frames to process in format (video, frame)

        with open(os.path.join(image_set_file), 'r') as lines:
            for line in lines:
                video = line.strip()
                n_frames = len(os.listdir(os.path.join(self.image_dir, video)))

                for frame in range(1, n_frames):  # Avoid frame_0 as it will be loaded as frame_0 or frame_t-1
                    self.frames.append((video, frame))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        video, frame = self.frames[index]
        # assert len(os.listdir(os.path.join(self.mask_dir, video))) > 1, 'No ground truth for training dataset'

        frames = torch.FloatTensor(3, 3, self.resize[0], self.resize[1])  # (frame_0 frame_t-1 frame_t, RGB, H, W)
        masks = torch.LongTensor(3, 1, self.resize[0], self.resize[1])    # (frame_0 frame_t-1 frame_t, Palette, H, W)

        frame_0_file = os.path.join(self.image_dir, video, f'{0:05d}.jpg')
        frame_prev_file = os.path.join(self.image_dir, video, f'{frame-1:05d}.jpg')
        frame_t_file = os.path.join(self.image_dir, video, f'{frame:05d}.jpg')
        frame_0_img = Image.open(frame_0_file).convert('RGB')
        frame_prev_img = Image.open(frame_prev_file).convert('RGB')
        frame_t_img = Image.open(frame_t_file).convert('RGB')

        mask_0_file = os.path.join(self.mask_dir, video, f'{0:05d}.png')
        mask_prev_file = os.path.join(self.mask_dir, video, f'{frame-1:05d}.png')
        mask_t_file = os.path.join(self.mask_dir, video, f'{frame:05d}.png')
        mask_0_img = Image.open(mask_0_file).convert('P')
        mask_prev_img = Image.open(mask_prev_file).convert('P')
        mask_t_img = Image.open(mask_t_file).convert('P')

        if self.transform is not None:
            frame_0_img = self.transform(frame_0_img)
            frame_prev_img = self.transform(frame_prev_img)
            frame_t_img = self.transform(frame_t_img)
            mask_0_img = self.transform(mask_0_img)
            mask_prev_img = self.transform(mask_prev_img)
            mask_t_img = self.transform(mask_t_img)

        if frame_0_img.size != (self.resize[1], self.resize[0]):
            frame_0_img = frame_0_img.resize((self.resize[1], self.resize[0]), Image.LANCZOS)  # AntiAliasing Filter
            frame_prev_img = frame_prev_img.resize((self.resize[1], self.resize[0]), Image.LANCZOS)
            frame_t_img = frame_t_img.resize((self.resize[1], self.resize[0]), Image.LANCZOS)
            mask_0_img = mask_0_img.resize((self.resize[1], self.resize[0]))  # NN Filter as we are in palette mode
            mask_prev_img = mask_prev_img.resize((self.resize[1], self.resize[0]))
            mask_t_img = mask_t_img.resize((self.resize[1], self.resize[0]))

        frame_0_np = np.array(frame_0_img).transpose((2, 0, 1))
        frame_prev_np = np.array(frame_prev_img).transpose((2, 0, 1))
        frame_t_np = np.array(frame_t_img).transpose((2, 0, 1))
        frame_0_np = frame_0_np / 255.  # RGB rescaled [0, 1]
        frame_prev_np = frame_prev_np / 255.
        frame_t_np = frame_t_np / 255.

        frames[0] = torch.from_numpy(frame_0_np)
        frames[1] = torch.from_numpy(frame_prev_np)
        frames[2] = torch.from_numpy(frame_t_np)
        masks[0] = torch.from_numpy(np.array(mask_0_img))
        masks[1] = torch.from_numpy(np.array(mask_prev_img))
        masks[2] = torch.from_numpy(np.array(mask_t_img))

        n_objects = np.max(np.array(mask_0_img)) + 1  # Including background

        if not self.MO:
            masks = (masks != 0).long()  # Collapse all non-zero (different from background) into same object
            n_objects = 2

        # Correct bug in dataset (appears new object to segment in frame != 0 with index 255)
        if video == 'tennis':
            masks = (masks != 255).long()*masks

        info = {
            'name': video,
            'frame': frame,
            'n_objects': n_objects
        }

        return frames, masks, info


class DAVISAllSequence(Dataset):

    """
    Loads one entire sequence of frames
    Used for validating and testing
    Returns frames FloatTensor(n_frames, ch=3, h, w)
    Returns masks LongTensor(n_frames, ch=1, h, w)
    Returns info dict
    """

    def __init__(self,
                 davis_root_dir,
                 image_set,
                 resize=(480, 864),
                 multi_object=True,
                 davis_year=2017,
                 resolution='480p',
                 transform=None):

        assert resolution == '480p', 'Other resolutions than 480p not implemented yet in dataloader'
        self.MO = multi_object
        self.image_dir = os.path.join(davis_root_dir, 'JPEGImages', resolution)
        self.mask_dir = os.path.join(davis_root_dir, 'Annotations', resolution)
        image_set_file = os.path.join(davis_root_dir, 'ImageSets', str(davis_year), f'{image_set}.txt')
        self.transform = transform
        self.target_size = resize

        self.videos = []

        with open(image_set_file, 'r') as lines:
            for line in lines:
                video = line.strip()
                self.videos.append(video)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        n_frames = len(os.listdir(os.path.join(self.image_dir, video)))
        has_gt = len(os.listdir(os.path.join(self.mask_dir, video))) > 1
        n_objects = palette = original_shape = None

        if self.target_size is None:
            size = Image.open(os.path.join(self.mask_dir, video, '00000.png')).size
            self.target_size = size[1], size[0]

        frames = torch.FloatTensor(n_frames, 3, self.target_size[0], self.target_size[1])
        if has_gt:
            masks = torch.LongTensor(n_frames, 1, self.target_size[0], self.target_size[1])
        else:
            masks = torch.LongTensor(1, 1, self.target_size[0], self.target_size[1])

        for frame in range(n_frames):
            # Frame loader
            image_file = os.path.join(self.image_dir, video, f'{frame:05d}.jpg')
            img = Image.open(image_file).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            if img.size != (self.target_size[1], self.target_size[0]):
                img = img.resize((self.target_size[1], self.target_size[0]), Image.LANCZOS)  # AntiAliasing Filter

            img_np = np.array(img).transpose((2, 0, 1))
            img_np = img_np / 255.  # RGB rescaled [0, 1]

            frames[frame] = torch.from_numpy(img_np)

            # Mask loader
            if frame == 0 or has_gt:
                mask_file = os.path.join(self.mask_dir, video, f'{frame:05d}.png')
                mask_img = Image.open(mask_file).convert('P')

                if mask_img.size != (self.target_size[1], self.target_size[0]):
                    mask_img_resized = mask_img.resize((self.target_size[1], self.target_size[0]))  # NN Filter
                else:
                    mask_img_resized = mask_img

                mask_np = np.array(mask_img_resized)

                # Loading Palette || Maximum number of objects only in frame 0
                if frame == 0:
                    palette = mask_img.getpalette()
                    n_objects = np.max(mask_np) + 1  # Including background
                    original_shape = mask_img.size

                mask = torch.from_numpy(mask_np)

                if not self.MO:
                    mask = (mask != 0).long()
                    n_objects = 2

                masks[frame][0] = mask

        # Correct bug in dataset (appears new object to segment in frame != 0 with index 255)
        if video == 'tennis':
            masks = (masks != 255).long()*masks

        info = {
            'name': video,
            'n_frames': n_frames,
            'n_objects': n_objects,
            'original_shape': original_shape,
            'has_gt': has_gt,
            'palette': torch.ByteTensor(palette)
        }

        # assert torch.max(masks).item() + 1 == n_objects, 'Error preprocessing data'

        return frames, masks, info
