import argparse

from models.my_resnet import resnetX
from dataloaders.davis import DAVISAllSequence
from tester import Tester

import torch
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='PiWiVOS')

parser.add_argument('--checkpoint_path', type=str, default='checkpoints/piwivos/piwivos.pth', help='path to checkpoint')
parser.add_argument('-p', '--path', type=str, default='data/DAVIS', help='path to DAVIS dataset')
parser.add_argument('--model_name', type=str, choices=['piwivos', 'piwivosf'], default='piwivos')
parser.add_argument('--image_set', type=str, choices=['val', 'test-dev', 'test-challenge'], default='val')
parser.add_argument('--workers', type=int, default=0, help='number of workers to load data')
parser.add_argument('--export', action='store_true', help='export predicted masks')

print('Parsing arguments...')
args = parser.parse_args()
# Default piwivos parameters
args.output_stride = 8 if args.model_name == 'piwivos' else 16
args.arch = 'resnet50' if args.model_name == 'piwivos' else 'resnet34'
args.k = [1, 10]
args.lambd = [0.5, 0.5]


# ########## DATASETS AND DATALOADERS ##########
print('Preparing datasets...')
test_set = DAVISAllSequence(davis_root_dir=args.path, image_set=args.image_set)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

# ########## Net Init ##########
print('Network initialization...')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = resnetX(arch=args.arch,
                pretrained=True,
                output_stride=args.output_stride)

model = model.to(device)

print('Tester initialization...')
tester = Tester(device, model, test_loader, args)
tester.test()
