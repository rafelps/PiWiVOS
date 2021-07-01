import argparse

from models.my_resnet import resnetX
from dataloaders.davis import DAVISAllSequence, DAVISCurrentFirstAndPrevious
from trainer import Trainer

import torch
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='PiWiVOS')

parser.add_argument('--job_name', type=str, required=True)
parser.add_argument('-p', '--path', type=str, default='data/DAVIS', help='path to DAVIS dataset')

method = parser.add_argument_group('method arguments')
method.add_argument('--model_name', type=str, choices=['piwivos', 'piwivosf'], default='piwivos', help='model to use')
method.add_argument('-k', nargs=2, metavar=('k_0', 'k_prev'), type=int, default=(1, 10), help='average of top k scores')
method.add_argument('-l', '--lambd', nargs=2, metavar=('lambda_0', 'lambda_prev'), type=float, default=(0.5, 0.5),
                    help='weight for distance penalization')
method.add_argument('-wl', '--weighted_loss', action='store_false', help='disable weighted CEL')
method.add_argument('-w0', '--weight0', type=float, default=1.0, help='weight for frame0 reference probabilities')

training = parser.add_argument_group('training arguments')
training.add_argument('-lr', '--learning_rate', type=float, default=5e-4, help='learning rate')
training.add_argument('-n', '--num_epochs', type=int, default=30, help='number of training epochs')
training.add_argument('-bs', '--batch_size', type=int, default=1, help='batch size for training')
training.add_argument('--workers', type=int, default=2, help='number of workers to load data')
training.add_argument('-wd', '--weight_decay', type=float, default=0, help='weight decay')
training.add_argument('--log_each', type=int, default=25, help='log each training batches')

print('Parsing arguments...')
args = parser.parse_args()
args.output_stride = 8 if args.model_name == 'piwivos' else 16
args.arch = 'resnet50' if args.model_name == 'piwivos' else 'resnet34'

# ########## DATASETS AND DATALOADERS ##########
print('Preparing datasets...')
train_set = DAVISCurrentFirstAndPrevious(davis_root_dir=args.path, image_set='train')
val_set = DAVISAllSequence(davis_root_dir=args.path, image_set='val')
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                          pin_memory=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

# ########## Net Init ##########
print('Network initialization...')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = resnetX(arch=args.arch,
                pretrained=True,
                output_stride=args.output_stride)

model = model.to(device)

print('Trainer initialization...')
trainer = Trainer(device, model, train_loader, val_loader, args)
trainer.train_model()
