import os
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torchvision import transforms
from models.center_surround import resnet50_center_surround
from models.local_rf import resnet50_local_rf
from models.cortical_magnification import resnet50_cortical_magnification
from models.composite import resnet50_div_norm, resnet50_tuned_norm, resnet50_composite_model


def train(model, dataloader, criterion, optimizer, scaler, lr_scheduler, epoch, device='cpu', batch_verbosity=100):
    t0 = time.time()
    model.train()
    num_batches = len(dataloader)
    epoch_nsamples, epoch_loss, epoch_correct = 0, 0, 0
    prev_nsamples, prev_loss, prev_correct = 0, 0, 0
    curr_time = time.time()
    criterion.to(device)

    for batch_idx, (img_batch, label_batch) in enumerate(dataloader):
        img_batch = img_batch.to(device)
        label_batch = label_batch.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            output = model(img_batch)
            loss = criterion(output, label_batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_nsamples += len(img_batch)
        epoch_loss += (loss.item()*len(img_batch))
        epoch_correct += torch.sum(torch.argmax(output.detach(), dim=-1) == label_batch).item()

        if (batch_verbosity > 0) and (((batch_idx+1) % batch_verbosity) == 0):
            sub_epoch_samples = epoch_nsamples - prev_nsamples
            sub_epoch_avg_loss = (epoch_loss-prev_loss)/sub_epoch_samples
            sub_epoch_acc = (epoch_correct-prev_correct)/sub_epoch_samples
            prev_nsamples, prev_loss, prev_correct = epoch_nsamples, epoch_loss, epoch_correct
            sub_time = time.time()
            print('Epoch {} [{}-{}/{}] ({:.2f} s):\t loss: {:.4f}\t accuracy: {:.4f}'\
                .format(epoch, (batch_idx-batch_verbosity)+1, batch_idx+1, num_batches, sub_time-curr_time, sub_epoch_avg_loss, sub_epoch_acc))
            curr_time = sub_time

    curr_lr = optimizer.param_groups[0]['lr']
    epoch_avg_loss = epoch_loss/epoch_nsamples
    epoch_acc = epoch_correct/epoch_nsamples
    print('Epoch {} ({:.2f}s):\t lr: {}\t loss: {:.4f}\t accuracy: {:.4f}'\
            .format(epoch, time.time()-t0, curr_lr, epoch_avg_loss, epoch_acc))
    return epoch_avg_loss, epoch_acc


def evaluate(model, dataloader, criterion, device='cpu'):
    model.eval()
    criterion.to(device)
    running_nsamples, running_loss, running_correct = 0, 0, 0
    y_true, y_hat = [], []
    with torch.no_grad():
        for img_batch, label_batch in dataloader:
            img_batch = img_batch.to(device)
            label_batch = label_batch.to(device)
            output = model(img_batch)
            loss = criterion(output, label_batch)
            running_nsamples += len(img_batch)
            running_loss += (loss.item()*len(img_batch))
            predictions = torch.argmax(output, dim=-1)
            running_correct += torch.sum(predictions == label_batch).item()
            y_true.append(label_batch.cpu().numpy())
            y_hat.append(predictions.cpu().numpy())
    avg_loss = running_loss/running_nsamples
    acc = running_correct/running_nsamples
    print('-'*100)
    print('Evaluation:\t loss: {:.4f}\t accuracy: {:.4f}'.format(avg_loss, acc))
    print('-'*100)
    return np.hstack(y_true), np.hstack(y_hat), avg_loss, acc


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['imagenet', 'tinyimagenet'], type=str, required=True, \
                    help='Dataset for training/evaluation')
parser.add_argument('--data_root', type=str, required=True, \
                    help='Path to data directory')
parser.add_argument('--model_type', choices=['resnet50', 'resnet50_center_surround', 'resnet50_local_rf', \
                                             'resnet50_div_norm', 'resnet50_tuned_norm', 'resnet50_cortical_magnification', \
                                             'resnet50_composite_a', 'resnet50_composite_b', 'resnet50_composite_c', \
                                             'resnet50_composite_d', 'resnet50_composite_e', 'resnet50_composite_f', \
                                             'resnet50_composite_g'], \
                    type=str, required=True, help='Model architecture')
parser.add_argument('--model_path', type=str, required=False, \
                    help='Path to saved model file')
parser.add_argument('--output_root', type=str, required=True, \
                    help='Path to directory where artifacts will be saved')
parser.add_argument('--mode', choices=['train', 'validate', 'finetune'], type=str, required=True, \
                    help='Train model or run evaluation')
parser.add_argument('--num_epochs', default=100, type=int, required=False, \
                    help='Number of training epochs')
parser.add_argument('--batch_size', default=128, type=int, required=False, \
                    help='Data batch size')
parser.add_argument('--lr', default=0.1, type=float, required=False, \
                    help='Optimizer initial learning rate')
parser.add_argument('--lr_step_milestones', default=[60, 80], nargs='+', type=int, required=False, \
                    help='Epochs to trigger learning rate updates')
parser.add_argument('--gamma', default=0.1, type=float, required=False, \
                    help='Learning rate scaling')
parser.add_argument('--momentum', default=0.9, type=float, required=False, \
                    help='Optimizer momentum factor')
parser.add_argument('--weight_decay', default=1e-5, type=float, required=False, help='Weight decay')
parser.add_argument('--save_freq', default=10, type=int, required=False, \
                    help='Save model artifacts at ever save_frequency epochs')
parser.add_argument('--iteration_verbosity', default=100, type=int, required=False, \
                    help='During training, loss and accuracy will be reported every iteration_verbosity batches')
parser.add_argument('--num_workers', default=16, type=int, required=False, help='Dataloader num_workers')                  
parser.add_argument('--device', default='cpu', type=str, required=False, help='Device to use for training/testing')
args = parser.parse_args()


torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':    
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    train_transform_list = [
        transforms.RandomResizedCrop(224),
        transforms.Resize((64,64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ]
    valid_transform_list = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ]
    if args.dataset == 'tinyimagenet':
        train_transform_list = train_transform_list[2:]
        valid_transform_list = valid_transform_list[3:]
    train_transform = transforms.Compose(train_transform_list)
    valid_transform = transforms.Compose(valid_transform_list)
    train_dataset = ImageFolder(os.path.join(args.data_root, 'train'), transform=train_transform)
    valid_dataset = ImageFolder(os.path.join(args.data_root, 'val'), transform=valid_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

    if args.model_type == 'resnet50':
        model = resnet50()
    elif args.model_type == 'resnet50_center_surround':
        # Center Surround only
        model = resnet50_center_surround()
    elif args.model_type == 'resnet50_local_rf':
        # Local RF only
        model = resnet50_local_rf()
    elif args.model_type == 'resnet50_div_norm':
        # Divisive normalization only
        model = resnet50_div_norm()
    elif args.model_type == 'resnet50_tuned_norm':
        # Tuned normalization only
        model = resnet50_tuned_norm()
    elif args.model_type == 'resnet50_cortical_magnification':
        # Cortical Magnification only
        model = resnet50_cortical_magnification()
    elif args.model_type == 'resnet50_composite_a':
        # All components (Center Surround, Local RF, Tuned Norm., Cortical Mag.)
        model = resnet50_composite_model('a')
    elif args.model_type == 'resnet50_composite_b':
        # Local RF, Tuned Norm., Cortical Mag.
        model = resnet50_composite_model('b')
    elif args.model_type == 'resnet50_composite_c':
        # Center Surround, Local RF, Cortical Mag.
        model = resnet50_composite_model('c')
    elif args.model_type == 'resnet50_composite_d':
        # Center Surround, Local RF, Tuned Norm.
        model = resnet50_composite_model('d')
    elif args.model_type == 'resnet50_composite_e':
        # Tuned Norm., Cortical Mag.
        model = resnet50_composite_model('e')
    elif args.model_type == 'resnet50_composite_f':
        # Local RF, Cortical Mag.
        model = resnet50_composite_model('f')
    elif args.model_type == 'resnet50_composite_g':
        # Local RF, Tuned Norm.
        model = resnet50_composite_model('g')

    if args.model_path is not None:
        print('Loading model from checkpoint')
        checkpoint = torch.load(args.model_path, map_location='cpu')
        current_epoch = checkpoint['current_epoch']
        valid_loss = checkpoint['valid_loss']
        min_val_loss = checkpoint['valid_loss']
        max_val_acc = checkpoint['valid_acc']
        model.load_state_dict(checkpoint['model'])
    else:
        current_epoch = 0
        min_val_loss = np.inf
        max_val_acc = 0

    if args.dataset == 'tinyimagenet':
        # Change number of classes in output layer to 200
        if args.model_type == 'resnet50_cortical_magnification':
            model[-1] = torch.nn.Linear(2048, 200)
        else:
            model.fc = torch.nn.Linear(2048, 200)

    if (args.mode == 'finetune'):
        # Freeze all parameters except for final fully connected layer
        current_epoch = 0
        min_val_loss = np.inf
        max_val_acc = 0
        for pname, param in model.named_parameters():
            if args.model_type == 'resnet50_polar':
                if ((pname == '10.weight') or (pname == '10.bias')):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                if (('fc.weight' in pname) or ('fc.bias' in pname)):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    device = args.device
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_step_milestones, gamma=args.gamma, last_epoch=-1)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    if (args.mode == 'train') or (args.mode == 'finetune'):
        print('Training...')
        for epoch in range(current_epoch, args.num_epochs):
            train_loss, train_acc = train(model, train_dataloader, criterion, optimizer, scaler, \
                                          lr_scheduler, epoch, device, batch_verbosity=args.iteration_verbosity)
            y_true_valid, y_hat_valid, valid_loss, valid_acc = evaluate(model, valid_dataloader, criterion, device)
            lr_scheduler.step()
            if (args.output_root is not None):
                if (valid_acc >= max_val_acc):
                    max_val_acc = valid_acc
                    outpath = os.path.join(args.output_root, 'best_model_acc.pt')
                    torch.save({'args': vars(args),
                            'device': device,
                            'current_epoch': epoch,
                            'valid_loss': valid_loss,
                            'valid_acc': valid_acc,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict()},
                            outpath)
                if ((epoch+1)%args.save_freq == 0):
                    outpath = os.path.join(args.output_root, 'epoch_{}.pt'.format(epoch))
                    torch.save({'args': vars(args),
                                'device': device,
                                'current_epoch': epoch,
                                'valid_loss': valid_loss,
                                'valid_acc': valid_acc,
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict()},
                                outpath)

    elif args.mode == 'validate':
        y_true_valid, y_hat_valid, valid_loss, valid_acc = evaluate(model, valid_dataloader, criterion, device)