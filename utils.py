import os
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

cmap = plt.cm.viridis


def parse_command():
    data_names = ['nyu', 'uow_dataset']
    loss_names = ['l1', 'l2']
    model_names = ['mobilenet-skipconcat', 'mobilenet-skipadd', 'resnet18-skipadd',
                   'resnet18-skipconcat', 'resnet50-skipadd', 'resnet50-skipconcat']
    decoder_names = ['nnconv5dw', 'blconv5dw', 'upproj', 'shuffle', 'upconv', 'deconv']

    from dataloaders.dataloader import MyDataloader
    modality_names = MyDataloader.modality_names

    import argparse
    parser = argparse.ArgumentParser(description='FastDepth')
    parser.add_argument('--data', metavar='DATA', default='nyu',
                        choices=data_names,
                        help='dataset: ' + ' | '.join(data_names) + ' (default: nyu)')
    parser.add_argument('--arch', default='mobilenet-nnconv5dw-skipconcat', choices=model_names,
                        help='model architecture: ' + ' | '.join(
                            model_names) + ' (default: mobilenet-nnconv5dw-skipconcat)')
    parser.add_argument('--decoder', '-d', metavar='DECODER', default='deconv2', choices=decoder_names,
                        help='decoder: ' + ' | '.join(decoder_names) + ' (default: deconv2)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run (default: 15)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-c', '--criterion', metavar='LOSS', default='l1', choices=loss_names,
                        help='loss function: ' + ' | '.join(loss_names) + ' (default: l1)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate (default 0.01)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb', choices=modality_names,
                        help='modality: ' + ' | '.join(modality_names) + ' (default: rgb)')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('-b', '--batch-size', default=8, type=int, help='mini-batch size (default: 8)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH', )
    parser.add_argument('--gpu', default='0', type=str, metavar='N', help="gpu id")
    parser.add_argument('--predict', default=False, type=bool)
    parser.set_defaults(cuda=True)

    args = parser.parse_args()
    return args


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C


def adjust_learning_rate(optimizer, epoch, lr_init):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = lr_init * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch - 1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)


def merge_into_row(input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])

    return img_merge


def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    img_merge = np.hstack([rgb, depth_input_col, depth_target_col, depth_pred_col])

    return img_merge


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)


def get_output_directory(args):
    output_directory = os.path.join('results',
                                    '{}.arch={}.decoder={}.criterion={}.lr={}.epochs={}'.
                                    format(args.data, args.arch, args.decoder, args.criterion, args.lr, args.epochs))
    return output_directory
