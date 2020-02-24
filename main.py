import os
import time
import csv
import numpy as np
import criteria
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.utils.data.dataloader
#import nonechucks as nc

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True

import models
from metrics import AverageMeter, Result
import utils
#from torch.utils.data._utils.collate import default_collate

def my_collate(batch):
    len_batch = len(batch) # original batch length
    batch = list(filter (lambda x:x is not None, batch)) # filter out all the Nones
    if len_batch > len(batch): # if there are samples missing just use existing members, doesn't work if you reject every sample in a batch
        diff = len_batch - len(batch)
        for i in range(diff):
            batch = batch + batch[:diff]
    return torch.utils.data.dataloader.default_collate(batch)

args = utils.parse_command()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu # Set the GPU.

fieldnames = ['rmse', 'mae', 'delta1', 'absrel',
            'lg10', 'mse', 'delta2', 'delta3', 'data_time', 'gpu_time']
best_fieldnames = ['best_epoch'] + fieldnames
best_result = Result()
best_result.set_to_worst()

def main():
    global args, best_result, output_directory, train_csv, test_csv
    # Random seed setting
    torch.manual_seed(16)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Data loading code
    print("=> creating data loaders...")
    data_dir = '/media/vasp/Data2/Users/vmhp806/depth-estimation'
    valdir = os.path.join(data_dir, 'data', args.data, 'val')
    traindir = os.path.join(data_dir, 'data', args.data, 'train')

    if args.data == 'nyudepthv2' or args.data == 'uow_dataset':
        from dataloaders.nyu import NYUDataset
        val_dataset = NYUDataset(valdir, split='val', modality=args.modality)
        #val_dataset = nc.SafeDataset(val_dataset)
        train_dataset = NYUDataset(traindir, split='train', modality=args.modality)
        #train_dataset = nc.SafeDataset(train_dataset)
    else:
        raise RuntimeError('Dataset not found.')

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(
        val_dataset,batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True,
        collate_fn=my_collate
    )
    if not args.evaluate:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True,
                                                   collate_fn=my_collate
        )
    print("=> data loaders created.")

    # evaluation mode
    if args.evaluate:
        assert os.path.isfile(args.evaluate), \
        "=> no model found at '{}'".format(args.evaluate)
        print("=> loading model '{}'".format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        if type(checkpoint) is dict:
            args.start_epoch = checkpoint['epoch']
            best_result = checkpoint['best_result']
            model = checkpoint['model']
            print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        else:
            model = checkpoint
            args.start_epoch = 0
        output_directory = os.path.dirname(args.evaluate)
        if args.predict:
            predict(val_loader,model,output_directory)
        else:
            validate(val_loader, model, args.start_epoch, write_to_file=False)
        return
        # optionally resume from a checkpoint
    elif args.resume:
        chkpt_path = args.resume
        assert os.path.isfile(chkpt_path), \
            "=> no checkpoint found at '{}'".format(chkpt_path)
        print("=> loading checkpoint "
              "'{}'".format(chkpt_path))
        checkpoint = torch.load(chkpt_path)
        #args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        optimizer = torch.optim.SGD(model.parameters(),lr=0.9)
        optimizer.load_state_dict(checkpoint['optimizer'])
        #optimizer = checkpoint['optimizer']
        output_directory = os.path.dirname(os.path.abspath(chkpt_path))
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        args.resume = True
    else:
        print("=> creating Model ({} - {}) ...".format(args.arch,args.decoder))
        #in_channels = len(args.modality)
        if args.arch == 'mobilenet-skipconcat':
            model = models.MobileNetSkipConcat(decoder=args.decoder,output_size=train_loader.dataset.output_size)
        elif args.arch == 'mobilenet-skipadd':
            model = models.MobileNetSkipAdd(decoder=args.decoder,output_size=train_loader.dataset.output_size)
        elif args.arch == 'resnet18-skipconcat':
            model = models.ResNetSkipConcat(layers=18,decoder=args.decoder,output_size=train_loader.dataset.output_size)
        elif args.arch == 'resnet18-skipadd':
            model = models.ResNetSkipAdd(layers=18,output_size=train_loader.dataset.output_size)
        else:
            raise Exception('Invalid architecture')
        print("=> model created.")
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # model = torch.nn.DataParallel(model).cuda() # for multi-gpu training
        model = model.cuda()
        start_epoch = 0
    # define loss function (criterion) and optimizer
    if args.criterion == 'l2':
        criterion = criteria.MaskedMSELoss().cuda()
    elif args.criterion == 'l1':
        criterion = criteria.MaskedL1Loss().cuda()

    # create results folder, if not already exists
    output_directory = utils.get_output_directory(args)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    train_csv = os.path.join(output_directory, 'train.csv')
    test_csv = os.path.join(output_directory, 'test.csv')
    best_txt = os.path.join(output_directory, 'best.txt')

    # create new csv files with only header
    if not args.resume:
        with open(train_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        with open(test_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    #start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        utils.adjust_learning_rate(optimizer, epoch, args.lr)
        train(train_loader, model, criterion, optimizer, epoch)  # train for one epoch
        result, img_merge = validate(val_loader, model, epoch, write_to_file=True)  # evaluate on validation set

        # remember best rmse and save checkpoint
        is_best = result.rmse < best_result.rmse
        if is_best:
            best_result = result
            with open(best_txt, 'w') as txtfile:
                txtfile.write(
                    "epoch={}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
                    format(epoch, result.mse, result.rmse, result.absrel, result.lg10, result.mae, result.delta1,
                           result.gpu_time))
            if img_merge is not None:
                img_filename = output_directory + '/comparison_best.png'
                utils.save_image(img_merge, img_filename)

        utils.save_checkpoint({
            'args': args,
            'epoch': epoch,
            #'arch': args.arch,
            'model': model,
            'best_result': best_result,
            'optimizer': optimizer.state_dict(),
        }, is_best, epoch, output_directory)

def train(train_loader, model, criterion, optimizer, epoch):
    average_meter = AverageMeter()
    model.train() # switch to train mode
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute pred
        end = time.time()
        pred = model(input)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward() # compute gradient and do SGD step
        optimizer.step()
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print('=> output: {}'.format(output_directory))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                  epoch, i+1, len(train_loader), data_time=data_time,
                  gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()
    with open(train_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
            'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
            'gpu_time': avg.gpu_time, 'data_time': avg.data_time})

def predict(val_loader, model, dir):
    predict_dir = os.path.join(dir,'predict')
    os.makedirs(predict_dir,exist_ok=True)
    model.eval()  # switch to evaluate mode
    end = time.time()
    for i, (input,target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        # torch.cuda.synchronize()

        with torch.no_grad():
            pred = model(input)
        pred_depth_map = transforms.ToPILImage()(pred[0].cpu())
        target_depth_map = transforms.ToPILImage()(target[0].cpu())
        input_img = transforms.ToPILImage('RGB')(input[0].cpu())

        f, axs = plt.subplots(1, 3)
        axs[0].imshow(input_img)
        axs[1].imshow(pred_depth_map)
        axs[2].imshow(target_depth_map)
        img_file = os.path.join(predict_dir,'{}.png'.format(i))
        plt.savefig(img_file)


def validate(val_loader, model, epoch, write_to_file=True):
    average_meter = AverageMeter()
    model.eval() # switch to evaluate mode
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        # torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred = model(input)
        # torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # save 8 images for visualization
        skip = 50

        if args.modality == 'rgb':
            rgb = input
        offset = 1
        if i == offset:
            img_merge = utils.merge_into_row(rgb, target, pred)
        elif (i < 8*skip+offset) and (i % skip == offset):
            row = utils.merge_into_row(rgb, target, pred)
            img_merge = utils.add_row(img_merge, row)
        elif i == (8*skip+offset):
            filename = output_directory + '/comparison_' + str(epoch) + '.png'
            utils.save_image(img_merge, filename)

        if (i+1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                   i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time})
    return avg, img_merge

if __name__ == '__main__':
    main()
