#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import json
import timeit
import argparse

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data

import networks
import utils.schp as schp
from datasets.datasets import LIPDataSet
from datasets.target_generation import generate_edge_tensor
from utils.transforms import BGR2RGB_transform
from utils.criterion_new import CriterionAll
from utils.encoding import DataParallelModel, DataParallelCriterion
from utils.warmup_scheduler import SGDRScheduler
from utils.constant import CLASS_WEIGHT_19, CLASS_WEIGHT_24


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    # Network Structure
    parser.add_argument("--arch", type=str, default='swin_cdg')
    # Data Preference
    parser.add_argument("--data-dir", type=str, default='/home/tsl/data/HumanParsing/mannequin_data/24_category_data/datasets_24')
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--input-size", type=str, default='1024,768')
    parser.add_argument("--num-classes", type=int, default=24)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--random-mirror", action="store_true")
    parser.add_argument("--random-scale", action="store_true")
    # Training Strategy
    parser.add_argument("--learning-rate", type=float, default=7e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--eval-epochs", type=int, default=2)
    # parser.add_argument("--imagenet-pretrain", type=str, default='./pretrained/solider_swin_base.pth')
    parser.add_argument("--imagenet-pretrain", type=str, default='./log/checkpoint_230.pth.tar')
    parser.add_argument("--log-dir", type=str, default='./log/test_swincdg')
    parser.add_argument("--model-restore", type=str, default='./log/checkpoint.pth.tar')
    parser.add_argument("--schp-start", type=int, default=10, help='schp start epoch')
    parser.add_argument("--cycle-epochs", type=int, default=5, help='schp cyclical epoch')
    parser.add_argument("--schp-restore", type=str, default='./log/schp_checkpoint.pth.tar')
    parser.add_argument("--lambda-s", type=float, default=1, help='segmentation loss weight')
    parser.add_argument("--lambda-e", type=float, default=1, help='edge loss weight')
    parser.add_argument("--lambda-c", type=float, default=0.1, help='segmentation-edge consistency loss weight')
    parser.add_argument("--syncbn", action="store_true", help='use syncbn or not')
    parser.add_argument("--imagenet", action="store_true", help='use syncbn or not')
    parser.add_argument("--optimizer", type=str, default='sgd', help='which optimizer to use')

    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--lr_divider", type=int, default=500)
    parser.add_argument("--cyclelr-divider", type=int, default=2)

    return parser.parse_args()


def main():
    args = get_arguments()

    start_epoch = 0
    cycle_n = 0

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        print('Create log directory: {}'.format(args.log_dir))

    with open(os.path.join(args.log_dir, 'args.json'), 'w') as opt_file:
        json.dump(vars(args), opt_file)

    print(args)
    gpus = [int(i) for i in args.gpu.split(',')]
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda:0")

    torch.cuda.set_device(device)
    input_size = list(map(int, args.input_size.split(',')))

    cudnn.enabled = True
    cudnn.benchmark = True

    # Model Initialization
    if args.imagenet:
        convert_weights = True
    else:
        convert_weights = False
    model = networks.init_model(args.arch, num_classes=args.num_classes, pretrained=args.imagenet_pretrain,
                                convert_weights=convert_weights)
    for name, param in model.named_parameters():
        #if name.startswith("backbone.patch_embed"):
        if "patch_embed" in name:
            print(name)
            param.requires_grad = False

    IMAGE_MEAN = model.mean
    IMAGE_STD = model.std
    INPUT_SPACE = model.input_space

    restore_from = args.model_restore
    if os.path.exists(restore_from):
        print('Resume training from {}'.format(restore_from))
        checkpoint = torch.load(restore_from, map_location='cpu')
        _state_dict = checkpoint['state_dict']
        if list(_state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in _state_dict.items()}
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(_state_dict)
        start_epoch = checkpoint['epoch']
    model.to(device)
    if args.syncbn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    model = DataParallelModel(model)

    schp_model = networks.init_model(args.arch, num_classes=args.num_classes, pretrained=args.imagenet_pretrain,
                                     convert_weights=convert_weights)
    #for name, param in schp_model.named_parameters():
    #if name.startswith("backbone.patch_embed"):
    #   if "patch_embed" in name:
    #      param.requires_grad = False

    if os.path.exists(args.schp_restore):
        print('Resuming schp checkpoint from {}'.format(args.schp_restore))
        schp_checkpoint = torch.load(args.schp_restore, map_location='cpu')
        schp_model_state_dict = schp_checkpoint['state_dict']
        cycle_n = schp_checkpoint['cycle_n']
        schp_model.load_state_dict(schp_model_state_dict)

    schp_model.to(device)
    if args.syncbn:
        print('----use syncBN in model!----')
        schp_model = nn.SyncBatchNorm.convert_sync_batchnorm(schp_model)
    # schp_model = DDP(schp_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    schp_model = DataParallelModel(schp_model)

    if args.num_classes == 19:
        use_class_weight = torch.tensor(CLASS_WEIGHT_19, dtype=torch.float32, device=device)
    elif args.num_classes == 24:
        use_class_weight = torch.tensor(CLASS_WEIGHT_24, dtype=torch.float32, device=device)
    else:
        use_class_weight = None

    # Loss Function
    criterion = CriterionAll(lambda_1=args.lambda_s, lambda_2=args.lambda_e, lambda_3=args.lambda_c,
                             num_classes=args.num_classes, use_class_weight=use_class_weight)
    criterion = DataParallelCriterion(criterion)
    criterion.to(device)

    # Data Loader
    if INPUT_SPACE == 'BGR':
        print('BGR Transformation')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),
        ])

    elif INPUT_SPACE == 'RGB':
        print('RGB Transformation')
        transform = transforms.Compose([
            transforms.ToTensor(),
            BGR2RGB_transform(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),
        ])

    train_dataset = LIPDataSet(args.data_dir, 'train', crop_size=input_size, transform=transform,
                               arch=args.arch, num_classes=args.num_classes)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size * len(gpus), shuffle=True,
                                   num_workers=8, pin_memory=True, drop_last=True)
    print('Total training samples: {}'.format(len(train_dataset)))

    # Optimizer Initialization
    if args.optimizer == 'sgd':
        print("using SGD optimizer")
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        print("using Adam optimizer")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
                               weight_decay=args.weight_decay)

    # Original warmup_epoch=10, changed to 3 for fix backbone finetune
    lr_scheduler = SGDRScheduler(optimizer, total_epoch=args.epochs,
                                 eta_min=args.learning_rate / args.lr_divider, warmup_epoch=args.warmup_epochs,
                                 start_cyclical=args.schp_start,
                                 cyclical_base_lr=args.learning_rate / args.cyclelr_divider,
                                 cyclical_epoch=args.cycle_epochs)

    total_iters = args.epochs * len(train_loader)
    start = timeit.default_timer()
    iter_start = timeit.default_timer()

    model.train()
    for epoch in range(start_epoch, args.epochs):
        lr_scheduler.step(epoch=epoch)
        lr = lr_scheduler.get_lr()[0]

        for i_iter, batch in enumerate(train_loader):
            i_iter += len(train_loader) * epoch
            images, labels, hgt, wgt, hwgt, _ = batch
            labels = labels.to(device)

            edges = generate_edge_tensor(labels)
            labels = labels.type(torch.cuda.LongTensor)
            edges = edges.type(torch.cuda.LongTensor)
            hgt = hgt.float().cuda(non_blocking=True)
            wgt = wgt.float().cuda(non_blocking=True)
            hwgt = hwgt.float().cuda(non_blocking=True)

            preds = model(images)

            # Online Self Correction Cycle with Label Refinement
            if cycle_n >= 1:
                with torch.no_grad():
                    soft_preds = schp_model(images)
                    soft_fused_preds = soft_preds[0][-1]
                    soft_edges = soft_preds[1][-1]
                    soft_preds = soft_fused_preds
            else:
                soft_preds = None
                soft_edges = None

            loss = criterion(preds, [labels, edges, soft_preds, soft_edges], cycle_n, hwgt=[hgt, wgt, hwgt])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_iter % 100 == 0:
                print('iter = {} of {} completed, lr = {}, loss = {}, time = {}'
                      .format(i_iter,
                              total_iters,
                              lr,
                              loss.data.cpu().numpy(),
                              (timeit.default_timer() - iter_start) / 100)
                      )
                iter_start = timeit.default_timer()
        lr_scheduler.step()
        if (epoch + 1) % (args.eval_epochs) == 0:
            schp.save_schp_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, False, args.log_dir, filename='checkpoint_{}.pth.tar'.format(epoch + 1))

        # Self Correction Cycle with Model Aggregation
        if (epoch + 1) >= args.schp_start and (epoch + 1 - args.schp_start) % args.cycle_epochs == 0:
            print('Self-correction cycle number {}'.format(cycle_n))
            schp.moving_average(schp_model, model, 1.0 / (cycle_n + 1))
            cycle_n += 1
            schp.bn_re_estimate(train_loader, schp_model)
            schp.save_schp_checkpoint({
                'state_dict': schp_model.state_dict(),
                'cycle_n': cycle_n,
            }, False, args.log_dir, filename='schp_{}_checkpoint.pth.tar'.format(cycle_n))

        torch.cuda.empty_cache()
        end = timeit.default_timer()
        print('epoch = {} of {} completed using {} s'.format(epoch, args.epochs,
                                                             (end - start) / (epoch - start_epoch + 1)))

    end = timeit.default_timer()
    print('Training Finished in {} seconds'.format(end - start))


if __name__ == '__main__':
    main()
