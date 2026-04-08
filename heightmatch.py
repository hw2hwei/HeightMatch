import os
import argparse
import logging
import pprint
import yaml

import torch
import torch.backends.cudnn as cudnn

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset.semi import SemiDataset
from model import Build_RGB_HGT_DinoV2
from evaluate import evaluate_rgb

from util.classes import CLASSES
from util.utils import AverageMeter, count_params, init_log
from util.losses import UnlCrossEntropyLoss
from util.dist_helper import setup_distributed, average_across_ranks


parser = argparse.ArgumentParser(description='HeightMatch in Semi-Supervised Biuilding Footprint Extraction')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)

parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--val-id-path', type=str, required=True)

parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    # ------- build model and optimizer ------- #
    model = Build_RGB_HGT_DinoV2(cfg['nclass'])

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    param_groups = [
        {'params': model.get_1x_lr_params(), 'lr': cfg['criterion']['lr']},
        {'params': model.get_mx_lr_params(), 'lr': cfg['criterion']['lr'] * cfg['criterion']['lr_multi']},
    ]
    optimizer = AdamW(
        param_groups, betas=(0.9, 0.999), weight_decay=0.01
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=True
    )

    # ------------ start training from scratch ------------ #
    start_epoch = 0
    previous_best = 0.0

    criterion_l = nn.CrossEntropyLoss().cuda(local_rank)
    criterion_u = UnlCrossEntropyLoss(cfg['conf_thresh'], local_rank)

    # ------------ build dataloader ----------- #
    trainset_u = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_u',
        cfg['class_bd'], cfg['class_bg'], cfg['crop_size'],
        id_path=args.unlabeled_id_path
    )
    trainset_l = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_l',
        cfg['class_bd'], cfg['class_bg'], cfg['crop_size'],
        id_path=args.labeled_id_path, nsample=len(trainset_u.ids)
    )
    valset = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'val',
        cfg['class_bd'], cfg['class_bg'], id_path=args.val_id_path
    )

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)

    trainloader_l = DataLoader(
        trainset_l,
        batch_size=cfg['batch_size'],
        pin_memory=False,
        num_workers=4,
        drop_last=True,
        sampler=trainsampler_l
    )
    trainloader_u = DataLoader(
        trainset_u,
        batch_size=cfg['batch_size'],
        pin_memory=False,
        num_workers=4,
        drop_last=True,
        sampler=trainsampler_u
    )
    valloader = DataLoader(
        valset,
        batch_size=cfg['batch_size'] * 2,
        pin_memory=False,
        num_workers=4,
        drop_last=False,
        sampler=valsampler
    )

    # ------------- training loop ------------- #
    total_iters = len(trainloader_u) * cfg['epochs']

    for epoch in range(start_epoch, cfg['epochs']):
        if rank == 0:
            logger.info(
                f"Epoch: {epoch}, LR: {optimizer.param_groups[0]['lr']:.2e}, "
                f"Previous best: {previous_best:.2f}"
            )

        total_loss = AverageMeter()
        total_loss_l = AverageMeter()
        total_loss_u = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)

        for i, (data_l, data_u) in enumerate(loader):
            model.train()

            rgb_l, dpt_l, mask_l = data_l
            rgb_l, dpt_l = rgb_l.cuda().detach(), dpt_l.cuda().detach()
            mask_l = mask_l.cuda().detach()

            rgb_w, rgb_s, dpt_w, dpt_s, _, _ = data_u
            rgb_w, rgb_s = rgb_w.cuda().detach(), rgb_s.cuda().detach()
            dpt_w, dpt_s = dpt_w.cuda().detach(), dpt_s.cuda().detach()

            rgb = torch.cat((rgb_l, rgb_w, rgb_s), dim=0)
            dpt = torch.cat((dpt_l, dpt_w, dpt_s), dim=0)

            pred_l_rgb, pred_l_fus, \
            pred_w_rgb, pred_w_fus, \
            pred_s_rgb, pred_s_fus = model(rgb, dpt, mode='train')

            # ---------- supervised training ---------- #
            loss_l_rgb = criterion_l(pred_l_rgb, mask_l)
            loss_l_fus = criterion_l(pred_l_fus, mask_l)
            loss_l = loss_l_rgb + loss_l_fus

            # --------- unsupervised training --------- #
            conf_w_fus, mask_w_fus = pred_w_fus.softmax(dim=1).max(dim=1)
            conf_w_rgb, mask_w_rgb = pred_w_rgb.softmax(dim=1).max(dim=1)

            conf_w = torch.maximum(conf_w_fus, conf_w_rgb)
            mask_w = torch.where(conf_w_fus >= conf_w_rgb, mask_w_fus, mask_w_rgb)
            mask_ratio = (conf_w_fus >= cfg['conf_thresh']).sum().item() / conf_w_fus.numel()

            loss_u_rgb = criterion_u(pred_s_rgb, mask_w, conf_w)
            loss_u_fus = criterion_u(pred_s_fus, mask_w_fus, conf_w_fus)
            loss_u = loss_u_rgb + loss_u_fus

            # -------------- overall loss ------------- #
            loss = (loss_l + loss_u) / 2.0
            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.module.update_ema()

            total_loss.update(loss.item())
            total_loss_l.update(loss_l.item())
            total_loss_u.update(loss_u.item())
            total_mask_ratio.update(mask_ratio)

            # ---------- adjust learning rate --------- #
            iters = epoch * len(trainloader_u) + i
            lr = cfg['criterion']['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['criterion']['lr_multi']

            # --------------- print info -------------- #
            if (i % (len(trainloader_u) // 8) == 0) and (i > 0) and (rank == 0):
                logger.info(
                    f'Iters: {i}, Total loss: {total_loss.avg:.3f} '
                    f'Loss l: {total_loss_l.avg:.3f}, Loss u: {total_loss_u.avg:.3f}, '
                    f'Mask ratio: {total_mask_ratio.avg:.3f}'
                )

        # --------- evaluate and save model --------- #
        iou_cls, f1_cls, prec_cls, recall_cls = evaluate_rgb(model, valloader, cfg)

        iou_cls = [average_across_ranks(iou, world_size) for iou in iou_cls]
        prec_cls = [average_across_ranks(prec, world_size) for prec in prec_cls]
        recall_cls = [average_across_ranks(recall, world_size) for recall in recall_cls]
        iou_buildings = iou_cls[1]

        if rank == 0:
            max_cls_name_length = max(len(name) for name in CLASSES[cfg['dataset']]) + 1
            for cls_idx, cls_name in enumerate(CLASSES[cfg['dataset']]):
                logger.info(
                    f"***** Evaluation ***** >>>> "
                    f"Class [{cls_idx} {cls_name:<{max_cls_name_length}}] "
                    f"IoU: {iou_cls[cls_idx]:.2f}, "
                    f"Precision: {prec_cls[cls_idx]:.2f}, "
                    f"Recall: {recall_cls[cls_idx]:.2f}"
                )
            logger.info('***** Evaluation ***** >>>> IoU_Buildings: {:.2f}\n'.format(iou_buildings))

        is_best = iou_buildings > previous_best
        previous_best = max(iou_buildings, previous_best)

        if rank == 0:
            checkpoint = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()