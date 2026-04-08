import torch
import torch.distributed as dist

from util.utils import Evaluator


def evaluate_rgb(model, val_loader, cfg):
    model.eval()
    metric = Evaluator(cfg['nclass'])

    with torch.no_grad():
        for rgb, dpt, mask, _ in val_loader:
            rgb = rgb.cuda()

            logit = model(rgb)
            conf = logit.softmax(dim=1)
            pred = conf.argmax(dim=1)

            metric.add_batch(mask, pred)
            dist.barrier()
    model.train()

    Precision, Recall = metric.Precision_Recall()
    IoU = metric.Intersection_over_Union()
    F1 = metric.F1_Score()
    return IoU * 100, F1 * 100, Precision, Recall


def evaluate_dpt(model, val_loader, cfg):
    model.eval()
    metric = Evaluator(cfg['nclass'])

    with torch.no_grad():
        for rgb, dpt, mask, _ in val_loader:
            dpt = dpt.cuda()

            logit = model(dpt)
            conf = logit.softmax(dim=1)
            pred = conf.argmax(dim=1)

            metric.add_batch(mask, pred)
            dist.barrier()
    model.train()

    Precision, Recall = metric.Precision_Recall()
    IoU = metric.Intersection_over_Union()
    F1 = metric.F1_Score()
    return IoU * 100, F1 * 100, Precision, Recall


def evaluate_fus(model, val_loader, cfg):
    model.eval()

    metric = Evaluator(cfg['nclass'])

    with torch.no_grad():
        for rgb, dpt, mask, _ in val_loader:
            rgb = rgb.cuda()
            dpt = dpt.cuda()

            logit = model(rgb=rgb, dpt=dpt)
            conf = logit.softmax(dim=1)
            pred = conf.argmax(dim=1)

            metric.add_batch(mask, pred)
            dist.barrier()
    model.train()

    Precision, Recall = metric.Precision_Recall()
    IoU = metric.Intersection_over_Union()
    F1 = metric.F1_Score()
    return IoU * 100, F1 * 100, Precision, Recall


