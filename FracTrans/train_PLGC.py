# -*- coding: utf-8 -*-
# @Time    : 2025/06/28
# @Author  : Grok (基于 Geng Qin 的代码修改)
# @File    : train_PLGC.py
# @Description: 训练胃癌高光谱数据集的分类模型（三分类，.npy 文件，优化版以减少 RAM 和 GPU 内存占用）

import os
import numpy as np
import torch
import sys
import argparse
import logging
import logging.handlers
import queue
import threading
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from torch.optim import AdamW
from tensorboardX import SummaryWriter
from sklearn.metrics import precision_score, recall_score, cohen_kappa_score, roc_auc_score, confusion_matrix, f1_score
from scipy import stats
import warnings
import psutil
import math
import gc
import traceback
from tqdm import tqdm
from model.FracTrans import get_SFT_Swin as FracTrans
from dataset.PLGC_dataset import PLGC_HSIClassification as HSI_Dataset, get_train_loader
warnings.filterwarnings("ignore")

# 设置 PyTorch 内存分配器配置以减少碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 导入模型和数据集


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/mnt/nvme1n1/workingkarl/dataset/duomotaiPLGC/PLGC_256_HSI',
                    help='Path to the hyperspectral dataset folder')
parser.add_argument('--csv_dir', type=str, default='/mnt/nvme1n1/workingkarl/dataset/duomotaiPLGC',
                    help='Directory containing PLGC_flod82train.txt and PLGC_flod82test.txt')
parser.add_argument('--exp', type=str, default='PLGC/xuedi_1', help='Experiment name')
parser.add_argument('--snapshot_dir', type=str, default='save_checkpoint_hsi',
                    help='Directory to save model checkpoints')
parser.add_argument('--model', type=str, default='SSFT_UnBlocksV2', help='Model name')
parser.add_argument('--nepochs', type=int, default=200, help='Maximum number of epochs to train')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU (reduced to minimize RAM usage)')
parser.add_argument('--accum_steps', type=int, default=1, help='Gradient accumulation steps')
parser.add_argument('--deterministic', type=int, default=1, help='Whether to use deterministic training')
parser.add_argument('--transforms', type=str, default='high_aug',
                    help='Data augmentation type: light_aug, medium_aug, high_aug')
parser.add_argument('--seed', type=int, default=43, help='Random seed')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers (0 to minimize RAM usage)')
parser.add_argument('--num_classes', type=int, default=3, help='Number of output classes (three-class classification)')
parser.add_argument('--fold', type=int, default=1, help='K-fold in the dataset')
parser.add_argument('--ema_decay', type=float, default=0.99, help='EMA decay rate')
parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
parser.add_argument('--base_lr', type=float, default=1e-4, help='Base learning rate')
parser.add_argument('--warm_up_epoch', type=int, default=20, help='Warm-up epochs for learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for AdamW optimizer')
parser.add_argument('--patience', type=int, default=40, help='Patience for early stopping')
parser.add_argument('--pe_type', type=str, default='learnable', help='Position encoding type: learnable or cosine')
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint for resuming training')
args = parser.parse_args()

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# 设置异步日志处理
log_queue = queue.Queue()


def setup_logging(snapshot_path, model_name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清空现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    log_file = os.path.join(snapshot_path, f"log_{model_name}.txt")
    file_handler = logging.handlers.QueueHandler(log_queue)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    def log_writer():
        with open(log_file, 'a', buffering=1000) as file_stream:
            while True:
                record = log_queue.get()
                if record is None:
                    break
                file_stream.write(formatter.format(record) + '\n')
                file_stream.flush()

    log_thread = threading.Thread(target=log_writer, daemon=True)
    log_thread.start()
    return logger, log_thread


logging.info(f"Using device: {device}")
if torch.cuda.is_available():
    logging.info(f"GPU device name: {torch.cuda.get_device_name(3)}")
    logging.info(f"CUDA version: {torch.version.cuda}")
else:
    logging.info("CUDA is not available. Running on CPU.")
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 启用同步 CUDA 以便详细错误追踪

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.log_softmax(inputs)
        if self.weight is not None:
            weights = self.weight[targets]
            log_probs = log_probs * weights.unsqueeze(1)
        n_classes = inputs.size(1)
        true_dist = torch.zeros_like(log_probs).fill_(self.smoothing / (n_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


def clear_batch_memory(**kwargs):
    """
    Clear GPU memory for specified tensors and call garbage collection.
    """
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            del value
    gc.collect()
    torch.cuda.empty_cache()
    logging.debug(f"Batch memory cleared: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB allocated, "
                  f"{torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB reserved")


def update_class_weights(model, true_labels, pred_labels, class_counts, device):
    """
    Update class weights based on training predictions without loading new data.
    """
    per_class_correct = [0] * len(class_counts)
    per_class_total = [0] * len(class_counts)

    for true, pred in zip(true_labels, pred_labels):
        label = int(true)
        per_class_total[label] += 1
        if pred == label:
            per_class_correct[label] += 1

    class_acc = [correct / total if total > 0 else 0 for correct, total in zip(per_class_correct, per_class_total)]
    inverse_acc = [1.0 / (acc + 1e-6) for acc in class_acc]
    total_inverse = sum(inverse_acc)
    new_weights = torch.tensor([freq / total_inverse for freq in inverse_acc], dtype=torch.float32).to(device)
    old_weights = model.class_weights if hasattr(model, 'class_weights') else new_weights
    new_weights = 0.8 * old_weights + 0.2 * new_weights
    new_weights = torch.clamp(new_weights, min=0.5, max=10.0)
    return new_weights


def get_network(args):
    if args.model == 'FracTrans':
        model = FracTrans(num_classes=args.num_classes)
    
    else:
        raise NotImplementedError(f"Model {args.model} not implemented.")
    return model.to(device)


def compute_metrics(true_labels, pred_labels, pred_probs, num_classes):
    try:
        # Overall Accuracy
        overall_acc = 100. * (pred_labels == true_labels).sum() / len(true_labels)

        # Per-class Accuracy
        per_class_acc = []
        for i in range(num_classes):
            class_true = (true_labels == i)
            class_pred = (pred_labels == i)
            if class_true.sum() > 0:
                per_class_acc.append(100. * (class_true & class_pred).sum() / class_true.sum())
            else:
                per_class_acc.append(0.0)
        avg_acc = np.mean(per_class_acc)

        # Precision, Recall, F1
        precision = 100. * precision_score(true_labels, pred_labels, average='macro', zero_division=0)
        recall = 100. * recall_score(true_labels, pred_labels, average='macro', zero_division=0)
        f1 = 100. * f1_score(true_labels, pred_labels, average='macro', zero_division=0)

        # Kappa
        kappa = cohen_kappa_score(true_labels, pred_labels)

        # Specificity (OvR-based for multi-class)
        cm = confusion_matrix(true_labels, pred_labels)
        spe = []
        for i in range(num_classes):
            binary_true = (true_labels == i).astype(int)
            binary_pred = (pred_labels == i).astype(int)
            tn = ((1 - binary_true) & (1 - binary_pred)).sum()
            fp = ((1 - binary_true) & binary_pred).sum()
            spe.append(100. * tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        spe = np.mean(spe)

        # AUC with enhanced validation
        auc_score = None
        if pred_probs is not None:
            if pred_probs.shape[1] == num_classes:  # Check if number of columns matches number of classes
                if not np.any(np.isnan(pred_probs)):  # Check for NaN values
                    # Verify that probabilities sum to approximately 1 for each sample
                    prob_sum = np.sum(pred_probs, axis=1)
                    if np.all(np.abs(prob_sum - 1.0) < 1e-5):  # Allow small numerical errors
                        auc_score = 100. * roc_auc_score(true_labels, pred_probs, multi_class='ovr')
                    else:
                        logging.warning("Probabilities do not sum to 1, AUC calculation skipped")
                else:
                    logging.warning("NaN values detected in pred_probs, AUC calculation skipped")
            else:
                logging.warning(
                    f"pred_probs shape {pred_probs.shape} does not match num_classes {num_classes}, AUC calculation skipped")

        # P-value
        p_values = []
        for class_idx in range(num_classes):
            binary_true = (true_labels == class_idx).astype(int)
            class_probs = pred_probs[:, class_idx]
            scores_class = class_probs[binary_true == 1]
            scores_rest = class_probs[binary_true == 0]
            if len(scores_class) > 0 and len(scores_rest) > 0:
                _, p_val = stats.mannwhitneyu(scores_class, scores_rest, alternative='two-sided')
                p_values.append(p_val)
            else:
                p_values.append(1.0)
        p_value = np.mean(p_values) if p_values else None

        return overall_acc, avg_acc, precision, recall, kappa, f1, spe, auc_score, per_class_acc, p_value
    except Exception as e:
        logging.error(f"Error in compute_metrics: {str(e)}\n{traceback.format_exc()}")
        return 0, 0, 0, 0, 0, 0, 0, 0, [0] * num_classes, 1.0


def get_learning_rate(base_lr, current_iter, warmup_iters, total_iters):
    min_lr = base_lr * 0.01
    if current_iter < warmup_iters:
        lr = base_lr * current_iter / warmup_iters
    else:
        cosine_iters = total_iters - warmup_iters
        current_cosine_iter = current_iter - warmup_iters
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * current_cosine_iter / cosine_iters))
    return lr


def save_checkpoint(state, filename):
    try:
        torch.save(state, filename)
        logging.info(f"Checkpoint saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving checkpoint: {str(e)}\n{traceback.format_exc()}")


def load_checkpoint(checkpoint_path, model, optimizer):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        iter_count = checkpoint['iter_count']
        best_val_acc = checkpoint.get('best_val_acc', 0)
        best_test_acc = checkpoint.get('best_test_acc', 0)
        patience_counter = checkpoint.get('patience_counter', 0)
        logging.info(f"Resumed from checkpoint: {checkpoint_path}, starting from epoch {start_epoch}")
        return start_epoch, iter_count, best_val_acc, best_test_acc, patience_counter
    except Exception as e:
        logging.error(f"Error loading checkpoint: {str(e)}\n{traceback.format_exc()}")
        raise


def delete_old_checkpoint(directory, current_best_val_path, current_best_test_path):
    for file in os.listdir(directory):
        if file.endswith('.pth') and file not in [os.path.basename(current_best_val_path),
                                                  os.path.basename(current_best_test_path)]:
            file_path = os.path.join(directory, file)
            try:
                os.remove(file_path)
                logging.info(f"Deleted old checkpoint: {file_path}")
            except Exception as e:
                logging.error(f"Error deleting old checkpoint {file_path}: {str(e)}")


def train(args, snapshot_path):
    train_transforms = args.transforms
    test_transforms = None

    try:
        train_dataset = HSI_Dataset(root=args.root_path, txt_name='PLGC_flod82train.txt', transforms=train_transforms,
                                    training=True)
        test_dataset = HSI_Dataset(root=args.root_path, txt_name='PLGC_flod82test.txt', transforms=test_transforms,
                                   training=False)

        class_counts = [train_dataset.labels.count(i) for i in range(args.num_classes)]
        num_samples = len(train_dataset.labels)
        weights = [1.0 / class_counts[label] for label in train_dataset.labels]
        sampler = WeightedRandomSampler(weights, num_samples, replacement=True)

        train_loader = get_train_loader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                        sampler=sampler)
        test_loader = get_train_loader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        logging.info(f"Train dataset size: {len(train_dataset)}")
        logging.info(f"Test dataset size: {len(test_dataset)}")

        model = get_network(args)
        param_count = sum(p.numel() for p in model.parameters())
        logging.info(f"Model {args.model} loaded with {param_count / 1e6:.2f}M parameters, pe_type: {args.pe_type}")

        optimizer = AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

        inverse_freq = [1.0 / (count + 1e-6) for count in class_counts]
        total_inverse = sum(inverse_freq)
        class_weights = torch.tensor([freq / total_inverse * 3.0 for freq in inverse_freq], dtype=torch.float32).to(
            device)
        class_weights = torch.clamp(class_weights, min=0.5, max=10.0)
        model.class_weights = class_weights
        logging.info(f'Initial Class Distribution: {class_counts}, Class Weights: {class_weights.tolist()}')

        ce_criterion = LabelSmoothingCrossEntropy(smoothing=0.1, weight=class_weights)
        focal_criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        alpha = 0.7

        logger = SummaryWriter(snapshot_path + '/log')
        total_iters = args.nepochs * len(train_loader)
        warmup_iters = args.warm_up_epoch * len(train_loader)

        start_epoch = 0
        iter_count = 0
        best_val_acc = 0
        best_test_acc = 0
        patience_counter = 0
        best_val_path = ''
        best_test_path = ''

        checkpoint_path = os.path.join(snapshot_path, 'checkpoint.pth')
        if args.resume and os.path.exists(args.resume):
            checkpoint_path = args.resume
        if os.path.exists(checkpoint_path):
            start_epoch, iter_count, best_val_acc, best_test_acc, patience_counter = load_checkpoint(checkpoint_path,
                                                                                                     model, optimizer)

        for epoch in range(start_epoch, args.nepochs):
            model.train()
            loss_total = 0
            ce_loss_total = 0
            focal_loss_total = 0
            mi_loss_total = 0
            train_true = []
            train_pred = []
            train_probs = []

            optimizer.zero_grad()
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.nepochs} [Training]", leave=False,
                             total=len(train_loader))
            for batch_idx, batch in enumerate(train_bar):
                try:
                    images, true_labels = batch
                    images = images.to(device=device, dtype=torch.float32, non_blocking=True)
                    true_labels = true_labels.to(device=device, dtype=torch.long, non_blocking=True)

                    if true_labels.min() < 0 or true_labels.max() >= args.num_classes:
                        logging.warning(
                            f"Invalid labels in batch {batch_idx}: min={true_labels.min()}, max={true_labels.max()}")
                        continue

                    outputs = model(images)
                    clear_batch_memory(images=images)  # Clear input after forward pass

                    if isinstance(outputs, tuple):
                        logits, aux_logits = outputs
                        ce_loss = ce_criterion(logits, true_labels)
                        focal_loss = focal_criterion(logits, true_labels)
                        aux_ce_losses = [ce_criterion(aux_logit, true_labels) for aux_logit in aux_logits]
                        aux_focal_losses = [focal_criterion(aux_logit, true_labels) for aux_logit in aux_logits]
                        mi_loss = torch.stack([alpha * ce + (1 - alpha) * focal for ce, focal in
                                               zip(aux_ce_losses, aux_focal_losses)]).mean()
                    else:
                        ce_loss = ce_criterion(outputs, true_labels)
                        focal_loss = focal_criterion(outputs, true_labels)
                        mi_loss = torch.tensor(0.0, device=device)

                    loss = alpha * ce_loss + (1 - alpha) * focal_loss + mi_loss

                    if torch.isnan(loss) or torch.isinf(loss):
                        logging.warning(f"Loss is NaN or Inf at batch {batch_idx}, skipping")
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    loss_total += loss.item()
                    ce_loss_total += ce_loss.item()
                    focal_loss_total += focal_loss.item()
                    mi_loss_total += mi_loss.item()

                    if (epoch + 1) % 5 == 0 or epoch == args.nepochs - 1:
                        main_outputs = outputs[0] if isinstance(outputs, tuple) else outputs
                        _, predicted = torch.max(main_outputs, 1)
                        probs = torch.softmax(main_outputs, dim=1).detach()
                        train_true.append(true_labels.cpu().numpy())
                        train_pred.append(predicted.cpu().numpy())
                        train_probs.append(probs.cpu().numpy())
                        clear_batch_memory(main_outputs=main_outputs, predicted=predicted, probs=probs)

                    iter_count += 1
                    lr = get_learning_rate(args.base_lr, iter_count, warmup_iters, total_iters)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                    train_bar.set_postfix({'Train Loss': f'{loss.item():.4f}'})

                    # Clear remaining batch tensors
                    clear_batch_memory(outputs=outputs, loss=loss, ce_loss=ce_loss, focal_loss=focal_loss,
                                       mi_loss=mi_loss,
                                       logits=logits if 'logits' in locals() else None,
                                       aux_logits=aux_logits if 'aux_logits' in locals() else None)

                except Exception as e:
                    logging.error(f"Error in batch {batch_idx}: {str(e)}\n{traceback.format_exc()}")
                    continue

            train_bar.close()

            train_loss = loss_total / len(train_loader) if len(train_loader) > 0 else 0
            ce_loss_avg = ce_loss_total / len(train_loader) if len(train_loader) > 0 else 0
            focal_loss_avg = focal_loss_total / len(train_loader) if len(train_loader) > 0 else 0
            mi_loss_avg = mi_loss_total / len(train_loader) if len(train_loader) > 0 else 0

            train_log = (f"Epoch {epoch + 1}/{args.nepochs} || Train Loss: {train_loss:.4f} || "
                         f"CE Loss: {ce_loss_avg:.4f} || Focal Loss: {focal_loss_avg:.4f} || MI Loss: {mi_loss_avg:.4f}")
            print(train_log)
            logging.info(train_log)
            logger.add_scalar('Train/Loss', train_loss, epoch)

            if (epoch + 1) % 5 == 0 or epoch == args.nepochs - 1:
                if train_true:
                    train_true = np.concatenate(train_true)
                    train_pred = np.concatenate(train_pred)
                    train_probs = np.concatenate(train_probs)
                    train_oa, train_aa, train_prec, train_rec, train_kappa, train_f1, train_spe, train_auc, per_class_acc, train_p_value = compute_metrics(
                        train_true, train_pred, train_probs, args.num_classes)

                    train_metrics_log = (f"Epoch {epoch + 1}/{args.nepochs} || Train Metrics || "
                                         f"Acc: {train_oa:.2f}% || Avg Acc: {train_aa:.2f}% || Prec: {train_prec:.2f}% || Sen: {train_rec:.2f}% || "
                                         f"Kappa: {train_kappa:.4f} || F1: {train_f1:.2f}% || Spe: {train_spe:.2f}% || AUC: {train_auc:.2f}% || "
                                         f"Per Class Acc: {per_class_acc} || p-value: {train_p_value:.4f}")
                    print(train_metrics_log)
                    logging.info(train_metrics_log)
                    logger.add_scalar('Train/Accuracy', train_oa, epoch)

                    # Update class weights using existing predictions
                    new_weights = update_class_weights(model, train_true, train_pred, class_counts, device)
                    model.class_weights = new_weights
                    ce_criterion.weight = new_weights
                    focal_criterion.alpha = new_weights
                    weights_log = f"Epoch {epoch + 1} || Updated Class Weights: {new_weights.tolist()}"
                    print(weights_log)
                    logging.info(weights_log)

                    # Clear metrics memory
                    del train_true, train_pred, train_probs
                    clear_batch_memory()

            if (epoch % 2 == 0 and epoch >= 10) or epoch == args.nepochs - 1:
                model.eval()
                val_loss_total = 0.0
                val_oa_total = 0
                val_count = 0
                val_true = []
                val_pred = []
                val_probs = []

                val_bar = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{args.nepochs} [Validation]", leave=False,
                               total=len(test_loader))
                for batch in val_bar:
                    try:
                        images, labels = batch
                        images = images.to(device, dtype=torch.float32, non_blocking=True)
                        labels = labels.to(device, dtype=torch.long, non_blocking=True)
                        with torch.no_grad():
                            outputs = model(images)
                            clear_batch_memory(images=images)

                            if isinstance(outputs, tuple):
                                logits, aux_logits = outputs
                                ce_loss = ce_criterion(logits, labels)
                                focal_loss = focal_criterion(logits, labels)
                                mi_loss = torch.stack([alpha * ce_criterion(aux_logit, labels) + (
                                        1 - alpha) * focal_criterion(aux_logit, labels) for aux_logit in
                                                       aux_logits]).mean()
                            else:
                                ce_loss = ce_criterion(outputs, labels)
                                focal_loss = focal_criterion(outputs, labels)
                                mi_loss = torch.tensor(0.0, device=device)

                            val_loss = alpha * ce_loss + (1 - alpha) * focal_loss + mi_loss
                            val_loss_total += val_loss.item()

                            main_outputs = outputs[0] if isinstance(outputs, tuple) else outputs
                            _, val_predicted = torch.max(main_outputs, 1)
                            probs = torch.softmax(main_outputs, dim=1).detach()
                            val_oa_total += (val_predicted == labels).sum().item()
                            val_count += labels.size(0)
                            val_true.append(labels.cpu().numpy())
                            val_pred.append(val_predicted.cpu().numpy())
                            val_probs.append(probs.cpu().numpy())

                            clear_batch_memory(main_outputs=main_outputs, val_predicted=val_predicted, probs=probs)

                        val_bar.set_postfix({'Val Loss': f'{val_loss.item():.4f}'})

                        # Clear remaining batch tensors
                        clear_batch_memory(outputs=outputs, ce_loss=ce_loss, focal_loss=focal_loss, mi_loss=mi_loss,
                                           val_loss=val_loss,
                                           logits=logits if 'logits' in locals() else None,
                                           aux_logits=aux_logits if 'aux_logits' in locals() else None)

                    except Exception as e:
                        logging.error(f"Error in validation batch: {str(e)}\n{traceback.format_exc()}")
                        continue

                val_bar.close()

                val_loss = val_loss_total / len(test_loader) if len(test_loader) > 0 else 0
                val_oa = 100. * val_oa_total / val_count if val_count > 0 else 0

                # Compute validation metrics
                if val_true:
                    val_true = np.concatenate(val_true)
                    val_pred = np.concatenate(val_pred)
                    val_probs = np.concatenate(val_probs)
                    val_oa, val_aa, val_prec, val_rec, val_kappa, val_f1, val_spe, val_auc, val_per_class_acc, val_p_value = compute_metrics(
                        val_true, val_pred, val_probs, args.num_classes)

                    val_metrics_log = (f"Epoch {epoch + 1}/{args.nepochs} || Validation Metrics || "
                                       f"Acc: {val_oa:.2f}% || Avg Acc: {val_aa:.2f}% || Prec: {val_prec:.2f}% || Sen: {val_rec:.2f}% || "
                                       f"Kappa: {val_kappa:.4f} || F1: {val_f1:.2f}% || Spe: {val_spe:.2f}% || AUC: {val_auc:.2f}% || "
                                       f"Per Class Acc: {val_per_class_acc} || p-value: {val_p_value:.4f}")
                    print("\n" + val_metrics_log)
                    logging.info(val_metrics_log)
                    logger.add_scalar('Val/Loss', val_loss, epoch)
                    logger.add_scalar('Val/Accuracy', val_oa, epoch)
                    logger.add_scalar('Val/Average_Accuracy', val_aa, epoch)
                    logger.add_scalar('Val/Precision', val_prec, epoch)
                    logger.add_scalar('Val/Recall', val_rec, epoch)
                    logger.add_scalar('Val/Kappa', val_kappa, epoch)
                    logger.add_scalar('Val/F1', val_f1, epoch)
                    logger.add_scalar('Val/Specificity', val_spe, epoch)
                    logger.add_scalar('Val/AUC', val_auc, epoch)
                    logger.add_scalar('Val/P_value', val_p_value, epoch)

                    # Clear validation metrics memory
                    del val_true, val_pred, val_probs
                    clear_batch_memory()

                model_l_savedir = os.path.join(args.snapshot_dir, f'model_{args.model}_fold{args.fold}')
                if not os.path.exists(model_l_savedir):
                    os.makedirs(model_l_savedir)

                if val_oa > best_val_acc:
                    old_val_path = best_val_path
                    best_val_acc = val_oa
                    patience_counter = 0
                    best_val_path = os.path.join(model_l_savedir,
                                                 f'fold{args.fold}_epoch{epoch}_{args.model}_best_val.pth')
                    torch.save(model.state_dict(), best_val_path)
                    best_log = f"Best validation model saved with Acc: {best_val_acc:.2f}% at {best_val_path}"
                    print(best_log)
                    logging.info(best_log)
                    if old_val_path and os.path.exists(old_val_path):
                        try:
                            os.remove(old_val_path)
                            logging.info(f"Deleted old validation checkpoint: {old_val_path}")
                        except Exception as e:
                            logging.error(f"Error deleting old validation checkpoint {old_val_path}: {str(e)}")

                if val_oa > best_test_acc:
                    old_test_path = best_test_path
                    best_test_acc = val_oa
                    best_test_path = os.path.join(model_l_savedir,
                                                  f'fold{args.fold}_epoch{epoch}_{args.model}_best_test.pth')
                    torch.save(model.state_dict(), best_test_path)
                    test_log = f"Best test model saved with Acc: {best_test_acc:.2f}% at {best_test_path}"
                    print(test_log)
                    logging.info(test_log)
                    if old_test_path and os.path.exists(old_test_path):
                        try:
                            os.remove(old_test_path)
                            logging.info(f"Deleted old test checkpoint: {old_test_path}")
                        except Exception as e:
                            logging.error(f"Error deleting old test checkpoint {old_test_path}: {str(e)}")

                delete_old_checkpoint(model_l_savedir, best_val_path, best_test_path)

                # Clear GPU memory after validation
                clear_batch_memory()

            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    early_stop_log = f"Early stopping at epoch {epoch + 1} due to no improvement in {args.patience} epochs"
                    print(early_stop_log)
                    logging.info(early_stop_log)
                    break

            checkpoint_state = {
                'epoch': epoch,
                'iter_count': iter_count,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'best_test_acc': best_test_acc,
                'patience_counter': patience_counter
            }
            save_checkpoint(checkpoint_state, checkpoint_path)

            print()

        final_model_path = os.path.join(model_l_savedir, f'fold{args.fold}_{args.model}_final.pth')
        torch.save(model.state_dict(), final_model_path)
        logging.info(f"Final model saved to {final_model_path}")

        model.eval()
        test_loss_total = 0.0
        test_oa_total = 0
        test_count = 0

        test_bar = tqdm(test_loader, desc="Testing", leave=False, total=len(test_loader))
        for batch in test_bar:
            try:
                images, labels = batch
                images = images.to(device, dtype=torch.float32, non_blocking=True)
                labels = labels.to(device, dtype=torch.long, non_blocking=True)
                with torch.no_grad():
                    outputs = model(images)
                    clear_batch_memory(images=images)

                    if isinstance(outputs, tuple):
                        logits, aux_logits = outputs
                        ce_loss = ce_criterion(logits, labels)
                        focal_loss = focal_criterion(logits, labels)
                        mi_loss = torch.stack(
                            [alpha * ce_criterion(aux_logit, labels) + (1 - alpha) * focal_criterion(aux_logit, labels)
                             for aux_logit in aux_logits]).mean()
                    else:
                        ce_loss = ce_criterion(outputs, labels)
                        focal_loss = focal_criterion(outputs, labels)
                        mi_loss = torch.tensor(0.0, device=device)

                    test_loss = alpha * ce_loss + (1 - alpha) * focal_loss + mi_loss
                    test_loss_total += test_loss.item()

                    main_outputs = outputs[0] if isinstance(outputs, tuple) else outputs
                    _, test_predicted = torch.max(main_outputs, 1)
                    test_oa_total += (test_predicted == labels).sum().item()
                    test_count += labels.size(0)

                    clear_batch_memory(main_outputs=main_outputs, test_predicted=test_predicted)

                test_bar.set_postfix({'Test Loss': f'{test_loss.item():.4f}'})

                # Clear remaining batch tensors
                clear_batch_memory(outputs=outputs, ce_loss=ce_loss, focal_loss=focal_loss, mi_loss=mi_loss,
                                   test_loss=test_loss,
                                   logits=logits if 'logits' in locals() else None,
                                   aux_logits=aux_logits if 'aux_logits' in locals() else None)

            except Exception as e:
                logging.error(f"Error in test batch: {str(e)}\n{traceback.format_exc()}")
                continue

        test_bar.close()

        test_loss = test_loss_total / len(test_loader) if len(test_loader) > 0 else 0
        test_oa = 100. * test_oa_total / test_count if test_count > 0 else 0

        test_log = f"Test Results || Test Loss: {test_loss:.4f} || Acc: {test_oa:.2f}%"
        print("\n" + test_log)
        logging.info(test_log)
        logger.add_scalar('Test/Loss', test_loss, epoch)
        logger.add_scalar('Test/Accuracy', test_oa, epoch)

        if test_oa > best_test_acc:
            old_test_path = best_test_path
            best_test_acc = test_oa
            best_test_path = os.path.join(model_l_savedir, f'fold{args.fold}_{args.model}_best_test.pth')
            torch.save(model.state_dict(), best_test_path)
            final_test_log = f"Best test model updated with Acc: {best_test_acc:.2f}% at {best_test_path}"
            print(final_test_log)
            logging.info(final_test_log)
            if old_test_path and os.path.exists(old_test_path):
                try:
                    os.remove(old_test_path)
                    logging.info(f"Deleted old test checkpoint: {old_test_path}")
                except Exception as e:
                    logging.error(f"Error deleting old test checkpoint {old_test_path}: {str(e)}")

        delete_old_checkpoint(model_l_savedir, best_val_path, best_test_path)

        # Clear GPU memory after testing
        clear_batch_memory()

        logger.close()
        log_queue.put(None)  # 通知日志线程退出
        print('Training completed')

    except Exception as e:
        logging.error(f"Training failed: {str(e)}\n{traceback.format_exc()}")
        log_queue.put(None)  # 确保日志线程退出
        raise


if __name__ == "__main__":
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = os.path.join("./{}_{}".format(args.exp, args.fold), args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logger, log_thread = setup_logging(snapshot_path, args.model)
    logging.info(str(args))
    try:
        train(args, snapshot_path)
    finally:
        log_queue.put(None)  # 确保日志线程退出
        log_thread.join()  # 等待日志线程完成




# -*- coding: utf-8 -*-
# @Time    : 2025/06/27
# @Author  : Grok (基于 Geng Qin 的代码修改)
# @File    : PLGC_dataset.py
# @Description: 胃癌高光谱数据的三分类数据集加载器（优化版，减少 RAM 占用）

import os
from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler

# 定义全局设备
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def open_file(datapath):
    """
    直接加载 .npy 文件到内存，优化为最小 RAM 占用。
    """
    if not os.path.exists(datapath):
        raise FileNotFoundError(f"文件 {datapath} 不存在。")

    try:
        data = np.load(datapath, allow_pickle=True)
    except Exception as e:
        raise ValueError(f"加载文件 {datapath} 失败：{e}")

    if data.ndim != 3:
        raise ValueError(f"不支持的 .npy 文件形状：{data.shape}。预期是 3D 数组。")

    H, W, C = data.shape if data.shape[2] in [40, 48] else (data.shape[1], data.shape[2], data.shape[0])

    if C not in [40, 48]:
        raise ValueError(f"不支持的 .npy 文件通道数：{C}。预期是 40 或 48。")

    if data.shape[2] not in [40, 48]:
        data = np.transpose(data, (1, 2, 0))  # [C, H, W] -> [H, W, C]

    if C == 48:
        data = data[:, :, :40]
        print(f"警告：文件 {datapath} 通道数为 48，已截取前 40 个通道。")

    return data

# GPU 上的数据增强函数
def gpu_light_aug(data, seed=42):
    """
    轻度数据增强，在 GPU 上执行。
    """
    torch.manual_seed(seed)
    if torch.rand(1, device=device) > 0.5:
        data = torch.flip(data, dims=[1])  # 垂直翻转
    if torch.rand(1, device=device) > 0.5:
        data = torch.rot90(data, k=1, dims=[1, 2])  # 旋转 90 度
    return data

def gpu_medium_aug(data, seed=42):
    """
    中度数据增强，在 GPU 上执行。
    """
    torch.manual_seed(seed)
    if torch.rand(1, device=device) > 0.5:
        data = torch.flip(data, dims=[1])  # 垂直翻转
    if torch.rand(1, device=device) > 0.5:
        data = torch.rot90(data, k=1, dims=[1, 2])  # 旋转 90 度
    return data

def gpu_high_aug(data, seed=42):
    """
    高度数据增强，在 GPU 上执行。
    """
    torch.manual_seed(seed)
    if torch.rand(1, device=device) > 0.5:
        data = torch.flip(data, dims=[1])  # 垂直翻转
    if torch.rand(1, device=device) > 0.5:
        data = torch.rot90(data, k=1, dims=[1, 2])  # 旋转 90 度
    return data

class PLGC_HSIClassification(Dataset):
    def __init__(self, root, txt_name: str = "PLGC_flod82train.txt", training=True, transforms=None):
        super(PLGC_HSIClassification, self).__init__()
        self.root = root
        self.training = training
        self.transforms = transforms

        self.label_value = {"NORMAL": 0, "GIN": 1, "IM": 2}
        txt_path = os.path.join("/home/workingkarl/work/code/SSTM/dataset", txt_name)
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Text file {txt_path} does not exist")

        self.images = []
        self.labels = []
        self.problem_files = []

        with open(txt_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('/')
                if len(parts) != 2:
                    print(f"Skipping invalid line in {txt_name}: {line}")
                    continue
                folder, filename = parts
                folder = folder.upper()
                if folder not in self.label_value:
                    print(f"Skipping invalid folder in {txt_name}: {folder}")
                    continue
                label = self.label_value[folder]
                image_path = os.path.join(self.root, line + '.npy')
                if not os.path.exists(image_path):
                    print(f"Skipping missing file: {image_path}")
                    continue
                self.images.append(image_path)
                self.labels.append(label)

        if len(self.images) == 0:
            raise ValueError(f"No valid data found in {txt_path}")
        assert len(self.images) == len(self.labels), "Images and labels count mismatch"
        print(f"Loaded {len(self.images)} samples from {txt_name}, labels distribution: {[self.labels.count(i) for i in range(3)]}")

        # 设置 transforms 为 GPU 函数
        if self.transforms == 'light_aug':
            self.transforms = gpu_light_aug
        elif self.transforms == 'medium_aug':
            self.transforms = gpu_medium_aug
        elif self.transforms == 'high_aug' or self.transforms == 'strong_aug':
            self.transforms = gpu_high_aug

    def __getitem__(self, index):
        # 按需加载数据
        try:
            img = open_file(self.images[index])
        except Exception as e:
            print(f"Error loading {self.images[index]}: {str(e)}")
            self.problem_files.append(self.images[index])
            img = np.zeros((256, 256, 40), dtype=np.float32)

        label = self.labels[index]

        # 转换为 tensor 并立即移至 GPU
        data = torch.from_numpy(img).permute(2, 0, 1).to(torch.float32).to(device)

        # 在 GPU 上进行归一化
        data_min = data.min()
        data -= data_min
        data_max = data.max()
        if data_max > 0:
            data /= (data_max + 1e-6)

        # 在 GPU 上应用数据增强
        if self.training and self.transforms:
            data = self.transforms(data, seed=index)

        return data, label

    def __len__(self):
        return len(self.images)

def get_train_loader(dataset, batch_size, num_workers, sampler=None, collate_fn=None):
    """
    创建优化后的数据加载器。
    """
    is_shuffle = True if dataset.training else False
    train_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # 限制为 0，避免多进程 RAM 占用
        drop_last=True,
        shuffle=is_shuffle if sampler is None else False,
        pin_memory=False,  # 禁用 pin_memory，减少 RAM 占用
        sampler=sampler,
        collate_fn=collate_fn
    )
    return train_loader

if __name__ == "__main__":
    root_path = "/home/workingkarl/work/dataset/PLGC_256"
    txt_dir = "/home/workingkarl/work/code/SSTM/dataset"

    train_dataset = PLGC_HSIClassification(
        root=root_path,
        txt_name='PLGC_flod82train.txt',
        transforms='high_aug',
        training=True
    )

    test_dataset = PLGC_HSIClassification(
        root=root_path,
        txt_name='PLGC_flod82test.txt',
        transforms=None,
        training=False
    )

    sample_weights = [1.0 / (train_dataset.labels.count(i) + 1e-6) for i in train_dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = get_train_loader(train_dataset, batch_size=2, num_workers=0, sampler=sampler)
    test_loader = get_train_loader(test_dataset, batch_size=2, num_workers=0)

    print(f"训练集大小：{len(train_dataset)}")
    print(f"测试集大小：{len(test_dataset)}")
    for data, label in train_loader:
        print("训练批次形状：", data.shape, label.shape)
        break
    for data, label in test_loader:
        print("测试批次形状：", data.shape, label.shape)
        break
