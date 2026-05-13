# -*- coding: utf-8 -*-
# @Time    : 2025/06/28
# @Author  : Grok (modified based on Geng Qin's code)
# @File    : train_DGA_SSFT.py
# @Description: Train SSFT model on DGA hyperspectral dataset (60 channels, binary classification, .mat files)

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
from tensorboardX import SummaryWriter
from scipy import stats
import warnings
import psutil
import math
import gc
import traceback
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix, roc_auc_score
from model.FracTrans import get_SFT_Swin as FracTrans
warnings.filterwarnings("ignore")

# Set PyTorch memory allocator configuration to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Import models and dataset
from dataset.DGA_dataset import HSI_Dataset, get_train_loader


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/workingkarl/work/dataset/DGA/HSI', help='Path to the hyperspectral data folder')
parser.add_argument('--csv_dir', type=str, default='/home/workingkarl/work/dataset/DGA/nobingren', help='nobingrenPath to the directory containing fold82train.txt and fold82test.txt')
parser.add_argument('--exp', type=str, default='DGA/SFT11', help='Experiment name  DGA/abloation/math or DGA/SFT1   DGA/ab/k128_3')
parser.add_argument('--snapshot_dir', type=str, default='save_checkpoint_hsi', help='Model checkpoint save path')
parser.add_argument('--model', type=str, default='SSFT_UnBlocksV2', help='Model name CCAET DAAB')
parser.add_argument('--nepochs', type=int, default=200, help='Maximum number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU (reduced to minimize RAM usage)')
parser.add_argument('--accum_steps', type=int, default=1, help='Gradient accumulation steps')
parser.add_argument('--deterministic', type=int, default=1, help='Whether to use deterministic training')
parser.add_argument('--transforms', type=str, default='high_aug', help='Data augmentation type: light_aug, medium_aug, high_aug')
parser.add_argument('--seed', type=int, default=43, help='Random seed')
parser.add_argument('--num_workers', type=int, default=0, help='Number of data loader workers (0 to minimize RAM usage)')
parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes (binary classification)')
parser.add_argument('--fold', type=int, default=1, help='K-fold index for dataset')
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

# Set up asynchronous logging
log_queue = queue.Queue()

def setup_logging(snapshot_path, model_name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
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
    logging.info("CUDA is not available, running on CPU.")
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

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
            log_probs = log_probs * self.weight[targets].unsqueeze(1)
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

def compute_metrics(true_labels, pred_labels, pred_probs, num_classes):
    try:
        overall_acc = 100. * (pred_labels == true_labels).sum() / len(true_labels)
        per_class_acc = [100. * ((true_labels == i) & (pred_labels == i)).sum() / (true_labels == i).sum() if (true_labels == i).sum() > 0 else 0.0 for i in range(num_classes)]
        avg_acc = np.mean(per_class_acc)
        precision = 100. * precision_score(true_labels, pred_labels, average='macro', zero_division=0)
        recall = 100. * recall_score(true_labels, pred_labels, average='macro', zero_division=0)
        kappa = cohen_kappa_score(true_labels, pred_labels)
        f1 = 100. * f1_score(true_labels, pred_labels, average='macro', zero_division=0)
        cm = confusion_matrix(true_labels, pred_labels)
        spe = 100. * cm[0, 0] / (cm[0, 0] + cm[0, 1]) if num_classes == 2 and (cm[0, 0] + cm[0, 1]) > 0 else 0.0
        auc = 100. * roc_auc_score(true_labels, pred_probs[:, 1]) if num_classes == 2 and pred_probs is not None else None
        p_values = [stats.mannwhitneyu(pred_probs[true_labels == i, i], pred_probs[true_labels != i, i], alternative='two-sided')[1] if len(pred_probs[true_labels == i, i]) > 0 and len(pred_probs[true_labels != i, i]) > 0 else 1.0 for i in range(num_classes)]
        p_value = np.mean(p_values) if p_values else None
        return {'accuracy': overall_acc, 'Aver_acc': avg_acc, 'precision': precision, 'recall': recall, 'kappa': kappa, 'f1-score': f1, 'specificity': spe, 'auc': auc, 'per_class_acc': per_class_acc, 'p_value': p_value}
    except Exception as e:
        logging.error(f"Error computing metrics: {str(e)}\n{traceback.format_exc()}")
        return {k: 0 if k != 'p_value' else 1.0 for k in ['accuracy', 'Aver_acc', 'precision', 'recall', 'kappa', 'f1-score', 'specificity', 'auc', 'per_class_acc', 'p_value']}

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
    old_weights = getattr(model, 'class_weights', new_weights)
    new_weights = 0.8 * old_weights + 0.2 * new_weights
    return torch.clamp(new_weights, min=0.5, max=10.0)

def get_network(args):
    model_dict = {
        'FracTrans': lambda: SSFTFEDV6Pis12(in_channels=60, num_classes=args.num_classes),
    }
    if args.model not in model_dict:
        raise NotImplementedError(f"Model {args.model} not implemented.")
    return model_dict[args.model]().to(device)

def get_learning_rate(base_lr, current_iter, warmup_iters, total_iters):
    min_lr = base_lr * 0.01
    if current_iter < warmup_iters:
        return base_lr * current_iter / warmup_iters
    cosine_iters = total_iters - warmup_iters
    current_cosine_iter = current_iter - warmup_iters
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * current_cosine_iter / cosine_iters))

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
        return checkpoint['epoch'] + 1, checkpoint['iter_count'], checkpoint.get('best_val_acc', 0), checkpoint.get('best_test_acc', 0), checkpoint.get('patience_counter', 0)
    except Exception as e:
        logging.error(f"Error loading checkpoint: {str(e)}\n{traceback.format_exc()}")
        raise

def delete_old_checkpoint(directory, current_best_val_path, current_best_test_path):
    for file in os.listdir(directory):
        if file.endswith('.pth') and file not in [os.path.basename(current_best_val_path), os.path.basename(current_best_test_path)]:
            try:
                os.remove(os.path.join(directory, file))
                logging.info(f"Deleted old checkpoint: {file}")
            except Exception as e:
                logging.error(f"Error deleting old checkpoint {file}: {str(e)}")

def train(args, snapshot_path):
    train_transforms = args.transforms
    test_transforms = None

    try:
        train_dataset = HSI_Dataset(root=args.root_path, txt_path=os.path.join(args.csv_dir, 'flod82train.txt'), transforms=train_transforms, training=True)
        test_dataset = HSI_Dataset(root=args.root_path, txt_path=os.path.join(args.csv_dir, 'flod82test.txt'), transforms=test_transforms, training=False)

        class_counts = [train_dataset.labels.count(i) for i in range(args.num_classes)]
        num_samples = len(train_dataset.labels)
        weights = [1.0 / class_counts[label] for label in train_dataset.labels]
        sampler = WeightedRandomSampler(weights, num_samples, replacement=True)

        train_loader = get_train_loader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=sampler)
        test_loader = get_train_loader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        logging.info(f"Training dataset size: {len(train_dataset)}")
        logging.info(f"Test dataset size: {len(test_dataset)}")

        model = get_network(args)
        param_count = sum(p.numel() for p in model.parameters())
        logging.info(f"Model {args.model} loaded with {param_count / 1e6:.2f}M parameters, pe_type: {args.pe_type}")

        optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

        inverse_freq = [1.0 / (count + 1e-6) for count in class_counts]
        total_inverse = sum(inverse_freq)
        class_weights = torch.tensor([freq / total_inverse * 3.0 for freq in inverse_freq], dtype=torch.float32).to(device)
        class_weights = torch.clamp(class_weights, min=0.5, max=10.0)
        model.class_weights = class_weights
        logging.info(f'Initial class distribution: {class_counts}, class weights: {class_weights.tolist()}')

        ce_criterion = LabelSmoothingCrossEntropy(smoothing=0.1, weight=class_weights)
        focal_criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        alpha = 0.7

        logger = SummaryWriter(snapshot_path + '/log')
        total_iters = args.nepochs * len(train_loader)
        warmup_iters = args.warm_up_epoch * len(train_loader)

        start_epoch, iter_count, best_val_acc, best_test_acc, patience_counter = 0, 0, 0, 0, 0
        best_val_path, best_test_path = '', ''
        checkpoint_path = os.path.join(snapshot_path, 'checkpoint.pth')
        if args.resume and os.path.exists(args.resume):
            checkpoint_path = args.resume
        if os.path.exists(checkpoint_path):
            start_epoch, iter_count, best_val_acc, best_test_acc, patience_counter = load_checkpoint(checkpoint_path, model, optimizer)

        for epoch in range(start_epoch, args.nepochs):
            model.train()
            loss_total = ce_loss_total = focal_loss_total = mi_loss_total = 0.0
            train_true, train_pred, train_probs = [], [], []

            train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.nepochs} [Training]", leave=False)
            for batch_idx, batch in enumerate(train_bar):
                try:
                    images, true_labels = batch
                    images = images.to(device=device, dtype=torch.float32, non_blocking=True)
                    true_labels = true_labels.to(device=device, dtype=torch.long, non_blocking=True)

                    if true_labels.min() < 0 or true_labels.max() >= args.num_classes:
                        logging.warning(f"Invalid labels in batch {batch_idx}: min={true_labels.min()}, max={true_labels.max()}")
                        continue

                    outputs = model(images)
                    clear_batch_memory(images=images)
                    # if true_labels.min() < 0 or true_labels.max() >= args.num_classes:
                    #     logging.warning(
                    #         f"Invalid labels in batch {batch_idx}: min={true_labels.min()}, max={true_labels.max()}")
                    #     continue
                    #
                    #     # =========================================================
                    #     # 新增：针对 CCAET 的专属两阶段训练分支
                    #     # =========================================================
                    #     # =========================================================
                    #     # 新增：针对 CCAET 的专属两阶段训练分支 (不再需要写采样代码，直接调)
                    #     # =========================================================
                    #     loss_unsup = torch.tensor(0.0, device=device)
                    #     if args.model == 'CCAET':
                    #         B, C, H, W = images.shape
                    #         # 阶段一：无监督特征学习
                    #         pixels, Z, X_hat = model.forward_stage1(images)
                    #
                    #         # 直接计算，模型内部会自动处理下采样防 OOM 且全图计算 MSE
                    #         loss_unsup = model.compute_total_unsupervised_loss(pixels, Z, X_hat)
                    #
                    #         # 阶段二：使用阶段一截断的特征(Z.detach())进行有监督分类
                    #         outputs = model.forward_stage2(Z.detach(), B, H, W)
                    #         clear_batch_memory(images=images, pixels=pixels, Z=Z, X_hat=X_hat)
                    #     else:
                    #         outputs = model(images)
                    #         clear_batch_memory(images=images)
                        # =========================================================
                    # =========================================================

                    if isinstance(outputs, tuple):
                        logits, aux_logits = outputs
                        ce_loss = ce_criterion(logits, true_labels)
                        focal_loss = focal_criterion(logits, true_labels)
                        aux_ce_losses = [ce_criterion(aux_logit, true_labels) for aux_logit in aux_logits]
                        aux_focal_losses = [focal_criterion(aux_logit, true_labels) for aux_logit in aux_logits]
                        mi_loss = torch.stack([alpha * ce + (1 - alpha) * focal for ce, focal in zip(aux_ce_losses, aux_focal_losses)]).mean()
                    else:
                        ce_loss = ce_criterion(outputs, true_labels)
                        focal_loss = focal_criterion(outputs, true_labels)
                        mi_loss = torch.tensor(0.0, device=device)

                    loss = alpha * ce_loss + (1 - alpha) * focal_loss + mi_loss

                    if torch.isnan(loss) or torch.isinf(loss):
                        logging.warning(f"Loss is NaN or Inf in batch {batch_idx}, skipping")
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
                    clear_batch_memory(outputs=outputs, loss=loss, ce_loss=ce_loss, focal_loss=focal_loss, mi_loss=mi_loss,
                                       logits=logits if 'logits' in locals() else None,
                                       aux_logits=aux_logits if 'aux_logits' in locals() else None)

                    # Monitor memory usage
                    memory = psutil.virtual_memory()
                    if memory.percent > 90:
                        logging.warning(f"High CPU memory usage: {memory.percent}%")
                        clear_batch_memory()
                        break
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated(device) / 1024**3
                        total_gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
                        if gpu_memory > 0.9 * total_gpu_memory:
                            logging.warning(f"High GPU memory usage: {gpu_memory:.2f}/{total_gpu_memory:.2f} GB")
                            clear_batch_memory()
                            break

                except Exception as e:
                    logging.error(f"Error in batch {batch_idx}: {str(e)}\n{traceback.format_exc()}")
                    clear_batch_memory()
                    continue

            train_bar.close()

            train_loss = loss_total / len(train_loader) if len(train_loader) > 0 else 0
            ce_loss_avg = ce_loss_total / len(train_loader) if len(train_loader) > 0 else 0
            focal_loss_avg = focal_loss_total / len(train_loader) if len(train_loader) > 0 else 0
            mi_loss_avg = mi_loss_total / len(train_loader) if len(train_loader) > 0 else 0

            train_log = (f"Epoch {epoch + 1}/{args.nepochs} || Training Loss: {train_loss:.4f} || "
                         f"CE Loss: {ce_loss_avg:.4f} || Focal Loss: {focal_loss_avg:.4f} || MI Loss: {mi_loss_avg:.4f}")
            print(train_log)
            logging.info(train_log)
            logger.add_scalar('Train/Loss', train_loss, epoch)

            if (epoch + 1) % 5 == 0 or epoch == args.nepochs - 1:
                if train_true:
                    train_true = np.concatenate(train_true)
                    train_pred = np.concatenate(train_pred)
                    train_probs = np.concatenate(train_probs)
                    train_metrics = compute_metrics(train_true, train_pred, train_probs, args.num_classes)

                    train_metrics_log = (f"Epoch {epoch + 1}/{args.nepochs} || Training Metrics || "
                                        f"Accuracy: {train_metrics['accuracy']:.2f}% || Average Accuracy: {train_metrics['Aver_acc']:.2f}% || "
                                        f"Precision: {train_metrics['precision']:.2f}% || Sensitivity: {train_metrics['recall']:.2f}% || "
                                        f"Specificity: {train_metrics['specificity']:.2f}% || AUC: {train_metrics['auc']:.2f}% || "
                                        f"Kappa: {train_metrics['kappa']:.4f} || F1: {train_metrics['f1-score']:.2f}% || "
                                        f"p-value: {train_metrics['p_value']:.4f}")
                    print(train_metrics_log)
                    logging.info(train_metrics_log)
                    logger.add_scalar('Train/Accuracy', train_metrics['accuracy'], epoch)
                    logger.add_scalar('Train/Average_Accuracy', train_metrics['Aver_acc'], epoch)
                    logger.add_scalar('Train/Precision', train_metrics['precision'], epoch)
                    logger.add_scalar('Train/Recall', train_metrics['recall'], epoch)
                    logger.add_scalar('Train/Kappa', train_metrics['kappa'], epoch)
                    logger.add_scalar('Train/F1', train_metrics['f1-score'], epoch)
                    logger.add_scalar('Train/Specificity', train_metrics['specificity'], epoch)
                    logger.add_scalar('Train/AUC', train_metrics['auc'], epoch)
                    logger.add_scalar('Train/P_value', train_metrics['p_value'], epoch)

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
                val_loss_total = ce_loss_val = focal_loss_val = mi_loss_val = 0.0
                val_true, val_pred, val_probs = [], [], []

                val_bar = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{args.nepochs} [Validation]", leave=False)
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
                                aux_ce_losses = [ce_criterion(aux_logit, labels) for aux_logit in aux_logits]
                                aux_focal_losses = [focal_criterion(aux_logit, labels) for aux_logit in aux_logits]
                                mi_loss = torch.stack([alpha * ce + (1 - alpha) * focal for ce, focal in zip(aux_ce_losses, aux_focal_losses)]).mean()
                            else:
                                ce_loss = ce_criterion(outputs, labels)
                                focal_loss = focal_criterion(outputs, labels)
                                mi_loss = torch.tensor(0.0, device=device)

                            val_loss = alpha * ce_loss + (1 - alpha) * focal_loss + mi_loss
                            val_loss_total += val_loss.item()
                            ce_loss_val += ce_loss.item()
                            focal_loss_val += focal_loss.item()
                            mi_loss_val += mi_loss.item()

                            main_outputs = outputs[0] if isinstance(outputs, tuple) else outputs
                            _, val_predicted = torch.max(main_outputs, 1)
                            probs = torch.softmax(main_outputs, dim=1).detach()
                            val_true.append(labels.cpu().numpy())
                            val_pred.append(val_predicted.cpu().numpy())
                            val_probs.append(probs.cpu().numpy())

                            val_bar.set_postfix({'Validation Loss': f'{val_loss.item():.4f}'})

                            clear_batch_memory(main_outputs=main_outputs, val_predicted=val_predicted, probs=probs,
                                               outputs=outputs, ce_loss=ce_loss, focal_loss=focal_loss, mi_loss=mi_loss,
                                               val_loss=val_loss,
                                               logits=logits if 'logits' in locals() else None,
                                               aux_logits=aux_logits if 'aux_logits' in locals() else None)

                    except Exception as e:
                        logging.error(f"Error in validation batch: {str(e)}\n{traceback.format_exc()}")
                        clear_batch_memory()
                        continue

                val_bar.close()

                val_true = np.concatenate(val_true)
                val_pred = np.concatenate(val_pred)
                val_probs = np.concatenate(val_probs)
                val_loss = val_loss_total / len(test_loader) if len(test_loader) > 0 else 0
                ce_loss_avg = ce_loss_val / len(test_loader) if len(test_loader) > 0 else 0
                focal_loss_avg = focal_loss_val / len(test_loader) if len(test_loader) > 0 else 0
                mi_loss_avg = mi_loss_val / len(test_loader) if len(test_loader) > 0 else 0
                val_metrics = compute_metrics(val_true, val_pred, val_probs, args.num_classes)

                val_log = (f"Epoch {epoch + 1}/{args.nepochs} || Validation Loss: {val_loss:.4f} || "
                           f"CE Loss: {ce_loss_avg:.4f} || Focal Loss: {focal_loss_avg:.4f} || MI Loss: {mi_loss_avg:.4f} || "
                           f"Accuracy: {val_metrics['accuracy']:.2f}% || Average Accuracy: {val_metrics['Aver_acc']:.2f}% || "
                           f"Precision: {val_metrics['precision']:.2f}% || Sensitivity: {val_metrics['recall']:.2f}% || "
                           f"Specificity: {val_metrics['specificity']:.2f}% || AUC: {val_metrics['auc']:.2f}% || "
                           f"Kappa: {val_metrics['kappa']:.4f} || F1: {val_metrics['f1-score']:.2f}% || "
                           f"p-value: {val_metrics['p_value']:.4f}")
                print("\n" + val_log)
                logging.info(val_log)
                logger.add_scalar('Val/Loss', val_loss, epoch)
                logger.add_scalar('Val/Accuracy', val_metrics['accuracy'], epoch)
                logger.add_scalar('Val/Average_Accuracy', val_metrics['Aver_acc'], epoch)
                logger.add_scalar('Val/Precision', val_metrics['precision'], epoch)
                logger.add_scalar('Val/Recall', val_metrics['recall'], epoch)
                logger.add_scalar('Val/Kappa', val_metrics['kappa'], epoch)
                logger.add_scalar('Val/F1', val_metrics['f1-score'], epoch)
                logger.add_scalar('Val/Specificity', val_metrics['specificity'], epoch)
                logger.add_scalar('Val/AUC', val_metrics['auc'], epoch)
                logger.add_scalar('Val/P_value', val_metrics['p_value'], epoch)

                model_l_savedir = os.path.join(args.snapshot_dir, f'model_{args.model}_fold{args.fold}')
                os.makedirs(model_l_savedir, exist_ok=True)

                if val_metrics['accuracy'] > best_val_acc:
                    old_val_path = best_val_path
                    best_val_acc = val_metrics['accuracy']
                    patience_counter = 0
                    best_val_path = os.path.join(model_l_savedir, f'fold{args.fold}_epoch{epoch}_{args.model}_best_val.pth')
                    torch.save(model.state_dict(), best_val_path)
                    print(f"Best validation model saved, Accuracy: {best_val_acc:.2f}%, Path: {best_val_path}")
                    logging.info(f"Best validation model saved, Accuracy: {best_val_acc:.2f}%, Path: {best_val_path}")
                    if old_val_path and os.path.exists(old_val_path):
                        try:
                            os.remove(old_val_path)
                            logging.info(f"Deleted old validation checkpoint: {old_val_path}")
                        except Exception as e:
                            logging.error(f"Error deleting old validation checkpoint {old_val_path}: {str(e)}")

                if val_metrics['accuracy'] > best_test_acc:
                    old_test_path = best_test_path
                    best_test_acc = val_metrics['accuracy']
                    best_test_path = os.path.join(model_l_savedir, f'fold{args.fold}_epoch{epoch}_{args.model}_best_test.pth')
                    torch.save(model.state_dict(), best_test_path)
                    print(f"Best test model saved, Accuracy: {best_test_acc:.2f}%, p-value: {val_metrics['p_value']:.4f}, Path: {best_test_path}")
                    logging.info(f"Best test model saved, Accuracy: {best_test_acc:.2f}%, p-value: {val_metrics['p_value']:.4f}, Path: {best_test_path}")
                    if old_test_path and os.path.exists(old_test_path):
                        try:
                            os.remove(old_test_path)
                            logging.info(f"Deleted old test checkpoint: {old_test_path}")
                        except Exception as e:
                            logging.error(f"Error deleting old test checkpoint {old_test_path}: {str(e)}")

                delete_old_checkpoint(model_l_savedir, best_val_path, best_test_path)

                # Clear validation metrics memory
                del val_true, val_pred, val_probs
                clear_batch_memory()

            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Epoch {epoch + 1} stopped early due to no improvement for {args.patience} epochs")
                    logging.info(f"Epoch {epoch + 1} stopped early due to no improvement for {args.patience} epochs")
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

        model.eval()
        test_loss_total = test_ce_loss_total = test_focal_loss_total = test_mi_loss_total = 0.0
        test_true, test_pred, test_probs = [], [], []

        test_bar = tqdm(test_loader, desc="Testing", leave=False)
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
                        aux_ce_losses = [ce_criterion(aux_logit, labels) for aux_logit in aux_logits]
                        aux_focal_losses = [focal_criterion(aux_logit, labels) for aux_logit in aux_logits]
                        mi_loss = torch.stack([alpha * ce + (1 - alpha) * focal for ce, focal in zip(aux_ce_losses, aux_focal_losses)]).mean()
                    else:
                        ce_loss = ce_criterion(outputs, labels)
                        focal_loss = focal_criterion(outputs, labels)
                        mi_loss = torch.tensor(0.0, device=device)

                    test_loss = alpha * ce_loss + (1 - alpha) * focal_loss + mi_loss
                    test_loss_total += test_loss.item()
                    test_ce_loss_total += ce_loss.item()
                    test_focal_loss_total += focal_loss.item()
                    test_mi_loss_total += mi_loss.item()

                    main_outputs = outputs[0] if isinstance(outputs, tuple) else outputs
                    _, test_predicted = torch.max(main_outputs, 1)
                    probs = torch.softmax(main_outputs, dim=1).detach()
                    test_true.append(labels.cpu().numpy())
                    test_pred.append(test_predicted.cpu().numpy())
                    test_probs.append(probs.cpu().numpy())

                    test_bar.set_postfix({'Test Loss': f'{test_loss.item():.4f}'})

                    clear_batch_memory(main_outputs=main_outputs, test_predicted=test_predicted, probs=probs,
                                       outputs=outputs, ce_loss=ce_loss, focal_loss=focal_loss, mi_loss=mi_loss,
                                       test_loss=test_loss,
                                       logits=logits if 'logits' in locals() else None,
                                       aux_logits=aux_logits if 'aux_logits' in locals() else None)

            except Exception as e:
                logging.error(f"Error in test batch: {str(e)}\n{traceback.format_exc()}")
                clear_batch_memory()
                continue

        test_bar.close()

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_probs = np.concatenate(test_probs)
        test_loss = test_loss_total / len(test_loader) if len(test_loader) > 0 else 0
        test_ce_loss_avg = test_ce_loss_total / len(test_loader) if len(test_loader) > 0 else 0
        test_focal_loss_avg = test_focal_loss_total / len(test_loader) if len(test_loader) > 0 else 0
        test_mi_loss_avg = test_mi_loss_total / len(test_loader) if len(test_loader) > 0 else 0
        test_metrics = compute_metrics(test_true, test_pred, test_probs, args.num_classes)

        test_log = (f"Test Results || Test Loss: {test_loss:.4f} || "
                    f"CE Loss: {test_ce_loss_avg:.4f} || Focal Loss: {test_focal_loss_avg:.4f} || MI Loss: {test_mi_loss_avg:.4f} || "
                    f"Accuracy: {test_metrics['accuracy']:.2f}% || Average Accuracy: {test_metrics['Aver_acc']:.2f}% || "
                    f"Precision: {test_metrics['precision']:.2f}% || Sensitivity: {test_metrics['recall']:.2f}% || "
                    f"Specificity: {test_metrics['specificity']:.2f}% || AUC: {test_metrics['auc']:.2f}% || "
                    f"Kappa: {test_metrics['kappa']:.4f} || F1: {test_metrics['f1-score']:.2f}% || "
                    f"p-value: {test_metrics['p_value']:.4f}")
        print("\n" + test_log)
        logging.info(test_log)
        logger.add_scalar('Test/Loss', test_loss, args.nepochs - 1)
        logger.add_scalar('Test/Accuracy', test_metrics['accuracy'], args.nepochs - 1)
        logger.add_scalar('Test/Average_Accuracy', test_metrics['Aver_acc'], args.nepochs - 1)
        logger.add_scalar('Test/Precision', test_metrics['precision'], args.nepochs - 1)
        logger.add_scalar('Test/Recall', test_metrics['recall'], args.nepochs - 1)
        logger.add_scalar('Test/Kappa', test_metrics['kappa'], args.nepochs - 1)
        logger.add_scalar('Test/F1', test_metrics['f1-score'], args.nepochs - 1)
        logger.add_scalar('Test/Specificity', test_metrics['specificity'], args.nepochs - 1)
        logger.add_scalar('Test/AUC', test_metrics['auc'], args.nepochs - 1)
        logger.add_scalar('Test/P_value', test_metrics['p_value'], args.nepochs - 1)

        if test_metrics['accuracy'] > best_test_acc:
            old_test_path = best_test_path
            best_test_acc = test_metrics['accuracy']
            model_l_savedir = os.path.join(args.snapshot_dir, f'model_{args.model}_fold{args.fold}')
            os.makedirs(model_l_savedir, exist_ok=True)
            best_test_path = os.path.join(model_l_savedir, f'fold{args.fold}_{args.model}_best_test.pth')
            torch.save(model.state_dict(), best_test_path)
            print(f"Best test model updated, Accuracy: {best_test_acc:.2f}%, p-value: {test_metrics['p_value']:.4f}, Path: {best_test_path}")
            logging.info(f"Best test model updated, Accuracy: {best_test_acc:.2f}%, p-value: {test_metrics['p_value']:.4f}, Path: {best_test_path}")
            if old_test_path and os.path.exists(old_test_path):
                try:
                    os.remove(old_test_path)
                    logging.info(f"Deleted old test checkpoint: {old_test_path}")
                except Exception as e:
                    logging.error(f"Error deleting old test checkpoint {old_test_path}: {str(e)}")

            delete_old_checkpoint(model_l_savedir, best_val_path, best_test_path)

        # Save final model
        final_model_path = os.path.join(model_l_savedir, f'fold{args.fold}_{args.model}_final.pth')
        torch.save(model.state_dict(), final_model_path)
        logging.info(f"Final model saved to {final_model_path}")

        # Clear final memory
        del test_true, test_pred, test_probs
        clear_batch_memory()

        logger.close()
        log_queue.put(None)
        print('Training completed')

    except Exception as e:
        logging.error(f"Training failed: {str(e)}\n{traceback.format_exc()}")
        log_queue.put(None)
        raise

if __name__ == "__main__":
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = os.path.join("./{}_{}".format(args.exp, args.fold), args.model)
    os.makedirs(snapshot_path, exist_ok=True)

    logger, log_thread = setup_logging(snapshot_path, args.model)
    logging.info(str(args))
    try:
        train(args, snapshot_path)
    finally:
        log_queue.put(None)
        log_thread.join()



# -*- coding: utf-8 -*-
# @Time    : 2025/06/28
# @Author  : Grok
# @File    : DGA_dataset_cls.py
# @Description: 数据集加载器，用于高光谱数据（.mat文件）的二分类，从txt文件读取路径

import os
from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils import data
from scipy.io import loadmat
import gc

# 定义全局设备
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def open_file(datapath):
    """
    加载 .mat 文件，优化内存使用。
    - 直接转换为 float32 减少内存占用。
    """
    _, ext = os.path.splitext(datapath)
    ext = ext.lower()
    if ext == '.mat':
        data = loadmat(datapath)
        if 'data' in data:
            data = data['data'].astype(np.float32)  # 优化：立即转换为 float32
        elif 'image' in data:
            data = data['image'].astype(np.float32)
        else:
            raise ValueError(f"未在 .mat 文件中找到有效数据键: {datapath}")
        if data.ndim == 3 and data.shape[2] == 60:
            return data  # [H, W, 60]
        elif data.ndim == 3 and data.shape[0] == 60:
            data = np.transpose(data, (1, 2, 0))  # [60, H, W] -> [H, W, 60]
            return data
        else:
            raise ValueError(f"不支持的 .mat 文件形状: {data.shape}")
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

def gpu_light_aug(data, seed=42):
    """
    轻度数据增强，在 GPU 上执行。
    """
    torch.manual_seed(seed)
    if torch.rand(1, device=device) > 0.5:
        data = torch.flip(data, dims=[1])  # 垂直翻转
    if torch.rand(1, device=device) > 0.5:
        data = torch.rot90(data, k=1, dims=[1, 2])  # 旋转 90 度
    noise = torch.normal(0, 0.01, size=data.shape, device=device)
    data = torch.clamp(data + noise, 0, 1)
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
    noise = torch.normal(0, 0.02, size=data.shape, device=device)
    data = torch.clamp(data + noise, 0, 1)
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
    if torch.rand(1, device=device) > 0.5:
        data = torch.flip(data, dims=[2])  # 水平翻转
    noise = torch.normal(0, 0.01, size=data.shape, device=device)
    data = torch.clamp(data + noise, 0, 1)
    return data

class HSI_Dataset(Dataset):
    def __init__(self, root, txt_path, training=True, transforms=None):
        super(HSI_Dataset, self).__init__()
        self.root = root
        self.txt_path = txt_path
        assert os.path.exists(self.txt_path), f"文件 '{self.txt_path}' 不存在。"

        self.images = []
        self.labels = []

        # 只读取路径和标签，不加载数据
        with open(self.txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split('/')
                    if len(parts) == 2:
                        folder, filename = parts
                        label = 0 if folder == 'N_256' else 1 if folder == 'P_256' else -1
                        if label >= 0:
                            image_path = os.path.join(self.root, folder, f"{filename}.mat")
                            if os.path.exists(image_path):
                                self.images.append(image_path)
                                self.labels.append(label)
                            else:
                                print(f"警告: 文件不存在，跳过: {image_path}")
                        else:
                            print(f"警告: 文件夹名称无效，跳过: {line}")

        if len(self.images) == 0:
            raise ValueError("未找到有效数据，请检查txt文件和图像路径。")

        assert len(self.images) == len(self.labels), "图像和标签数量不匹配"
        self.is_training = training
        self.transforms = transforms

        # 映射增强函数到 GPU
        if self.transforms == 'light_aug':
            self.transforms = gpu_light_aug
        elif self.transforms == 'medium_aug':
            self.transforms = gpu_medium_aug
        elif self.transforms == 'high_aug' or self.transforms == 'strong_aug':
            self.transforms = gpu_high_aug

        # 打印类别分布
        class_counts = [self.labels.count(0), self.labels.count(1)]
        print(f"数据集加载完成: {len(self.images)} 张图像，通道数: 60，类别分布: {class_counts}")

    def __getitem__(self, index):
        # 按需加载单个图像
        try:
            img = open_file(self.images[index])
        except Exception as e:
            print(f"Error loading {self.images[index]}: {str(e)}")
            img = np.zeros((256, 256, 60), dtype=np.float32)

        label = self.labels[index]

        # 转换为 tensor 并立即移至 GPU
        data = torch.from_numpy(img).permute(2, 0, 1).to(device=device, dtype=torch.float32)

        # 在 GPU 上进行归一化
        data_min = data.min()
        data -= data_min
        data_max = data.max()
        if data_max > 0:
            data /= (data_max + 1e-6)

        # 在 GPU 上应用数据增强
        if self.is_training and self.transforms:
            data = self.transforms(data, seed=index)

        # 优化：释放 CPU 内存
        del img
        gc.collect()

        return data, label

    def __len__(self):
        return len(self.images)

def get_train_loader(dataset, batch_size, num_workers, sampler=None, collate_fn=None):
    is_shuffle = True if dataset.is_training else False
    train_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=is_shuffle if sampler is None else False,
        pin_memory=False,
        sampler=sampler,
        collate_fn=collate_fn
    )
    return train_loader

if __name__ == "__main__":
    root_path = "/home/workingkarl/work/shixiong/shixiongcode/dataset(danguanai)/DGA_fenlei"
    txt_dir = "/home/workingkarl/work/myself_code/dataset"

    train_dataset = HSI_Dataset(
        root=root_path,
        txt_path=os.path.join(txt_dir, 'fold82train.txt'),
        transforms='high_aug',
        training=True
    )

    test_dataset = HSI_Dataset(
        root=root_path,
        txt_path=os.path.join(txt_dir, 'fold82test.txt'),
        transforms=None,
        training=False
    )

    sample_weights = [1.0 / (train_dataset.labels.count(i) + 1e-6) for i in train_dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = get_train_loader(train_dataset, batch_size=2, num_workers=0, sampler=sampler)
    test_loader = get_train_loader(test_dataset, batch_size=2, num_workers=0)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    for data, label in train_loader:
        print("训练批次形状:", data.shape, label.shape)
        break
    for data, label in test_loader:
        print("测试批次形状:", data.shape, label.shape)
        break
