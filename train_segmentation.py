# training
import argparse as arg
import os
import torch
import matplotlib.pyplot as plt
from models import MAE, VesselSegmentationModel, CombinedSegmentationLoss
from Dataset import VesselSegmentationDataset
import numpy as np
import tqdm
from utils import denorm

"""
Complete Segmentation Loss and Metrics
Includes: BCE, Dice, Tversky, Focal, Thin Vessel Weighting, Edge Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# INDIVIDUAL LOSS COMPONENTS
# ============================================================

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    Good for handling class imbalance
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: (B,1,H,W) logits
            target: (B,1,H,W) ground truth vessel mask
        """
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )

        return 1 - dice


class TverskyLoss(nn.Module):
    """
    Tversky Loss - emphasizes recall over precision
    For vessels we want high recall: alpha=0.3, beta=0.7
    This penalizes missing vessels more than false positives
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        TP = (pred * target).sum()              # True positive
        FP = (pred * (1 - target)).sum()        # False positive
        FN = ((1 - pred) * target).sum()        # False negative

        tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )

        return 1 - tversky


class FocalLoss(nn.Module):
    """
    Focal Loss - focuses on hard examples
    Useful for small vessels that are hard to detect
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """
        Args:
            pred: (B,1,H,W) logits
            target: (B,1,H,W) ground truth vessel binary mask
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )

        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)

        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * bce_loss

        return focal_loss.mean()


# ============================================================
# COMBINED LOSS
# ============================================================

class CombinedSegmentationLoss(nn.Module):
    """
    Combined Loss criterion for vessel segmentation

    Includes:
    - BCE: Basic pixel-wise loss
    - Dice: Overlap measure
    - Tversky: Recall-focused (catches thin vessels)
    - Focal: Hard example mining
    - Thin Vessel Weighting: Extra weight on thin vessels
    - Edge Loss: Sharp vessel boundaries
    """
    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        tversky_weight=1.0,
        focal_weight=0.5,
        thin_weight=2.5,
        edge_weight=1.5,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        focal_alpha=0.25,
        focal_gamma=2.0
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.thin_weight = thin_weight
        self.edge_weight = edge_weight
        self.focal_weight = focal_weight

        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # Loss components
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, pred, target):
        """
        Args:
            pred: (B,1,H,W) logits (before sigmoid)
            target: (B,1,H,W) binary mask (0 or 1)

        Returns:
            total_loss: scalar tensor
            losses: dict of individual loss values
        """
        # Ensure target is valid (0 to 1, no NaN/Inf)
        target = torch.clamp(target, 0.0, 1.0)
        pred_sig = torch.sigmoid(pred)

        # ========================================
        # 1. Basic losses
        # ========================================
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        tversky_loss = self.tversky(pred, target)
        focal_loss = self.focal(pred, target)

        # ========================================
        # 2. Thin vessel weighting
        # ========================================
        thickness_map = self._compute_vessel_thickness(target)
        thin_vessel_mask = (thickness_map < 3.0).float()  # Vessels < 3 pixels wide
        weight_map = 1.0 + 3.0 * thin_vessel_mask * target  # 4x weight on thin vessels

        bce_per_pixel = F.binary_cross_entropy(pred_sig, target, reduction='none')
        thin_weighted_bce = (bce_per_pixel * weight_map).mean()

        # ========================================
        # 3. Edge loss
        # ========================================
        edge_loss = self._safe_edge_loss(pred_sig, target)

        # ========================================
        # 4. Total loss
        # ========================================
        total_loss = (
            self.bce_weight * bce_loss +
            self.dice_weight * dice_loss +
            self.tversky_weight * tversky_loss +
            self.thin_weight * thin_weighted_bce +
            self.edge_weight * edge_loss +
            self.focal_weight * focal_loss
        )

        # Return loss breakdown
        losses = {
            'total': float(total_loss.detach().cpu()),
            'bce': float(bce_loss.detach().cpu()),
            'dice': float(dice_loss.detach().cpu()),
            'tversky': float(tversky_loss.detach().cpu()),
            'focal': float(focal_loss.detach().cpu()),
            'thin_weighted': float(thin_weighted_bce.detach().cpu()),
            'edge': float(edge_loss.detach().cpu()),
        }

        return total_loss, losses

    def _compute_vessel_thickness(self, mask):
        """
        Approximate vessel thickness using max pooling

        Args:
            mask: (B,1,H,W) binary mask

        Returns:
            thickness: (B,1,H,W) approximate thickness map
        """
        pooled = F.max_pool2d(mask, kernel_size=5, stride=1, padding=2)
        thickness = pooled * 5.0  # Scale to approximate pixel width
        return thickness

    def _safe_edge_loss(self, pred, target):
        """
        Safe edge loss using Sobel filters
        Creates filters dynamically on correct device

        Args:
            pred: (B,1,H,W) predictions (after sigmoid, 0-1)
            target: (B,1,H,W) ground truth (0-1)

        Returns:
            edge_loss: scalar tensor
        """
        try:
            device = pred.device
            dtype = pred.dtype

            # Create Sobel filters on correct device
            sobel_x = torch.tensor(
                [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                dtype=dtype,
                device=device
            ).view(1, 1, 3, 3)

            sobel_y = torch.tensor(
                [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                dtype=dtype,
                device=device
            ).view(1, 1, 3, 3)

            # Clamp inputs to prevent numerical issues
            pred = torch.clamp(pred, 0.0, 1.0)
            target = torch.clamp(target, 0.0, 1.0)

            # Compute edges for prediction
            pred_edge_x = F.conv2d(pred, sobel_x, padding=1)
            pred_edge_y = F.conv2d(pred, sobel_y, padding=1)
            pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-6)

            # Compute edges for target
            target_edge_x = F.conv2d(target, sobel_x, padding=1)
            target_edge_y = F.conv2d(target, sobel_y, padding=1)
            target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-6)

            # Clamp edge magnitudes to prevent outliers
            pred_edge = torch.clamp(pred_edge, 0, 10)
            target_edge = torch.clamp(target_edge, 0, 10)

            # MSE loss between edges
            return F.mse_loss(pred_edge, target_edge)

        except Exception as e:
            # If edge loss fails, return zero and print warning
            print(f"⚠️ Edge loss failed: {e}. Continuing without it.")
            return torch.tensor(0.0, device=pred.device, requires_grad=True)


# ============================================================
# METRICS CALCULATION
# ============================================================

@torch.no_grad()
def calculate_metrics(pred, target, threshold=0.64):
    """
    Calculate segmentation metrics

    Args:
        pred: (B,1,H,W) logits or probabilities
        target: (B,1,H,W) ground truth vessel mask
        threshold: threshold for converting to binary

    Returns:
        dict with keys: accuracy, precision, recall, f1, iou, dice
    """
    # Convert to probabilities if needed
    if pred.max() > 1.0 or pred.min() < 0.0:
        pred = torch.sigmoid(pred)

    # Binarize prediction
    pred_binary = (pred > threshold).float()

    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)

    # Calculate confusion matrix components
    TP = (pred_flat * target_flat).sum()
    TN = ((1 - pred_flat) * (1 - target_flat)).sum()
    FP = (pred_flat * (1 - target_flat)).sum()
    FN = ((1 - pred_flat) * target_flat).sum()

    # Calculate metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)

    # Dice coefficient (alternative calculation)
    intersection = (pred_binary * target).sum()
    dice = (2 * intersection) / (pred_binary.sum() + target.sum() + 1e-8)

    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'iou': iou.item(),
        'dice': dice.item()
    }


# ============================================================
# BATCH METRICS (for efficiency)
# ============================================================

@torch.no_grad()
def calculate_batch_metrics(pred, target, threshold=0.64):
    """
    Calculate metrics for entire batch at once (more efficient)

    Args:
        pred: (B,1,H,W) predictions
        target: (B,1,H,W) ground truth
        threshold: binarization threshold

    Returns:
        dict with averaged metrics across batch
    """
    batch_size = pred.shape[0]

    # Accumulate metrics
    total_metrics = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'iou': 0.0,
        'dice': 0.0
    }

    # Calculate per-sample and accumulate
    for i in range(batch_size):
        metrics = calculate_metrics(pred[i:i+1], target[i:i+1], threshold)
        for k in total_metrics:
            total_metrics[k] += metrics[k]

    # Average across batch
    avg_metrics = {k: v / batch_size for k, v in total_metrics.items()}

    return avg_metrics


# ============================================================
# OPTIMAL THRESHOLD SEARCH
# ============================================================

@torch.no_grad()
def find_optimal_threshold(pred, target, thresholds=None):
    """
    Find optimal threshold that maximizes F1 score

    Args:
        pred: (B,1,H,W) predictions (probabilities)
        target: (B,1,H,W) ground truth
        thresholds: list of thresholds to test (default: 0.3 to 0.7 in steps of 0.05)

    Returns:
        best_threshold: float
        best_f1: float
        all_results: list of dicts with threshold and metrics
    """
    if thresholds is None:
        thresholds = [t / 100.0 for t in range(30, 71, 5)]  # 0.30 to 0.70

    best_f1 = 0.0
    best_threshold = 0.5
    all_results = []

    for thresh in thresholds:
        metrics = calculate_metrics(pred, target, threshold=thresh)

        all_results.append({
            'threshold': thresh,
            **metrics
        })

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = thresh

    return best_threshold, best_f1, all_results


# ============================================================
# TESTING
# ============================================================

if __name__ == '__main__':
    """Test all components"""

    print("="*60)
    print("TESTING LOSS AND METRICS")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create criterion
    criterion = CombinedSegmentationLoss(
        bce_weight=0.5,
        dice_weight=1.0,
        tversky_weight=2.0,
        focal_weight=1.0,
        thin_weight=2.5,
        edge_weight=1.5,
    ).to(device)

    # Test data
    batch_size = 2
    pred = torch.randn(batch_size, 1, 256, 256, device=device)
    target = torch.rand(batch_size, 1, 256, 256, device=device)
    target = (target > 0.5).float()  # Binary target

    print(f"\nTest data shapes:")
    print(f"  pred: {pred.shape}")
    print(f"  target: {target.shape}")

    # Test loss
    print("\n" + "-"*60)
    print("Testing Loss Calculation")
    print("-"*60)

    try:
        loss, loss_dict = criterion(pred, target)
        print("✓ Loss calculation successful!")
        print(f"\nTotal Loss: {loss_dict['total']:.6f}")
        print("\nLoss Breakdown:")
        for k, v in loss_dict.items():
            if k != 'total':
                print(f"  {k:20s}: {v:.6f}")

        if loss_dict['edge'] > 0:
            print("\n✓ Edge loss is working!")
        else:
            print("\n⚠️ Edge loss returned 0")

    except Exception as e:
        print(f"❌ Loss calculation failed: {e}")
        import traceback
        traceback.print_exc()

    # Test metrics
    print("\n" + "-"*60)
    print("Testing Metrics Calculation")
    print("-"*60)

    try:
        metrics = calculate_metrics(pred, target, threshold=0.5)
        print("✓ Metrics calculation successful!")
        print("\nMetrics:")
        for k, v in metrics.items():
            print(f"  {k:15s}: {v:.4f}")

    except Exception as e:
        print(f"❌ Metrics calculation failed: {e}")
        import traceback
        traceback.print_exc()

    # Test optimal threshold
    print("\n" + "-"*60)
    print("Testing Optimal Threshold Search")
    print("-"*60)

    try:
        best_thresh, best_f1, results = find_optimal_threshold(
            torch.sigmoid(pred), target
        )
        print(f"✓ Threshold search successful!")
        print(f"\nBest threshold: {best_thresh:.2f}")
        print(f"Best F1: {best_f1:.4f}")

    except Exception as e:
        print(f"❌ Threshold search failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)

# ============================================================
# training modules

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
  model.train()

  total_losses = {
      'total': 0, 'bce': 0, 'dice': 0, 'tversky': 0,
      'focal': 0, 'thin_weighted': 0, 'edge': 0
  }
  all_metrics = {
      'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0, 'dice': 0
  }

  num_batches = 0

  pbar = tqdm.tqdm(dataloader, desc='Epoch {}'.format(epoch))
  for images, masks in pbar:
    images = images.to(device)
    masks = masks.to(device)

    outputs = model(images)
    loss, loss_dict = criterion(outputs, masks)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    for k,v in loss_dict.items():
      total_losses[k] += v

    with torch.no_grad():
      metrics = calculate_metrics(outputs, masks, threshold)
      for k in all_metrics:
        all_metrics[k] += metrics[k]

    num_batches += 1



    pbar.set_postfix({
        'loss': f'{loss_dict['total']:.4f}',
        'f1': f'{metrics["f1"]:.4f}',
        'thin': f'{loss_dict['thin_weighted']:.4f}'
    })


  # average metrics
  results = {
      'losses': {k: v / num_batches for k,v in total_losses.items()},
      'metrics': {k: v / num_batches for k,v in all_metrics.items()}
  }

  return results

@torch.no_grad()
def validate(model, dataloader, criterion, device, threshold):
  model.eval()

  total_losses = {
      'total': 0, 'bce': 0, 'dice': 0, 'tversky': 0,
      'focal': 0, 'thin_weighted': 0, 'edge': 0
  }
  all_metrics = {
      'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0, 'dice': 0
  }

  num_batches = 0

  for images, masks in tqdm.tqdm(dataloader, desc='Validating'):
    images = images.to(device)
    masks = masks.to(device)

    if images.ndim == 3:
      images = images.unsqueeze(1)
    if masks.ndim == 3:
      masks = masks.unsqueeze(1)

    outputs = model(images)
    loss, loss_dict = criterion(outputs, masks)

    for k,v in loss_dict.items():
      total_losses[k] += v

    metrics = calculate_metrics(outputs, masks, threshold=threshold)
    for k in all_metrics:
      all_metrics[k] += metrics[k]

    num_batches += 1


  results = {
      'losses': {k: v / num_batches for k,v in total_losses.items()},
      'metrics': {k: v / num_batches for k,v in all_metrics.items()}
  }

  return results

@torch.no_grad()
def visualize_predictions(model, dataloader, device, save_path, threshold, num_samples=4):
  model.eval()

  images,masks = next(iter(dataloader))
  images = images[:num_samples].to(device)
  masks = masks[:num_samples].to(device)

  outputs = model(images)
  preds = (torch.sigmoid(outputs) > threshold).float()

  fig, axes = plt.subplots(num_samples, 4, figsize=(16,4 * num_samples))

  for i in range(num_samples):

    img = images[i, 0].cpu()
    img = np.clip(denorm(img),0,1)
    axes[i,0].imshow(img, cmap='gray')
    axes[i,0].set_title('Input image')
    axes[i,0].axis('off')

    axes[i,1].imshow(np.clip(masks[i,0].cpu(), 0,1), cmap='gray')
    axes[i,1].set_title('Ground truth')
    axes[i,1].axis('off')

    axes[i,2].imshow(preds[i,0].cpu(), cmap='gray')
    axes[i,2].set_title('Prediction')
    axes[i,2].axis('off')

    gt = masks[i,0].cpu().numpy()
    pred = preds[i,0].cpu().numpy()

    overlay = np.zeros((*gt.shape, 3))
    overlay[...,1] = (gt * pred) # True positives (green)
    overlay[...,0] = (pred * (1-gt)) # False positives (red)
    overlay[...,2] = ((1-pred) * gt) # False negatives (blue)

    axes[i,3].imshow(overlay)
    axes[i,3].set_title('Overlay (G=TP, R=FP, B=FN)')
    axes[i,3].axis('off')

  plt.tight_layout()
  plt.savefig(save_path, dpi=150, bbox_inches='tight')
  plt.close()

  print(f'Predictions saved to {save_path}')


if __name__ == '__main__':
    parser = arg.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--freeze_encoder', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--bce_weight', type=float, default=0.5)
    parser.add_argument('--dice_weight', type=float, default=1.0)
    parser.add_argument('--tversky_weight', type=float, default=2.0)
    parser.add_argument('--focal_weight', type=float, default=1.0),
    parser.add_argument('--thin_weight', type=float, default=2.5)
    parser.add_argument('--edge_weight', type=float, default=1.5)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--encoder_weights_dir', type=str, default='best_model.pth')
    parser.add_argument('--save_dir', type=str, default='segmentation_results')
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--save_vis', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')

    args, _ = parser.parse_known_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    encoder_weights_dir = os.path.abspath(args.encoder_weights_dir)
    
    save_predictions_dir = os.path.join(args.save_dir, 'visualizations')
    save_checkpoints_dir = os.path.join(args.save_dir, 'checkpoints')
    save_model_dir = os.path.join(args.save_dir, 'models')
    save_graphs_dir = os.path.join(args.save_dir, 'graphs')
    os.makedirs(save_graphs_dir, exist_ok=True)
    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(save_checkpoints_dir, exist_ok=True)
    os.makedirs(save_predictions_dir, exist_ok=True)
    print(f'Saving graphs to {save_graphs_dir}')
    print(f'Saving predictions to {save_predictions_dir}')
    print(f'Saving models to {save_model_dir}')
    print(f'Saving checkpoints to {save_checkpoints_dir}')

    print('\n' + '='*60)
    print('Loading Pretrained MAE')
    print('='*60)

    mae = MAE(
        512,
        16,
        1,
        768,
        8,
        8,
        512,
        4,
        8,
        mask_ratio=0.5
        )
    state = torch.load(encoder_weights_dir, weights_only=False,map_location=device)
    mae.load_state_dict(state['model_state_dict'])
    mae.to(device)

    train_loss_history = state['history']
    last_train_loss = train_loss_history['train_loss'][-1]
    last_mse_loss = train_loss_history['train_mse'][-1]
    last_edge_loss = train_loss_history['train_edge'][-1]

    print(f'✓ Loaded MAE from epoch {state['epoch']}')
    print(f' Final pretrain loss: {last_train_loss:.4f}')
    print(f' Final pretrain MSE loss: {last_mse_loss:.4f}')
    print(f' Final pretrain edge loss: {last_edge_loss:.4f}')

    print('\n' + '='*60)
    print('Creating Segmentation Model')
    print('='*60)

    model = VesselSegmentationModel(
        mae_model = mae,
        freeze_encoder = args.freeze_encoder,
        dropout = args.dropout
    )

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Total parameters: {total_params/1e6:.1f}M')
    print(f'Total trainable paraemeters: {trainable_params/1e6:.1f}M')

    print('\n' + '='*60)
    print('Loading Data')
    print('='*60)

    train_data_loader = torch.utils.data.DataLoader(VesselSegmentationDataset('train'), batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(VesselSegmentationDataset('test'), batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    print('\n Testing Train Dataset')
    image, mask = next(iter(train_data_loader))
    print(f' Batch: Image {image.shape}, Mask {mask.shape}')

    print('\n Testing Test Dataset')
    image, mask = next(iter(test_data_loader))
    print(f' Batch: Image {image.shape}, Mask {mask.shape}')

    print(f'\n ✓ Loaded {len(train_data_loader)} batches of training images.')
    print(f'✓ Loaded {len(test_data_loader)} batches of testing images.')

    print('\n' + '='*60)
    print('Training')
    print('='*60)

    criterion = CombinedSegmentationLoss(
    bce_weight=0.5,
    dice_weight=1.0,
    tversky_weight=2.5,
    focal_weight=1.0,
    thin_weight=2.5,
    edge_weight=0.0,
    tversky_alpha=0.2,
    tversky_beta=0.8,
    ).to(device)


    criterion = criterion.to(device)

    threshold = args.threshold

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10,
        min_lr=1e-6
        )

    print('✓ Loss function: Combined (BCE + Dice + Tversky + Focal)')
    print(f'✓ Optimizer: AdamW (lr={args.learning_rate})')
    print('✓ Scheduler: ReduceLROnPlateau')

    best_f1 = 0
    history = {'train': [], 'val': [], 'lr': []}

    for epoch in range(1, args.epochs + 1):
      print(f'\n Epoch {epoch}/{args.epochs}')
      print('-'*60)

      train_results = train_epoch(
          model, train_data_loader, optimizer, criterion, device, epoch
      )

      print(f"\nTrain Results:")
      print(f"  Loss: {train_results['losses']['total']:.4f}")
      print(f"    ├─ BCE: {train_results['losses']['bce']:.4f}")
      print(f"    ├─ Dice: {train_results['losses']['dice']:.4f}")
      print(f"    ├─ Tversky: {train_results['losses']['tversky']:.4f}")
      print(f"    ├─ Thin Weighted: {train_results['losses']['thin_weighted']:.4f} ← Key!")
      print(f"    ├─ Edge: {train_results['losses']['edge']:.4f}")
      print(f"    └─ Focal: {train_results['losses']['focal']:.4f}")
      print(f"  F1: {train_results['metrics']['f1']:.4f} | "
            f"Precision: {train_results['metrics']['precision']:.4f} | "
            f"Recall: {train_results['metrics']['recall']:.4f}")
      history['train'].append(train_results)

      val_results = validate(model, test_data_loader, criterion, device, threshold)
      history['val'].append(val_results)

      print(f"\nValidation Results:")
      print(f"  Loss: {val_results['losses']['total']:.4f}")
      print(f"  F1: {val_results['metrics']['f1']:.4f} | "
            f"Precision: {val_results['metrics']['precision']:.4f} | "
            f"Recall: {val_results['metrics']['recall']:.4f}")

        # Save best model
      if val_results['metrics']['f1'] > best_f1:
          best_f1 = val_results['metrics']['f1']
          torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'f1': val_results['metrics']['f1'],
              'iou': val_results['metrics']['iou'],
              'dice': val_results['metrics']['dice'],
          }, os.path.join(save_model_dir, 'best_model.pth'))
          print(f'✓ Best model saved (F1: {best_f1:.4f})')

      # Update learning rate
      scheduler.step(val_results['metrics']['f1'])
      current_lr = optimizer.param_groups[0]['lr']
      history['lr'].append(current_lr)
      print(f'✓ Learning rate updated to {current_lr:.2e}')

      # Visualize
      if epoch % args.save_vis == 0:
          vis_path = os.path.join(save_predictions_dir, f'predictions_epoch_{epoch:03d}.png')
          visualize_predictions(model, test_data_loader, device, vis_path, threshold)

      # Save checkpoint
      if epoch % args.save_freq == 0:
          torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'history': history,
          }, os.path.join(save_checkpoints_dir, f'checkpoint_epoch_{epoch:03d}.pth'))


    plt.figure(figsize=(16,8))
    plt.plot(history['lr'], linewidth=2)
    plt.title('Learning Rate History')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_graphs_dir, 'lr_schedule.png'))

    print('\n' + '='*60)
    print(f'Training Complete! Best F1: {best_f1:.4f}')
    print('='*60)