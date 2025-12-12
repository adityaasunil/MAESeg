import argparse as arg
import os
from pathlib import Path
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
from Dataset import UnlabelledTrainingDataset
from models import MAE
from models import VesselFocusedMAELoss
from utils import PROJECT_ROOT

if __name__ == '__main__':
  parser = arg.ArgumentParser()
  parser.add_argument('--img_size', type=int, default=384)
  parser.add_argument('--patch_size', type=int, default=16)
  parser.add_argument('--masking_ratio', type=float, default=0.5)
  parser.add_argument('--edge_weight',type=float,default=0.2)
  parser.add_argument('--weight_decay', type=float, default=0.5)
  parser.add_argument('--min_batch_size', type=int, default=16)
  parser.add_argument('--max_batch_size', type=int, default=16)
  parser.add_argument('--num_workers', type=int, default=4)
  parser.add_argument('--learning_rate', type=float, default=1.5e-4)
  parser.add_argument('--warmup_epochs', type=int, default=10)
  parser.add_argument('--maximum_epochs', type=int, default=180)
  parser.add_argument('--encoder_save_name', type=str, default='best_encoder.pth')
  parser.add_argument('--save_dir',type=str,default='ckpoints')
  parser.add_argument('--model_save_name', type=str, default='best_model.pth')
  parser.add_argument('--device', type=str, default='cuda')
  parser.add_argument('--seed', type=int, default=42)

  args, _ = parser.parse_known_args()

  device = torch.device(args.device)
  print('Using device: {}'.format(device))

  scaler = torch.amp.GradScaler('cuda')


  save_dir = os.path.join(PROJECT_ROOT, args.save_dir)
  save_curve_dir = os.path.join(PROJECT_ROOT, 'visualizations/curves')
  save_img_dir = os.path.join(PROJECT_ROOT, 'visualizations/reconstructions')
  os.makedirs(save_img_dir, exist_ok=True)
  os.makedirs(save_curve_dir, exist_ok=True)
  os.makedirs(save_dir, exist_ok=True)
  os.makedirs(os.path.join(save_dir, 'visualizations'), exist_ok=True)

  print('Loading Data....')
  train_data_loader = torch.utils.data.DataLoader(UnlabelledTrainingDataset(), batch_size=min(args.min_batch_size, args.max_batch_size), shuffle=True, num_workers=args.num_workers,pin_memory=True)

  print('Train samples: {}'.format(len(UnlabelledTrainingDataset())))

  print('Creating Model.....')
  model = MAE(
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
  ).to(device)

  model_parameters = sum(p.numel() for p in model.parameters()) / 1e6
  print('model parameters: {:.4f}M'.format(model_parameters))
  print('Masking ratio: {}'.format(args.masking_ratio))

  criterion = VesselFocusedMAELoss(args.edge_weight).to(device)

  optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9,0.95))

  def lr_lambda(epoch):
    if epoch < args.warmup_epochs:
      return epoch / args.maximum_epochs
    else:
      return 0.5 * (1 + np.cos(np.pi * (epoch - args.warmup_epochs) / (args.maximum_epochs - args.warmup_epochs)))

  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


  def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_mse = 0
    total_edge = 0

    pbar = tqdm.tqdm(dataloader, desc=f'Epoch {epoch+1}')
    for batch_idx, images in enumerate(pbar):
        images = images.to(device)

        # Forward pass
        loss, pred, mask = model(images)

        pred_imgs = model.unpatchify(pred)

        # Patchify target
        target = model.patchify(images)

        # Compute loss only on masked patches
        loss, mse_loss, edge_loss = criterion(pred_imgs, images)

        # Apply mask to loss
        mask_expanded = mask.unsqueeze(-1).repeat(1, 1, target.shape[-1])
        loss_masked = (loss * mask_expanded).sum() / mask_expanded.sum()

        # Backward pass
        optimizer.zero_grad()
        loss_masked.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track metrics
        total_loss += loss_masked.item()
        total_mse += mse_loss.item()
        total_edge += edge_loss.item()

        pbar.set_postfix({
            'loss': f'{loss_masked.item():.4f}',
            'mse': f'{mse_loss.item():.4f}',
            'edge': f'{edge_loss.item():.4f}'
        })

    return {
        'loss': total_loss / len(dataloader),
        'mse': total_mse / len(dataloader),
        'edge': total_edge / len(dataloader)
    }


  def plot_training_curves(history, save_name):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = ['loss', 'mse', 'edge']
    titles = ['Total Loss', 'MSE Loss', 'Edge Loss']

    for ax, metric, title in zip(axes, metrics, titles):
        ax.plot(history[f'train_{metric}'], label='Train', linewidth=2)
        if f'val_{metric}' in history:
            ax.plot(history[f'val_{metric}'], label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_curve_dir, f'{save_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Training curves saved to {save_curve_dir}')

  @torch.no_grad()
  def visualize_reconstruction(model, dataloader, device, save_path, num_samples=4):
      """Visualize reconstructions"""
      model.eval()

      images = next(iter(dataloader))[:num_samples].to(device)
      loss, pred, mask = model(images)

      # Unpatchify
      pred_imgs = model.unpatchify(pred)
      mask_imgs = mask.unsqueeze(-1).repeat(1, 1, model.patch_size ** 2)
      mask_imgs = model.unpatchify(mask_imgs)

      # Create visualization
      fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

      for i in range(num_samples):
          # Original
          axes[i, 0].imshow(images[i, 0].cpu(), cmap='gray')
          axes[i, 0].set_title('Original')
          axes[i, 0].axis('off')

          # Masked (visible patches)
          masked = images[i] * (1 - mask_imgs[i])
          axes[i, 1].imshow(masked[0].cpu(), cmap='gray')
          axes[i, 1].set_title(f'Visible ({int((1-model.mask_ratio)*100)}%)')
          axes[i, 1].axis('off')

          # Reconstruction
          axes[i, 2].imshow(pred_imgs[i, 0].cpu(), cmap='gray')
          axes[i, 2].set_title('Reconstructed')
          axes[i, 2].axis('off')

          # Difference
          diff = torch.abs(images[i] - pred_imgs[i])
          axes[i, 3].imshow(diff[0].cpu(), cmap='hot')
          axes[i, 3].set_title('Difference')
          axes[i, 3].axis('off')

      plt.tight_layout()
      plt.savefig(os.path.join(save_img_dir, f'epoch_{epoch}.png'), dpi=150, bbox_inches='tight')
      plt.close()
      print(f'Visualization saved to {save_img_dir}')

  print('Starting training...')
  best_val_loss = float('inf')
  history={'train_loss': [], 'train_mse': [], 'train_edge': []}

  for epoch in range(0, args.maximum_epochs + 1):
    print(f'\n{'='*60}')
    print('Epoch {}/{}'.format(epoch+1,args.maximum_epochs))
    print(f'{'='*60}')

    train_metrics = train_epoch(model, train_data_loader, optimizer,criterion, device, epoch)

    history['train_loss'].append(train_metrics['loss'])
    history['train_mse'].append(train_metrics['mse'])
    history['train_edge'].append(train_metrics['edge'])

    print(f'Train loss: {train_metrics["loss"]:.4f}')
    print(f'Train MSE: {train_metrics["mse"]:.4f}')
    print(f'Train edge: {train_metrics["edge"]:.4f}')

    scheduler.step()
    print(f'Learning rate: {scheduler.get_last_lr()[0]:.6f}')

    if epoch % 5 == 0:
      vis_path = save_dir / 'visualizations' / f'epoch_{epoch}.png'
      visualize_reconstruction(model, train_data_loader, device,vis_path)

    if epoch % 10 == 0:
      torch.save({
          epoch: epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'scheduler_state_dict': scheduler.state_dict(),
          'history': history,
      },os.path.join(save_dir , f'checkpoint_epoch_{epoch}.pth'))
      print(f'✓ Checkpoint saved')

    if train_metrics['loss'] < best_val_loss:
      best_val_loss = train_metrics['loss']
      torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
      }, os.path.join(save_dir, 'final_model.pth'))
      print('\n✓ Training complete!')

      plot_training_curves(history, 'training_curves.png')
      print('✓ Successfully saved training curves')