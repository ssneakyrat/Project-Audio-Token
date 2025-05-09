"""
Training script for VocalTokenizer codec using SingingVoiceDataset.
Single-GPU/CPU implementation (non-distributed) with TensorBoard visualization.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np
import random
import torchaudio

from model import VocalTokenizer
from loss import VocalTokenizerLoss, AdversarialLoss
from config import CodecConfig

# Import the SingingVoiceDataset and utils from dataset_decoder.py
from dataset_decoder import (
    SingingVoiceDataset, 
    get_dataloader,
    standardized_collate_fn
)


def train_epoch(model, discriminator, optimizer, optimizer_d, data_loader, loss_fn, device):
    """
    Train for one epoch
    """
    model.train()
    if discriminator is not None:
        discriminator.train()
    
    total_loss = 0.0
    
    for batch in tqdm(data_loader, desc="Training"):
        # Move data to device
        audio = batch['audio'].to(device)  # This is the raw waveform
        mel = batch['mel'].to(device)
        f0 = batch['f0'].to(device)
        # Note: SingingVoiceDataset doesn't have a 'voiced' key, 
        # but we can derive it from f0 (non-zero f0 values indicate voiced frames)
        voiced = (f0 > 0).float().to(device)
        
        # Add singer and phone information
        phone_seq_mel = batch['phone_seq_mel'].to(device)
        singer_id = batch['singer_id'].to(device)
        language_id = batch['language_id'].to(device)
        
        # Prepare targets
        target = {
            'waveform': audio,
            'mel': mel,
            'f0': f0,
            'voiced': voiced,
            # Add additional targets from our dataset
            'phone_seq_mel': phone_seq_mel,
            'singer_id': singer_id,
            'language_id': language_id
        }
        
        # Forward pass (using audio as input instead of waveform)
        model_output = model(audio)
        
        # Compute model loss
        losses = loss_fn(model_output, target)
        loss = losses['total']
        
        # Backpropagation for generator
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Train discriminator if available
        if discriminator is not None and optimizer_d is not None and model_output.get('waveform') is not None:
            # Update discriminator
            optimizer_d.zero_grad()
            
            d_loss = discriminator.disc_loss(model_output['waveform'].detach(), audio)
            d_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            
            optimizer_d.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)


def plot_mel_comparison(target_mel, predicted_mel, title=None):
    """
    Create a figure with target and predicted mel spectrograms side by side.
    
    Args:
        target_mel: Target mel spectrogram numpy array
        predicted_mel: Predicted mel spectrogram numpy array
        title: Optional title for the figure
        
    Returns:
        matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    mel_target_aligned = target_mel.transpose(1, 0)

    # Plot target mel
    im = axes[0].imshow(mel_target_aligned, aspect='auto', origin='lower', interpolation='none')
    axes[0].set_title('Target Mel Spectrogram')
    axes[0].set_ylabel('Mel Bins')
    axes[0].set_xlabel('Frames')
    
    # Plot predicted mel
    im = axes[1].imshow(predicted_mel, aspect='auto', origin='lower', interpolation='none')
    axes[1].set_title('Predicted Mel Spectrogram')
    axes[1].set_xlabel('Frames')
    
    # Add a colorbar
    plt.colorbar(im, ax=axes[1])
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    
    return fig


def validate(model, data_loader, loss_fn, device, writer, epoch):
    """
    Validate the model and log mel spectrogram visualizations to TensorBoard
    """
    model.eval()
    total_loss = 0.0
    
    # For visualization, only log a few samples
    max_vis_samples = 4  # Number of samples to visualize
    vis_sample_count = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            # Move data to device - using SingingVoiceDataset keys
            audio = batch['audio'].to(device)
            mel = batch['mel'].to(device)
            f0 = batch['f0'].to(device)
            voiced = (f0 > 0).float().to(device)
            
            # Add singer and phone information
            phone_seq_mel = batch['phone_seq_mel'].to(device)
            singer_id = batch['singer_id'].to(device)
            language_id = batch['language_id'].to(device)
            
            # Prepare targets
            target = {
                'waveform': audio,
                'mel': mel,
                'f0': f0,
                'voiced': voiced,
                # Add additional targets
                'phone_seq_mel': phone_seq_mel,
                'singer_id': singer_id,
                'language_id': language_id
            }
            
            # Forward pass
            model_output = model(audio)
            
            # Compute loss
            losses = loss_fn(model_output, target)
            loss = losses['total']
            
            total_loss += loss.item()
            
            # Log mel spectrogram visualizations
            if vis_sample_count < max_vis_samples and 'mel_output' in model_output:
                # Get predicted and target mel specs
                predicted_mel = model_output['mel_output']
                target_mel = mel
                
                # Log visualizations for selected samples in the batch
                for i in range(min(audio.size(0), max_vis_samples - vis_sample_count)):
                    # Create figure for this sample
                    fig = plot_mel_comparison(
                        target_mel[i].cpu().numpy(),
                        predicted_mel[i].cpu().numpy(),
                        title=f"Mel Spectrogram Sample {vis_sample_count + i + 1}"
                    )
                    
                    # Log figure to TensorBoard
                    writer.add_figure(
                        f'Mel_Comparison/sample_{vis_sample_count + i + 1}',
                        fig,
                        global_step=epoch
                    )
                
                vis_sample_count += min(audio.size(0), max_vis_samples - vis_sample_count)
                
                # If we've logged enough samples, break the inner loop
                if vis_sample_count >= max_vis_samples:
                    break
    
    return total_loss / len(data_loader)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save model checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)


def get_next_experiment_number(base_log_dir):
    """
    Get the next experiment number based on existing experiment directories.
    """
    if not os.path.exists(base_log_dir):
        os.makedirs(base_log_dir, exist_ok=True)
        return 1
        
    # Look for directories named "exp_X" where X is a number
    existing_exps = [d for d in os.listdir(base_log_dir) 
                     if os.path.isdir(os.path.join(base_log_dir, d)) and d.startswith("exp_")]
    
    if not existing_exps:
        return 1
        
    # Extract numbers from directory names
    exp_numbers = [int(exp.split("_")[1]) for exp in existing_exps if len(exp.split("_")) > 1 and exp.split("_")[1].isdigit()]
    
    if not exp_numbers:
        return 1
        
    # Return the next experiment number
    return max(exp_numbers) + 1


def train_model(args):
    """
    Main training function (single device)
    """
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create model
    config = CodecConfig()
    model = VocalTokenizer(config)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Create discriminator for adversarial training
    discriminator = None
    optimizer_d = None
    if config.adversarial_loss_weight > 0:
        discriminator = AdversarialLoss().discriminator
        discriminator = discriminator.to(device)
        optimizer_d = optim.Adam(discriminator.parameters(), lr=config.learning_rate)
    
    # Create loss function
    loss_fn = VocalTokenizerLoss(config)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Create datasets and data loaders using the SingingVoiceDataset
    # Configure the dataset directories
    dataset_dir = args.train_dir
    cache_dir = os.path.join(args.checkpoint_dir, "dataset_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set up appropriate segment length for context window
    context_window_sec = args.context_window_sec
    
    print("Creating datasets...")
    
    # Option 1: Use the get_dataloader function
    if args.use_dataloader_function:
        print("Using get_dataloader helper function...")
        train_loader, val_loader, train_dataset, val_dataset = get_dataloader(
            batch_size=config.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            train_files=args.train_files,
            val_files=args.val_files,
            device=device,
            context_window_sec=context_window_sec,
            seed=args.seed,
            create_val=True
        )
    # Option 2: Create datasets and loaders manually
    else:
        # Create the datasets first
        print("Creating train dataset...")
        train_dataset = SingingVoiceDataset(
            dataset_dir=dataset_dir,
            cache_dir=cache_dir,
            sample_rate=config.sample_rate,
            rebuild_cache=args.rebuild_cache,
            context_window_sec=context_window_sec,
            is_train=True,
            train_files=args.train_files,
            val_files=args.val_files,
            device=device,
            seed=args.seed
        )
        
        print("Creating validation dataset...")
        val_dataset = SingingVoiceDataset(
            dataset_dir=dataset_dir,
            cache_dir=cache_dir,
            sample_rate=config.sample_rate,
            rebuild_cache=False,  # Only rebuild for training if needed
            context_window_sec=context_window_sec,
            is_train=False,
            train_files=args.train_files,
            val_files=args.val_files,
            device=device,
            seed=args.seed
        )
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            collate_fn=standardized_collate_fn
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            collate_fn=standardized_collate_fn
        )
    
    # Create incremental experiment directory for tensorboard logs
    exp_number = get_next_experiment_number(args.log_dir)
    exp_dir = os.path.join(args.log_dir, f"exp_{exp_number}")
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Logging to experiment directory: {exp_dir}")
    
    # Create tensorboard writer with incremental experiment name
    writer = SummaryWriter(log_dir=exp_dir)
    
    # Create experiment-specific checkpoint directory
    exp_checkpoint_dir = os.path.join(args.checkpoint_dir, f"exp_{exp_number}")
    os.makedirs(exp_checkpoint_dir, exist_ok=True)
    print(f"Saving checkpoints to: {exp_checkpoint_dir}")
    
    # Training loop
    best_val_loss = float('inf')
    
    print(f"Starting training for {config.max_epochs} epochs")
    
    for epoch in range(config.max_epochs):
        print(f"Epoch {epoch+1}/{config.max_epochs}")
        
        # Train for one epoch
        train_loss = train_epoch(model, discriminator, optimizer, optimizer_d, train_loader, loss_fn, device)
        
        # Validate - pass the writer and epoch number to log visualizations
        val_loss = validate(model, val_loader, loss_fn, device, writer, epoch)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss, 
                os.path.join(exp_checkpoint_dir, 'best_model.pt')
            )
        
        # Save regular checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(exp_checkpoint_dir, f'model_epoch_{epoch+1}.pt')
            )
    
    # Add experiment config to tensorboard
    config_text = (
        f"Experiment {exp_number}\n"
        f"Train files: {args.train_files}\n"
        f"Val files: {args.val_files}\n"
        f"Context window: {args.context_window_sec} sec\n"
        f"Batch size: {config.batch_size}\n"
        f"Learning rate: {config.learning_rate}\n"
        f"Seed: {args.seed}\n"
    )
    writer.add_text('Experiment/config', config_text)
    writer.close()


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="Train VocalTokenizer codec")
    parser.add_argument('--train_dir', type=str, default="./datasets", help="Directory with training data")
    parser.add_argument('--log_dir', type=str, default='logs', help="Tensorboard log directory")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Checkpoint directory")
    parser.add_argument('--save_every', type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument('--rebuild_cache', action='store_true', help="Rebuild dataset cache")
    parser.add_argument('--train_files', type=int, default=100, help="Number of files to use for training")
    parser.add_argument('--val_files', type=int, default=10, help="Number of files to use for validation")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of worker processes for data loading")
    parser.add_argument('--context_window_sec', type=float, default=4.0, help="Context window size in seconds")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--use_dataloader_function', action='store_true', 
                        help="Use the get_dataloader function instead of creating datasets manually")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Start training
    train_model(args)


if __name__ == "__main__":
    main()