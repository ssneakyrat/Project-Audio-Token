"""
Script to tokenize a singing voice dataset using a trained VocalTokenizer.

This converts raw audio to discrete tokens to prepare for training the token prediction model.
"""

import os
import argparse
import torch
import torchaudio
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

from model import VocalTokenizer
from config import CodecConfig


def load_model(checkpoint_path):
    """
    Load a trained VocalTokenizer model
    """
    config = CodecConfig()
    model = VocalTokenizer(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to eval mode
    model.eval()
    
    return model


def process_audio_file(file_path, model, device, sample_rate=22050, chunk_size=22050*10):
    """
    Process a single audio file and extract tokens
    
    Args:
        file_path: path to audio file
        model: VocalTokenizer model
        device: torch device
        sample_rate: target sample rate
        chunk_size: process audio in chunks of this size
    
    Returns:
        Dictionary with tokens and metadata
    """
    # Load audio
    waveform, sr = torchaudio.load(file_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    
    # Move to device
    waveform = waveform.to(device)
    
    # Process in chunks to avoid OOM
    all_indices = []
    all_f0 = []
    all_voiced = []
    
    for i in range(0, waveform.shape[1], chunk_size):
        chunk = waveform[:, i:i+chunk_size]
        
        # Pad last chunk if needed
        if chunk.shape[1] < chunk_size:
            padding = chunk_size - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (0, padding))
        
        # Encode
        with torch.no_grad():
            indices, f0, voiced = model.encode(chunk)
        
        # Store results
        if isinstance(indices, list):
            # For ResidualVQ, store indices from each codebook
            for j, codebook_indices in enumerate(indices):
                if len(all_indices) <= j:
                    all_indices.append([])
                all_indices[j].append(codebook_indices.cpu().numpy())
        else:
            # For other VQ types
            all_indices.append(indices.cpu().numpy())
        
        all_f0.append(f0.cpu().numpy())
        all_voiced.append(voiced.cpu().numpy())
    
    # Concatenate chunks
    if isinstance(all_indices[0], list):
        # For ResidualVQ
        concatenated_indices = []
        for codebook_indices in zip(*all_indices):
            concatenated_indices.append(np.concatenate(codebook_indices, axis=1))
    else:
        # For other VQ types
        concatenated_indices = np.concatenate(all_indices, axis=1)
    
    concatenated_f0 = np.concatenate(all_f0, axis=1)
    concatenated_voiced = np.concatenate(all_voiced, axis=1)
    
    # Create metadata
    audio_info = {
        'sample_rate': sample_rate,
        'duration': waveform.shape[1] / sample_rate,
        'n_frames': concatenated_f0.shape[1],
        'hop_length': model.encoder.hop_length  # Assuming this attribute exists
    }
    
    return {
        'indices': concatenated_indices,
        'f0': concatenated_f0,
        'voiced': concatenated_voiced,
        'audio_info': audio_info
    }


def tokenize_dataset(input_dir, output_dir, checkpoint_path, batch_size=16):
    """
    Tokenize an entire dataset
    
    Args:
        input_dir: directory with audio files
        output_dir: directory to save tokenized files
        checkpoint_path: path to model checkpoint
        batch_size: batch size for processing
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path)
    model = model.to(device)
    
    # Get list of audio files
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files.")
    
    # Process files
    for file_path in tqdm(audio_files, desc="Tokenizing dataset"):
        # Process file
        result = process_audio_file(file_path, model, device)
        
        # Create output path (maintain directory structure)
        rel_path = os.path.relpath(file_path, input_dir)
        output_path = os.path.join(output_dir, Path(rel_path).stem + '.npz')
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save tokens
        np.savez_compressed(
            output_path,
            indices=result['indices'],
            f0=result['f0'],
            voiced=result['voiced'],
            audio_info=json.dumps(result['audio_info'])
        )
    
    print(f"Tokenization complete. Tokenized files saved to {output_dir}")


def verify_tokens(tokens_path, checkpoint_path, output_dir):
    """
    Verify tokenization by decoding tokens back to audio
    
    Args:
        tokens_path: path to tokenized file (.npz)
        checkpoint_path: path to model checkpoint
        output_dir: directory to save reconstructed audio
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path)
    model = model.to(device)
    
    # Load tokens
    data = np.load(tokens_path, allow_pickle=True)
    indices = data['indices']
    f0 = torch.tensor(data['f0']).to(device)
    voiced = torch.tensor(data['voiced']).to(device)
    audio_info = json.loads(str(data['audio_info']))
    
    # Convert indices to tensor
    if isinstance(indices, list):
        # For ResidualVQ
        indices = [torch.tensor(idx).to(device) for idx in indices]
    else:
        # For other VQ types
        indices = torch.tensor(indices).to(device)
    
    # Decode
    with torch.no_grad():
        output = model.decode(indices, f0, voiced)
    
    # Save audio
    output_path = os.path.join(output_dir, Path(tokens_path).stem + '_reconstructed.wav')
    
    if output.dim() == 2:  # [batch, samples]
        output = output[0]  # Get first item in batch
    
    torchaudio.save(
        output_path,
        output.cpu().unsqueeze(0),
        sample_rate=audio_info['sample_rate']
    )
    
    print(f"Reconstructed audio saved to {output_path}")


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="Tokenize singing voice dataset")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory with audio files")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save tokenized files")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--verify', action='store_true', help="Verify tokenization by reconstructing audio")
    parser.add_argument('--verify_file', type=str, help="Specific file to verify (if --verify is set)")
    
    args = parser.parse_args()
    
    if args.verify:
        if args.verify_file:
            verify_tokens(args.verify_file, args.checkpoint, args.output_dir)
        else:
            print("Please specify a file to verify with --verify_file")
    else:
        tokenize_dataset(args.input_dir, args.output_dir, args.checkpoint)


if __name__ == "__main__":
    main()
