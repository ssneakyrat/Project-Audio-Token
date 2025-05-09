"""
Main model architecture for VocalTokenizer codec.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import (
    MelSpectrogram, 
    PitchExtractor,
    TransformerEncoderLayer, 
    TransformerDecoderLayer,
    PositionalEncoding
)
from vector_quantizer import ResidualVQ, HierarchicalVQ


class VocalEncoder(nn.Module):
    """
    Encoder for the VocalTokenizer codec
    """
    def __init__(self, 
                 n_mels=128,
                 hidden_dim=512,
                 n_layers=6,
                 n_heads=8,
                 ff_dim=2048,
                 dropout=0.1,
                 sample_rate=22050,
                 n_fft=1024,
                 hop_length=256,
                 win_length=1024,
                 f0_min=50,
                 f0_max=1000):
        super(VocalEncoder, self).__init__()
        
        # Mel spectrogram extraction
        self.mel_extractor = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels
        )
        
        # Pitch extraction
        self.pitch_extractor = PitchExtractor(
            input_dim=n_mels,
            hidden_dim=hidden_dim // 2,
            f0_min=f0_min,
            f0_max=f0_max
        )
        
        # Initial projection
        self.input_projection = nn.Conv1d(n_mels, hidden_dim, kernel_size=1)
        
        # Pitch conditioning
        self.pitch_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
        
        # Voiced/unvoiced conditioning
        self.voiced_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, waveform):
        """
        Args:
            waveform: [B, T_samples] audio waveform
            
        Returns:
            features: [B, T_frames, hidden_dim] encoded features
            f0: [B, T_frames] pitch in Hz
            voiced: [B, T_frames] voicing probability
        """
        # Extract mel spectrogram
        mel = self.mel_extractor(waveform)  # [B, n_mels, T_frames]
        
        # Extract pitch
        f0, voiced = self.pitch_extractor(mel)  # [B, T_frames], [B, T_frames]
        
        # Initial projection
        x = self.input_projection(mel)  # [B, hidden_dim, T_frames]
        x = x.transpose(1, 2)  # [B, T_frames, hidden_dim]
        
        # Add pitch and voicing information
        f0_emb = self.pitch_embedding(f0.unsqueeze(-1))  # [B, T_frames, hidden_dim]
        voiced_emb = self.voiced_embedding(voiced.unsqueeze(-1))  # [B, T_frames, hidden_dim]
        
        x = x + f0_emb + voiced_emb
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        return x, f0, voiced


class VocalDecoder(nn.Module):
    """
    Decoder for VocalTokenizer codec
    """
    def __init__(self,
                 hidden_dim=512,
                 n_layers=6,
                 n_heads=8,
                 ff_dim=2048,
                 dropout=0.1,
                 n_mels=128):
        super(VocalDecoder, self).__init__()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        
        # Pitch conditioning
        self.pitch_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
        
        # Voiced/unvoiced conditioning
        self.voiced_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
        
        # Transformer decoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection to mel spectrogram
        self.mel_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_mels)
        )
        
        # Neural vocoder placeholder (would be implemented or imported separately)
        # This could be a WaveNet, WaveGlow, HiFi-GAN, etc.
        self.vocoder = None
        
    def forward(self, quantized_features, f0=None, voiced=None, encoder_output=None):
        """
        Args:
            quantized_features: [B, T, hidden_dim] quantized features from VQ
            f0: [B, T] pitch in Hz (optional)
            voiced: [B, T] voicing probability (optional)
            encoder_output: [B, T, hidden_dim] encoder output for attention (optional)
            
        Returns:
            mel_output: [B, n_mels, T] reconstructed mel spectrogram
            waveform: [B, T_samples] reconstructed waveform (if vocoder is implemented)
        """
        # Add positional encoding
        x = self.pos_encoder(quantized_features)
        
        # Add pitch and voicing information if provided
        if f0 is not None and voiced is not None:
            f0_emb = self.pitch_embedding(f0.unsqueeze(-1))
            voiced_emb = self.voiced_embedding(voiced.unsqueeze(-1))
            x = x + f0_emb + voiced_emb
        
        # Apply transformer layers
        memory = encoder_output if encoder_output is not None else quantized_features
        for layer in self.transformer_layers:
            x = layer(x, memory)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        # Project to mel spectrogram
        mel_output = self.mel_projection(x)
        mel_output = mel_output.transpose(1, 2)  # [B, n_mels, T]
        
        # Generate waveform if vocoder is implemented
        waveform = None
        if self.vocoder is not None:
            waveform = self.vocoder(mel_output)
            
        return mel_output, waveform


class VocalTokenizer(nn.Module):
    """
    Complete VocalTokenizer codec with encoder, VQ, and decoder
    """
    def __init__(self, config):
        """
        Args:
            config: configuration object with model parameters
        """
        super(VocalTokenizer, self).__init__()
        
        # Encoder
        self.encoder = VocalEncoder(
            n_mels=config.n_mels,
            hidden_dim=config.encoder_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            ff_dim=config.encoder_dim * 4,
            dropout=config.dropout,
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            f0_min=config.f0_min,
            f0_max=config.f0_max
        )
        
        # Vector Quantizer (select one of the available options)
        # Option 1: Residual VQ
        self.quantizer = ResidualVQ(
            num_quantizers=config.n_codebooks,
            num_embeddings=config.codebook_size,
            embedding_dim=config.encoder_dim,
            commitment_cost=config.commitment_cost
        )
        
        # Option 2: Hierarchical VQ
        # Uncomment to use instead of ResidualVQ
        # self.quantizer = HierarchicalVQ(
        #     timbre_codebook_size=512,
        #     expression_codebook_size=256,
        #     detail_codebook_size=512,
        #     embedding_dim=config.encoder_dim,
        #     commitment_cost=config.commitment_cost
        # )
        
        # Decoder
        self.decoder = VocalDecoder(
            hidden_dim=config.decoder_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            ff_dim=config.decoder_dim * 4,
            dropout=config.dropout,
            n_mels=config.n_mels
        )
        
    def encode(self, waveform):
        """
        Encode waveform to discrete tokens
        
        Args:
            waveform: [B, T_samples] audio waveform
            
        Returns:
            indices: token indices
            f0: pitch information
            voiced: voicing information
        """
        # Encode to continuous latent
        encoder_output, f0, voiced = self.encoder(waveform)
        
        # Quantize to discrete tokens
        quantized, indices, _ = self.quantizer(encoder_output)
        
        return indices, f0, voiced
    
    def decode(self, indices, f0=None, voiced=None):
        """
        Decode from discrete tokens to waveform
        
        Args:
            indices: token indices from encode()
            f0: pitch information (optional)
            voiced: voicing information (optional)
            
        Returns:
            waveform: reconstructed audio
        """
        # Look up embeddings from indices
        # This depends on the quantizer type
        if isinstance(self.quantizer, ResidualVQ):
            # For ResidualVQ, rebuild the embedding from all codebooks
            quantized = torch.zeros((indices[0].shape[0], indices[0].shape[1], self.encoder.hidden_dim))
            for i, idx_tensor in enumerate(indices):
                quantized = quantized + self.quantizer.quantizers[i].codebook(idx_tensor)
        else:
            # For other quantizers, implement appropriate lookup
            raise NotImplementedError("Decoding for this quantizer type not implemented")
        
        # Generate mel spectrogram and waveform
        mel_output, waveform = self.decoder(quantized, f0, voiced)
        
        # Return waveform if vocoder is implemented, otherwise mel
        if waveform is not None:
            return waveform
        else:
            return mel_output
    
    def forward(self, waveform):
        """
        Full forward pass through encoder, VQ, and decoder
        
        Args:
            waveform: [B, T_samples] audio waveform
            
        Returns:
            mel_output: reconstructed mel spectrogram
            waveform: reconstructed audio (if vocoder is implemented)
            vq_loss: vector quantization loss
            indices: token indices
        """
        # Encoder
        encoder_output, f0, voiced = self.encoder(waveform)
        
        # Vector quantization
        quantized, indices, vq_loss = self.quantizer(encoder_output)
        
        # Decoder
        mel_output, waveform = self.decoder(quantized, f0, voiced, encoder_output)
        
        output = {
            'mel_output': mel_output,
            'waveform': waveform,
            'vq_loss': vq_loss,
            'indices': indices,
            'f0': f0,
            'voiced': voiced
        }
        
        return output
