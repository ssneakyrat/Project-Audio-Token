"""
Helper modules for VocalTokenizer codec.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models
    """
    def __init__(self, d_model, max_seq_length=2000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, attn_mask=None):
        """
        Args:
            query: [batch_size, query_len, embed_dim]
            key: [batch_size, key_len, embed_dim]
            value: [batch_size, value_len, embed_dim]
            attn_mask: [batch_size, query_len, key_len] or None
        """
        batch_size = query.shape[0]
        
        # Project inputs
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape output
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        # Final projection
        output = self.out_proj(output)
        
        return output


class FeedForward(nn.Module):
    """
    Feed-forward network for transformer blocks
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Single layer of transformer encoder
    """
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        """
        Args:
            src: [batch_size, src_len, d_model]
            src_mask: [batch_size, src_len, src_len] or None
        """
        # Self attention block
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, src_mask)
        src = src + self.dropout(src2)
        
        # Feed-forward block
        src2 = self.norm2(src)
        src2 = self.feed_forward(src2)
        src = src + self.dropout(src2)
        
        return src


class TransformerDecoderLayer(nn.Module):
    """
    Single layer of transformer decoder
    """
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt: [batch_size, tgt_len, d_model]
            memory: [batch_size, src_len, d_model] 
            tgt_mask: [batch_size, tgt_len, tgt_len] or None
            memory_mask: [batch_size, tgt_len, src_len] or None
        """
        # Self attention block
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt2, tgt2, tgt2, tgt_mask)
        tgt = tgt + self.dropout(tgt2)
        
        # Cross attention block
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn(tgt2, memory, memory, memory_mask)
        tgt = tgt + self.dropout(tgt2)
        
        # Feed-forward block
        tgt2 = self.norm3(tgt)
        tgt2 = self.feed_forward(tgt2)
        tgt = tgt + self.dropout(tgt2)
        
        return tgt


class MelSpectrogram(nn.Module):
    """
    Differentiable mel-spectrogram transform
    """
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256, 
                 win_length=1024, n_mels=128, mel_fmin=0.0, mel_fmax=None):
        super(MelSpectrogram, self).__init__()
        
        import torchaudio.transforms as T
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length, 
            f_min=mel_fmin,
            f_max=mel_fmax if mel_fmax else sample_rate // 2,
            n_mels=n_mels,
            center=True,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="slaney"
        )
    
    def forward(self, waveform):
        """
        Args:
            waveform: [batch_size, n_samples]
        """
        mel = self.mel_transform(waveform)
        # Convert power to dB
        mel = torch.log10(torch.clamp(mel, min=1e-5))
        return mel


class PitchExtractor(nn.Module):
    """
    Neural pitch extractor for singing voice
    """
    def __init__(self, input_dim, hidden_dim=256, f0_min=50, f0_max=1000):
        super(PitchExtractor, self).__init__()
        
        self.f0_min = f0_min
        self.f0_max = f0_max
        
        # CNN-based feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
        )
        
        # Pitch prediction head
        self.pitch_predictor = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=3, padding=1)
        )
        
        # Voicing prediction (whether a frame has a valid pitch)
        self.voicing_predictor = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, mel_spec):
        """
        Args:
            mel_spec: [batch_size, n_mels, time]
            
        Returns:
            f0: [batch_size, time] - pitch in Hz
            voiced: [batch_size, time] - voicing probability
        """
        # Extract features
        features = self.conv_layers(mel_spec)
        
        # Predict pitch (in log-Hz)
        pitch_logits = self.pitch_predictor(features).squeeze(1)
        
        # Predict voicing
        voiced = self.voicing_predictor(features).squeeze(1)
        
        # Convert to Hz using sigmoid and scaling
        log_f0_min = torch.log(torch.tensor(self.f0_min))
        log_f0_max = torch.log(torch.tensor(self.f0_max))
        
        f0 = torch.exp(log_f0_min + torch.sigmoid(pitch_logits) * (log_f0_max - log_f0_min))
        
        return f0, voiced
