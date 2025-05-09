"""
Loss functions for training the VocalTokenizer codec.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class SpectralLoss(nn.Module):
    """
    Multi-resolution spectral loss for audio reconstruction
    """
    def __init__(self, fft_sizes=[512, 1024, 2048], 
                 hop_sizes=[128, 256, 512], 
                 win_lengths=[512, 1024, 2048],
                 sample_rate=22050):
        super(SpectralLoss, self).__init__()
        
        self.transforms = nn.ModuleList([
            torchaudio.transforms.Spectrogram(
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                power=1.0,  # magnitude spectrogram
                center=True,
                pad_mode="reflect",
                normalized=False
            )
            for fft_size, hop_size, win_length in zip(fft_sizes, hop_sizes, win_lengths)
        ])
        
        self.sample_rate = sample_rate
        
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: [B, T_samples] predicted waveform
            y_true: [B, T_samples] true waveform
        """
        loss = 0.0
        
        for transform in self.transforms:
            # Compute spectrograms
            S_true = transform(y_true)
            S_pred = transform(y_pred)
            
            # Log magnitude in power space
            log_S_true = torch.log(torch.clamp(S_true, min=1e-7))
            log_S_pred = torch.log(torch.clamp(S_pred, min=1e-7))
            
            # Compute L1 loss on log magnitudes
            mag_loss = F.l1_loss(log_S_true, log_S_pred)
            
            # Compute L2 loss on linear magnitudes
            lin_loss = F.mse_loss(S_true, S_pred)
            
            loss = loss + mag_loss + lin_loss
            
        return loss / len(self.transforms)


class MelReconstructionLoss(nn.Module):
    """
    Loss for mel-spectrogram reconstruction
    """
    def __init__(self):
        super(MelReconstructionLoss, self).__init__()
    
    def forward(self, mel_pred, mel_true):
        """
        Args:
            mel_pred: [B, n_mels, T] predicted mel-spectrogram
            mel_true: [B, n_mels, T] true mel-spectrogram
        """
        #print( mel_pred.shape )
        #print( mel_true.shape )

        mel_pred_aligned = mel_pred.permute(0, 2, 1)

        # L1 loss
        l1_loss = F.l1_loss(mel_pred_aligned, mel_true)
        
        # L2 loss
        l2_loss = F.mse_loss(mel_pred_aligned, mel_true)
        
        # Combine losses
        loss = l1_loss + l2_loss
        
        return loss


class PitchLoss(nn.Module):
    """
    Loss for pitch prediction
    """
    def __init__(self):
        super(PitchLoss, self).__init__()
    
    def forward(self, f0_pred, f0_true, voiced_pred, voiced_true):
        """
        Args:
            f0_pred: [B, T] predicted f0 (Hz)
            f0_true: [B, T] true f0 (Hz)
            voiced_pred: [B, T] predicted voicing probability (0-1)
            voiced_true: [B, T] true voicing mask (0 or 1)
        """
        # Mask for voiced frames
        voiced_mask = (voiced_true > 0.5).float()
        
        # Convert to log domain for perceptually-relevant distance
        log_f0_pred = torch.log(torch.clamp(f0_pred, min=1.0))
        log_f0_true = torch.log(torch.clamp(f0_true, min=1.0))
        
        # MSE loss on log-f0 for voiced frames
        f0_loss = F.mse_loss(log_f0_pred * voiced_mask, log_f0_true * voiced_mask, reduction='sum')
        f0_loss = f0_loss / (voiced_mask.sum() + 1e-8)  # Average over voiced frames
        
        # Binary cross-entropy for voicing detection
        voiced_loss = F.binary_cross_entropy(voiced_pred, voiced_true)
        
        # Combine losses
        loss = f0_loss + voiced_loss
        
        return loss


class VocalDiscriminator(nn.Module):
    """
    Special discriminator for singing voice perceptual loss
    """
    def __init__(self, input_channels=1, channels=32, n_layers=4):
        super(VocalDiscriminator, self).__init__()
        
        layers = []
        # Initial convolution
        layers.append(nn.Conv1d(input_channels, channels, kernel_size=15, stride=1, padding=7))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Increasing channels with stride
        current_channels = channels
        for i in range(n_layers):
            layers.append(
                nn.Conv1d(
                    current_channels, 
                    current_channels * 2, 
                    kernel_size=41, 
                    stride=2, 
                    padding=20, 
                    groups=current_channels
                )
            )
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            # Point-wise convolution
            layers.append(
                nn.Conv1d(
                    current_channels * 2, 
                    current_channels * 2, 
                    kernel_size=1
                )
            )
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            current_channels *= 2
        
        # Final 1x1 conv for classification
        layers.append(nn.Conv1d(current_channels, 1, kernel_size=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: [B, 1, T] audio waveform
        """
        return self.model(x)


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-period discriminator from HiFi-GAN
    """
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super(MultiPeriodDiscriminator, self).__init__()
        
        self.discriminators = nn.ModuleList([
            VocalDiscriminator(input_channels=1)
            for _ in periods
        ])
        
        self.periods = periods
    
    def forward(self, x):
        """
        Args:
            x: [B, T] audio waveform
        """
        x = x.unsqueeze(1)  # [B, 1, T]
        outputs = []
        
        for period, disc in zip(self.periods, self.discriminators):
            # Reshape for period
            batch_size, _, length = x.shape
            padded_length = (length // period + 1) * period
            padded_x = F.pad(x, (0, padded_length - length))
            
            # Reshape to [B, 1, period, T//period]
            reshaped_x = padded_x.view(batch_size, 1, period, -1)
            # Permute to [B, 1, T//period, period]
            reshaped_x = reshaped_x.permute(0, 1, 3, 2)
            # Reshape to [B, period, T//period]
            reshaped_x = reshaped_x.reshape(batch_size, period, -1)
            
            # Apply discriminator
            outputs.append(disc(reshaped_x))
        
        return outputs


class PerceptualLoss(nn.Module):
    """
    GAN-based perceptual loss using vocal discriminator
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        
        self.discriminator = VocalDiscriminator()
        
        # Feature extraction layers (indices in the discriminator)
        self.feature_layers = [1, 3, 5, 7, 9]
    
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: [B, T] predicted waveform
            y_true: [B, T] true waveform
        """
        # Extract features from real audio
        real_features = self._extract_features(y_true.unsqueeze(1))
        
        # Extract features from generated audio
        fake_features = self._extract_features(y_pred.unsqueeze(1))
        
        # Calculate feature matching loss
        loss = 0.0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss = loss + F.l1_loss(fake_feat, real_feat)
        
        return loss / len(self.feature_layers)
    
    def _extract_features(self, x):
        """
        Extract intermediate features from the discriminator
        """
        features = []
        for i, layer in enumerate(self.discriminator.model):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        
        return features


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for training the generator
    """
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        
        self.discriminator = MultiPeriodDiscriminator()
    
    def disc_loss(self, y_pred, y_true):
        """
        Compute discriminator loss
        
        Args:
            y_pred: [B, T] predicted waveform
            y_true: [B, T] true waveform
        """
        # Get discriminator outputs for real and fake
        real_outputs = self.discriminator(y_true)
        fake_outputs = self.discriminator(y_pred.detach())
        
        # Compute losses
        d_loss = 0.0
        
        for real_out, fake_out in zip(real_outputs, fake_outputs):
            d_real_loss = torch.mean((1 - real_out)**2)
            d_fake_loss = torch.mean(fake_out**2)
            d_loss = d_loss + d_real_loss + d_fake_loss
        
        return d_loss / len(real_outputs)
    
    def gen_loss(self, y_pred):
        """
        Compute generator loss
        
        Args:
            y_pred: [B, T] predicted waveform
        """
        # Get discriminator outputs for fake
        fake_outputs = self.discriminator(y_pred)
        
        # Compute loss
        g_loss = 0.0
        
        for fake_out in fake_outputs:
            g_loss = g_loss + torch.mean((1 - fake_out)**2)
        
        return g_loss / len(fake_outputs)


class VocalTokenizerLoss(nn.Module):
    """
    Combined loss function for VocalTokenizer training
    """
    def __init__(self, config):
        super(VocalTokenizerLoss, self).__init__()
        
        self.spectral_loss = SpectralLoss(sample_rate=config.sample_rate)
        self.mel_loss = MelReconstructionLoss()
        self.pitch_loss = PitchLoss()
        self.perceptual_loss = PerceptualLoss()
        self.adversarial_loss = AdversarialLoss()
        
        # Loss weights
        self.spectral_weight = config.spectral_loss_weight
        self.waveform_weight = config.waveform_loss_weight
        self.perceptual_weight = config.perceptual_loss_weight
        self.adversarial_weight = config.adversarial_loss_weight
    
    def forward(self, model_output, target):
        """
        Compute total loss
        
        Args:
            model_output: dict with model outputs
            target: dict with targets
        """
        losses = {}
        
        # VQ loss
        losses['vq_loss'] = model_output['vq_loss']
        
        # Mel reconstruction loss
        losses['mel_loss'] = self.mel_loss(model_output['mel_output'], target['mel'])
        
        # Pitch loss
        if 'f0' in model_output and 'f0' in target:
            losses['pitch_loss'] = self.pitch_loss(
                model_output['f0'], 
                target['f0'],
                model_output['voiced'], 
                target['voiced']
            )
        
        # Waveform reconstruction losses (if vocoder is implemented)
        if model_output['waveform'] is not None and 'waveform' in target:
            losses['spectral_loss'] = self.spectral_loss(
                model_output['waveform'], target['waveform']
            ) * self.spectral_weight
            
            losses['waveform_loss'] = F.l1_loss(
                model_output['waveform'], target['waveform']
            ) * self.waveform_weight
            
            losses['perceptual_loss'] = self.perceptual_loss(
                model_output['waveform'], target['waveform']
            ) * self.perceptual_weight
            
            losses['adversarial_loss'] = self.adversarial_loss.gen_loss(
                model_output['waveform']
            ) * self.adversarial_weight
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
