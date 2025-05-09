"""
Configuration for VocalTokenizer codec.
"""

class CodecConfig:
    # Audio processing
    sample_rate = 22050
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    n_mels = 128
    
    # Model dimensions
    encoder_dim = 512
    decoder_dim = 512
    n_heads = 8
    n_layers = 6
    dropout = 0.1
    
    # Vector quantization
    codebook_size = 1024
    n_codebooks = 4
    commitment_cost = 0.25
    
    # Pitch extraction
    f0_min = 40
    f0_max = 12000
    
    # Training
    batch_size = 16
    learning_rate = 3e-4
    max_epochs = 100
    grad_clip = 1.0
    
    # Loss weights
    spectral_loss_weight = 1.0
    waveform_loss_weight = 0.1
    perceptual_loss_weight = 0.5
    adversarial_loss_weight = 0.1
