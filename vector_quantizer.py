"""
Vector Quantization modules for VocalTokenizer codec.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Neural Discrete Representation Learning (VQ-VAE)
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        """
        Args:
            num_embeddings: size of the codebook
            embedding_dim: dimension of each codebook vector
            commitment_cost: weight for commitment loss
        """
        super(VectorQuantizer, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize codebook
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
    
    def forward(self, inputs):
        """
        Args:
            inputs: [B, T, D] tensor of features to be quantized
            
        Returns:
            quantized: [B, T, D] tensor of quantized features
            indices: [B, T] tensor of codebook indices
            loss: commitment loss
        """
        # Flatten input except last dimension
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances to all codebook vectors
        distances = torch.sum(flat_input**2, dim=1, keepdim=True) + \
                    torch.sum(self.codebook.weight**2, dim=1) - \
                    2 * torch.matmul(flat_input, self.codebook.weight.t())
        
        # Find nearest codebook vector
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Reshape indices to match input batch shape
        encoding_indices = encoding_indices.view(inputs.shape[0], inputs.shape[1])
        
        # Convert indices to one-hot encodings
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Quantize: look up selected embeddings
        quantized = torch.matmul(encodings, self.codebook.weight)
        quantized = quantized.view_as(inputs)
        
        # Compute loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, encoding_indices, loss


class ResidualVQ(nn.Module):
    """
    Residual Vector Quantization for multi-stage quantization
    """
    def __init__(self, num_quantizers, num_embeddings, embedding_dim, commitment_cost=0.25):
        """
        Args:
            num_quantizers: number of VQ stages in cascade
            num_embeddings: size of each codebook
            embedding_dim: dimension of each codebook vector
            commitment_cost: weight for commitment loss
        """
        super(ResidualVQ, self).__init__()
        
        self.num_quantizers = num_quantizers
        self.embedding_dim = embedding_dim
        
        # Create multiple VQ layers
        self.quantizers = nn.ModuleList([
            VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
            for _ in range(num_quantizers)
        ])
    
    def forward(self, x):
        """
        Args:
            x: [B, T, D] input features
            
        Returns:
            quantized: [B, T, D] quantized features
            indices: List of [B, T] tensors with codebook indices
            loss: combined quantization loss
        """
        residual = x
        quantized_out = torch.zeros_like(x)
        all_indices = []
        total_loss = 0.0
        
        # Apply each quantizer in sequence to the residual
        for i, quantizer in enumerate(self.quantizers):
            quantized, indices, loss = quantizer(residual)
            
            # Update the quantized output and residual
            quantized_out = quantized_out + quantized
            residual = residual - quantized
            
            all_indices.append(indices)
            total_loss = total_loss + loss
        
        return quantized_out, all_indices, total_loss


class HierarchicalVQ(nn.Module):
    """
    Hierarchical Vector Quantization with separate codebooks for different aspects
    """
    def __init__(self, 
                 timbre_codebook_size=512, 
                 expression_codebook_size=256, 
                 detail_codebook_size=512,
                 embedding_dim=512,
                 commitment_cost=0.25):
        """
        Args:
            timbre_codebook_size: size of the main timbre codebook
            expression_codebook_size: size of the expression codebook
            detail_codebook_size: size of the fine detail codebook 
            embedding_dim: dimension of codebook vectors
            commitment_cost: weight for commitment loss
        """
        super(HierarchicalVQ, self).__init__()
        
        # Feature projection networks
        self.timbre_proj = nn.Linear(embedding_dim, embedding_dim)
        self.expression_proj = nn.Linear(embedding_dim, embedding_dim)
        self.detail_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Quantizers for different aspects
        self.timbre_vq = VectorQuantizer(timbre_codebook_size, embedding_dim, commitment_cost)
        self.expression_vq = VectorQuantizer(expression_codebook_size, embedding_dim, commitment_cost)
        self.detail_vq = ResidualVQ(2, detail_codebook_size, embedding_dim, commitment_cost)
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim * 3, embedding_dim)
        
    def forward(self, x):
        """
        Args:
            x: [B, T, D] input features
            
        Returns:
            quantized: [B, T, D] quantized features 
            indices: dict of codebook indices
            loss: combined quantization loss
        """
        # Project input to different feature spaces
        timbre_feat = self.timbre_proj(x)
        expression_feat = self.expression_proj(x)
        detail_feat = self.detail_proj(x)
        
        # Quantize each feature stream
        timbre_q, timbre_idx, timbre_loss = self.timbre_vq(timbre_feat)
        expression_q, expression_idx, expression_loss = self.expression_vq(expression_feat)
        detail_q, detail_idx, detail_loss = self.detail_vq(detail_feat)
        
        # Combine quantized representations
        combined = torch.cat([timbre_q, expression_q, detail_q], dim=-1)
        quantized = self.output_proj(combined)
        
        # Collect indices and losses
        indices = {
            'timbre': timbre_idx,
            'expression': expression_idx,
            'detail': detail_idx
        }
        
        loss = timbre_loss + expression_loss + detail_loss
        
        return quantized, indices, loss
