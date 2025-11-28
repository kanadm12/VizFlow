import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for Image Compression and Reconstruction
    
    Architecture:
    - Encoder: 4 convolutional blocks with pooling
    - Bottleneck: Compressed latent representation
    - Decoder: 4 transposed convolutions with upsampling
    - Final: Reconstructed output matching input dimensions
    """
    
    def __init__(self, latent_dim=256):
        super(ConvolutionalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        # ============ ENCODER ============
        # Input: (batch, 3, 224, 224)
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.enc_pool1 = nn.MaxPool2d(2, 2)  # → (batch, 32, 112, 112)
        
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.enc_pool2 = nn.MaxPool2d(2, 2)  # → (batch, 64, 56, 56)
        
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.enc_pool3 = nn.MaxPool2d(2, 2)  # → (batch, 128, 28, 28)
        
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.enc_pool4 = nn.MaxPool2d(2, 2)  # → (batch, 256, 14, 14)
        
        # Bottleneck: Fully connected layers
        self.bottleneck_fc1 = nn.Linear(256 * 14 * 14, latent_dim)
        self.bottleneck_fc2 = nn.Linear(latent_dim, 256 * 14 * 14)
        
        # ============ DECODER ============
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2)
        
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout2d(0.3)
    
    def encode(self, x):
        """Encoder path: Compress input to latent vector"""
        # Block 1
        x = self.relu(self.enc_conv1(x))
        x = self.batch_norm1(x)
        x = self.enc_pool1(x)
        x = self.dropout(x)
        
        # Block 2
        x = self.relu(self.enc_conv2(x))
        x = self.batch_norm2(x)
        x = self.enc_pool2(x)
        x = self.dropout(x)
        
        # Block 3
        x = self.relu(self.enc_conv3(x))
        x = self.batch_norm3(x)
        x = self.enc_pool3(x)
        x = self.dropout(x)
        
        # Block 4
        x = self.relu(self.enc_conv4(x))
        x = self.enc_pool4(x)
        
        # Flatten and bottleneck
        x = x.view(x.size(0), -1)
        x = self.relu(self.bottleneck_fc1(x))
        
        return x
    
    def decode(self, z):
        """Decoder path: Reconstruct from latent vector"""
        # Reconstruct spatial dimensions
        x = self.relu(self.bottleneck_fc2(z))
        x = x.view(-1, 256, 14, 14)
        
        # Deconvolution Block 1
        x = self.relu(self.dec_conv1(x))
        x = self.batch_norm3(x)
        x = self.dropout(x)
        
        # Deconvolution Block 2
        x = self.relu(self.dec_conv2(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)
        
        # Deconvolution Block 3
        x = self.relu(self.dec_conv3(x))
        x = self.batch_norm1(x)
        x = self.dropout(x)
        
        # Output Block
        x = torch.sigmoid(self.dec_conv4(x))
        
        return x
    
    def forward(self, x):
        """Complete forward pass: Encode → Bottleneck → Decode"""
        latent = self.encode(x)
        output = self.decode(latent)
        return output, latent


# ============ USAGE EXAMPLE ============
if __name__ == "__main__":
    # Initialize model
    model = ConvolutionalAutoencoder(latent_dim=512)
    
    # Print model architecture
    print(f"Model: ConvolutionalAutoencoder")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    output, latent = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Reconstruction loss: MSE between input and output")
