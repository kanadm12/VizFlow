# 🎨 VizFlow - Example Code Gallery

## Example 1: Convolutional Autoencoder (Recommended for New Users)

**What it does:** Image compression and reconstruction using encoding-decoding architecture

**Architecture Highlights:**
- **Encoder:** 4 convolutional blocks progressively compressing the image
- **Bottleneck:** Compressed latent representation (512-dim)
- **Decoder:** 4 transposed convolutions reconstructing the image
- **Features:** Batch normalization, dropout, ReLU activations

**Best Visualized As:** DAG Layout (shows encoder→bottleneck→decoder flow clearly)

**Code to Copy:**
```python
import torch
import torch.nn as nn

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(ConvolutionalAutoencoder, self).__init__()
        
        # ENCODER
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.enc_pool1 = nn.MaxPool2d(2, 2)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc_pool2 = nn.MaxPool2d(2, 2)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_pool3 = nn.MaxPool2d(2, 2)
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc_pool4 = nn.MaxPool2d(2, 2)
        
        # BOTTLENECK
        self.bottleneck_fc1 = nn.Linear(256 * 14 * 14, latent_dim)
        self.bottleneck_fc2 = nn.Linear(latent_dim, 256 * 14 * 14)
        
        # DECODER
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2)
        
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
    
    def encode(self, x):
        x = self.relu(self.enc_conv1(x))
        x = self.enc_pool1(x)
        x = self.relu(self.enc_conv2(x))
        x = self.enc_pool2(x)
        x = self.relu(self.enc_conv3(x))
        x = self.enc_pool3(x)
        x = self.relu(self.enc_conv4(x))
        x = self.enc_pool4(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.bottleneck_fc1(x))
        return x
    
    def decode(self, z):
        x = self.relu(self.bottleneck_fc2(z))
        x = x.view(-1, 256, 14, 14)
        x = self.relu(self.dec_conv1(x))
        x = self.relu(self.dec_conv2(x))
        x = self.relu(self.dec_conv3(x))
        x = torch.sigmoid(self.dec_conv4(x))
        return x
    
    def forward(self, x):
        latent = self.encode(x)
        output = self.decode(latent)
        return output, latent

model = ConvolutionalAutoencoder(latent_dim=512)
```

**What You'll See:**
- Input layer (3 channels)
- 4 encoder conv blocks with pooling
- Bottleneck compression layer
- 4 decoder transposed conv blocks
- Output layer (3 channels)
- **Curved connecting lines between all layers** ✨

---

## Example 2: Vision Transformer (ViT)

**What it does:** Image classification using transformer architecture on image patches

**Architecture Highlights:**
- Patch embedding (divides image into 16x16 patches)
- Positional encoding
- Transformer encoder blocks
- Classification head

**Best Visualized As:** DAG or Tree Layout

**Code to Copy:**
```python
import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, embed_dim=768):
        super(VisionTransformer, self).__init__()
        
        num_patches = (img_size // patch_size) ** 2
        
        # Patch Embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=12, batch_first=True),
            num_layers=12
        )
        
        # Classification Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Classification
        x = self.norm(x[:, 0])
        x = self.head(x)
        
        return x

model = VisionTransformer()
```

---

## Example 3: LSTM Sequence-to-Sequence (Seq2Seq)

**What it does:** Sequence translation (e.g., machine translation, summarization)

**Architecture Highlights:**
- Encoder LSTM (processes input sequence)
- Decoder LSTM (generates output sequence)
- Attention mechanism
- Embedding layers

**Best Visualized As:** Flowchart Layout

**Code to Copy:**
```python
import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=256, hidden_dim=512):
        super(Seq2Seq, self).__init__()
        
        # Encoder
        self.embedding_enc = nn.Embedding(vocab_size, embed_dim)
        self.lstm_enc = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_enc = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Decoder
        self.embedding_dec = nn.Embedding(vocab_size, embed_dim)
        self.lstm_dec = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        # Attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
    
    def encode(self, src):
        embedded = self.embedding_enc(src)
        encoder_outputs, (hidden, cell) = self.lstm_enc(embedded)
        hidden = self.fc_enc(hidden[-1])
        return encoder_outputs, hidden, cell
    
    def decode(self, tgt, encoder_outputs, hidden, cell):
        embedded = self.embedding_dec(tgt)
        decoder_output, (hidden, cell) = self.lstm_dec(embedded, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        
        # Apply attention
        attn_output, _ = self.attention(decoder_output, encoder_outputs, encoder_outputs)
        
        output = self.fc_out(attn_output)
        return output, hidden.squeeze(0), cell.squeeze(0)
    
    def forward(self, src, tgt):
        encoder_outputs, hidden, cell = self.encode(src)
        decoder_out, _, _ = self.decode(tgt, encoder_outputs, hidden, cell)
        return decoder_out

model = Seq2Seq()
```

---

## Example 4: ResNet-like Architecture

**What it does:** Deep residual network with skip connections for image classification

**Architecture Highlights:**
- Convolutional layers with residual blocks
- Skip connections
- Batch normalization
- Adaptive average pooling

**Best Visualized As:** Graph Layout (shows skip connections clearly)

**Code to Copy:**
```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = ResNet()
```

---

## Example 5: Graph Neural Network (GNN)

**What it does:** Process graph-structured data for node classification or link prediction

**Architecture Highlights:**
- Graph convolution layers
- Aggregation operations
- Message passing
- Pooling layers

**Best Visualized As:** Graph Layout

**Code to Copy:**
```python
import torch
import torch.nn as nn

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x, adj):
        x = self.weight(x)
        x = torch.bmm(adj, x)
        x = x + self.bias
        return x

class GNN(nn.Module):
    def __init__(self, in_features=10, hidden_dim=64, num_classes=4):
        super(GNN, self).__init__()
        self.gc1 = GraphConvLayer(in_features, hidden_dim)
        self.gc2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.gc3 = GraphConvLayer(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x, adj):
        x = self.relu(self.gc1(x, adj))
        x = self.dropout(x)
        x = self.relu(self.gc2(x, adj))
        x = self.dropout(x)
        x = self.gc3(x, adj)
        return x

model = GNN()
```

---

## How to Use These Examples

### Step 1: Copy Example Code
- Select code from above
- Paste into VizFlow editor

### Step 2: Run Code
- Click **RUN** button
- VizFlow parses the model

### Step 3: Choose Visualization
- **📊 DAG** - For encoders/decoders, sequential networks
- **🌳 Tree** - For hierarchical structures
- **🔀 Flowchart** - For sequential processes
- **🔗 Graph** - For complex connections, GNNs

### Step 4: Interact
- **Click nodes** to see details
- **Zoom** to explore
- **Switch layouts** to see different perspectives

---

## Visual Features You'll See

### ✨ Enhanced Connections
- **Curved connecting lines** between blocks
- **Animated flow pulses** moving along connections
- **Glow effects** on connections
- **Arrow markers** showing data direction

### 🎨 Color Coding
- 🔵 **Blue** - Regular layers
- 🟢 **Green** - Selected layer
- 💫 **Glow** - Interactive feedback

### 📊 Information
- Layer type and name
- Parameter counts
- Shape transformations
- Connection labels

---

## Tips for Visualization

### Best Practices:
1. **Start with DAG layout** (easiest to understand)
2. **Use zoom** for detailed exploration
3. **Click layers** to see properties
4. **Try different layouts** to see different perspectives
5. **Watch the animated pulses** flowing through connections

### For Complex Models:
- Zoom out to see full architecture
- Use Graph layout to see all relationships
- Click on layers of interest for details
- Take screenshots for documentation

---

## What the Curved Links Show

**Features of the new enhanced linking:**

✨ **Curved Paths:** Bezier curves prevent overlapping lines  
💫 **Glow Effects:** Subtle glow for visual feedback  
🔴 **Flow Pulses:** Animated dots moving along connections  
🎯 **Arrow Markers:** Direction of data flow  
🖱️ **Hover Effects:** Connections highlight on hover  

---

**Created:** November 17, 2025  
**Status:** ✅ Ready to Use

Pick any example above and visualize it! 🚀
