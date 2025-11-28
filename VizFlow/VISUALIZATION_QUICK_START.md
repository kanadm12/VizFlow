# VizFlow Enhanced Visualization - Quick Start Guide

## 🎬 Get Started in 30 Seconds

### Step 1: Write a Model
Paste any PyTorch model definition into the code editor:

```python
import torch.nn as nn

class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

model = MyNetwork()
```

### Step 2: Run the Code
Click the **RUN** button (or press Ctrl+Enter)

### Step 3: Choose Visualization Type

| Button | Type | Best For |
|--------|------|----------|
| 📊 | DAG | Neural Networks |
| 🌳 | Tree | Hierarchies |
| 🔀 | Flowchart | Process Flows |
| 🔗 | Graph | Dependencies |

---

## 📊 Diagram Type Guide

### DAG (Directed Acyclic Graph)
**Best for: Neural Networks, ML Models**

```
     Input Layer
         ↓
    [FC1: 784→256]
         ↓
    [ReLU: 256]
         ↓
    [FC2: 256→128]
         ↓
    [FC3: 128→10]
         ↓
    Output Layer
```

✅ **Features:**
- Automatic layer ordering
- Clear data flow visualization
- Optimal for understanding model architecture
- Best readability for sequential networks

---

### Tree Structure
**Best for: Inheritance Hierarchies, Class Structures**

```
        BaseNetwork
           /    \
       Input   Hidden
        /         \
      Fc1         ReLU
       \           /
         Fc2→Fc3→Output
```

✅ **Features:**
- Shows parent-child relationships
- Hierarchical organization
- Best for understanding inheritance
- Useful for class diagrams

---

### Flowchart
**Best for: Algorithm Flows, Process Diagrams**

```
[Start] → [Define Model]
   ↓
[Create Layers]
   ↓
[Implement Forward]
   ↓
[Train] → [End]
```

✅ **Features:**
- Sequential step visualization
- Clear start/end points
- Good for process documentation
- Easy to follow logic flow

---

### Force-Directed Graph
**Best for: Complex Relationships, General Graphs**

```
    Layer1  ←→  Layer2
       ↑         ↓
    Layer3  ←→  Layer4
       ↓         ↑
    Layer5  ←→  Layer6
```

✅ **Features:**
- Shows all relationships clearly
- Organic, natural positioning
- Good for discovering connections
- Best for non-hierarchical structures

---

## 🎮 Interactive Features

### Selecting Nodes
- **Click on any layer** to select it
- **Blue highlight** = selected layer
- **Info panel** shows layer details (appears at bottom)
- **Click again** to deselect

### Zooming
- **+ Button** = Zoom in (up to 200%)
- **- Button** = Zoom out (down to 50%)
- **Percentage display** = Current zoom level
- **Smooth zoom** = No jarring jumps

### Switching Layouts
- **Instant switching** between diagram types
- **Smooth animations** between layouts
- **Automatic recalculation** of positions
- **Preserves selection** when switching

### Info Panel
Shows when you select a layer:

```
═══════════════════════════════════
📋 Layer Name
─────────────────────────────────
Type: Linear
Parameters: 200,448
Output: Linear(256)
═══════════════════════════════════
```

---

## 💡 Example: From Code to Visualization

### Input Code:
```python
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64*30*30, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = ConvNet()
```

### What VizFlow Does:

1. **Parses** the class definition
2. **Extracts** all layers (conv1, relu, pool, conv2, fc1, fc2)
3. **Analyzes** the forward() method to determine connections
4. **Calculates** parameters for each layer
5. **Generates** graph structure
6. **Applies** best layout algorithm
7. **Renders** with smooth animations
8. **Enables** interactive exploration

### Result in DAG View:
```
Input (3, 224, 224)
        ↓
[Conv2d(3→32)] - 896 params
        ↓
[ReLU]
        ↓
[MaxPool2d(2)]
        ↓
[Conv2d(32→64)] - 18,496 params
        ↓
[ReLU]
        ↓
[MaxPool2d(2)]
        ↓
[Flatten]
        ↓
[Linear(57,600→256)] - 14,745,856 params
        ↓
[Linear(256→10)] - 2,570 params
        ↓
Output (10)
```

---

## 🔧 Tips & Tricks

### Tip 1: Click Multiple Layers
- Select a layer to see its properties
- Each click toggles selection
- Info panel updates instantly

### Tip 2: Use Zoom for Details
- **Zoom in** to see small details
- **Zoom out** to see entire architecture
- **% indicator** shows current zoom

### Tip 3: Try Different Layouts
- **DAG** = Best for understanding flow
- **Tree** = Best for showing hierarchies
- **Flowchart** = Best for presentation
- **Graph** = Best for exploring connections

### Tip 4: Watch Loading Animation
- Small gear icon = Computing layout
- Wait for it to complete
- Smooth animations appear after

### Tip 5: Use Info Panel
- Shows layer name, type, parameters
- Appears when you click a layer
- Scroll down in panel for more details

---

## ⚡ Keyboard Shortcuts (Coming Soon)

| Shortcut | Action |
|----------|--------|
| `Ctrl + Mouse Wheel` | Zoom in/out |
| `Space + Drag` | Pan around |
| `D` | Toggle DAG layout |
| `T` | Toggle Tree layout |
| `F` | Toggle Flowchart layout |
| `G` | Toggle Graph layout |
| `C` | Clear selection |

---

## 📈 Supported Model Types

### ✅ PyTorch
- `nn.Linear`
- `nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`
- `nn.RNN`, `nn.LSTM`, `nn.GRU`
- `nn.BatchNorm1d`, `nn.BatchNorm2d`
- `nn.Dropout`, `nn.Dropout2d`
- `nn.ReLU`, `nn.Sigmoid`, `nn.Tanh`
- Custom layers (basic support)

### ✅ TensorFlow/Keras
- `Dense` layers
- `Conv1D`, `Conv2D`, `Conv3D`
- `LSTM`, `GRU`
- `BatchNormalization`
- `Dropout`
- Activation layers

### ✅ Generic Python
- Any class definitions
- Function definitions
- Variable assignments
- Process flows

---

## 🚀 Advanced Usage

### Customizing Layouts

By default, VizFlow **auto-detects** the best layout:
- Single-root acyclic graph → **Tree layout**
- Multi-root acyclic graph → **DAG layout**
- General graphs → **Force layout**

But you can **override** by clicking layout buttons!

### Performance Tips

**For large models (>200 layers):**
1. Use **DAG or Flowchart** layout (faster)
2. Avoid **Force layout** (slower)
3. Use **Zoom out** to see all layers
4. Selection works best on zoomed-in views

**For complex models:**
1. Start with **DAG layout**
2. Switch to other layouts as needed
3. Use info panel to understand layers
4. Take screenshots for documentation

---

## ❓ FAQ

**Q: Why are layers in different positions?**
A: Different layouts optimize for different aspects. DAG optimizes for readability, Force optimizes for showing all relationships.

**Q: Can I export the diagram?**
A: Yes! Right-click on the visualization and select "Save image as..."

**Q: What if my model doesn't parse?**
A: Ensure you have a proper `class MyModel(nn.Module)` definition and `forward()` method.

**Q: How many layers can I visualize?**
A: Tested up to 1000 layers. Performance degrades gradually with size.

**Q: Can I use this for non-neural networks?**
A: Yes! The flowchart view works great for any Python code structure.

---

## 🎨 Color Guide

| Color | Meaning |
|-------|---------|
| 🔵 Blue | Regular layer |
| 🟢 Green | Selected layer |
| 🟡 Yellow | Being hovered |
| ⚪ White | Labels & text |
| 💫 Glow | Special effect on selection |

---

## 📞 Support

**Issues?**
- Check browser console for errors
- Try refreshing the page
- Switch layout types
- Clear browser cache

**Feature requests?**
- File on GitHub Issues
- Describe use case
- Include example code

---

**Happy Visualizing! 🎉**

Created: November 17, 2025  
Status: ✅ Ready to Use
