# VizFlow - Professional AI/ML Model Visualizer

> **Professional, modular architecture for visualizing neural network architectures in real-time**

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/kanadm12/VizFlow)
[![Beta Release](https://img.shields.io/badge/version-2.0-blue)](https://github.com/kanadm12/VizFlow/releases)
[![Dynamic UI](https://img.shields.io/badge/UI-Animated-orange)](https://github.com/kanadm12/VizFlow)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 🎯 Overview

VizFlow is a professional web-based IDE for visualizing and analyzing AI/ML model architectures. Built with React, Tailwind CSS, Framer Motion, and React Hot Toast, it provides an intuitive, modern interface for developers to understand complex neural network designs.

### ✨ Key Features

#### Frontend Features
- **💻 Interactive Code Editor** - Python syntax highlighting with error detection
- **📤 File Upload** - Load local Python files into the editor
- **📊 Real-time Visualization** - Interactive SVG-based model architecture diagrams
- **🖥️ Terminal Window** - Built-in terminal for viewing code execution output
- **🔍 Layer Inspector** - Click any layer to view detailed parameter information
- **⚙️ Model Statistics** - Automatic parameter counting and layer analysis
- **📈 Zoom & Pan** - Intuitive controls for large model graphs
- **🎬 Smooth Animations** - Professional Framer Motion animations throughout
- **🔔 Smart Notifications** - React Hot Toast feedback on all interactions
- **♿ Accessibility** - Respects reduced motion preferences

#### Backend Features
- **🔐 User Authentication** - Secure login and signup system
- **💾 Cloud Storage** - Save and manage your projects in the cloud
- **🤖 AI Provider Tracking** - Track which AI assistant (Claude, Gemini, ChatGPT, Copilot) users prefer
- **📈 Usage Analytics** - Monitor AI provider usage statistics per user
- **🗂️ Project Management** - Create, save, and organize multiple projects
- **📜 Execution History** - Track all code executions with timestamps
- **🔒 Secure API** - JWT-based authentication for all API endpoints

## 🚀 Quick Start

### Prerequisites
- Node.js 16+
- npm or yarn

### Installation

#### Frontend Setup

```bash
# Clone the repository
git clone https://github.com/kanadm12/VizFlow.git
cd VizFlow/VizFlow

# Install dependencies
npm install

# Start development server
npm run dev
```

The application will be available at `http://localhost:5174`

#### Backend Setup

```bash
# Navigate to server directory
cd VizFlow/VizFlow/server

# Install backend dependencies
npm install

# Create .env file
cp .env.example .env

# Edit .env file with your MongoDB URI and JWT secret
# MONGODB_URI=mongodb://localhost:27017/vizflow
# JWT_SECRET=your_secure_secret_key

# Start MongoDB (if running locally)
# mongod

# Start backend server
npm run dev
```

The API will be available at `http://localhost:5000`

#### Database Setup

1. **Install MongoDB** (if not already installed):
   - Download from [https://www.mongodb.com/try/download/community](https://www.mongodb.com/try/download/community)
   - Or use MongoDB Atlas (cloud): [https://www.mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)

2. **Start MongoDB**:
   ```bash
   mongod
   ```

3. The database will be created automatically when you run the backend server

## 📁 Project Structure

```
vizflow/
├── src/
│   ├── components/          # Reusable React components
│   │   ├── Toolbar.jsx      # Top navigation bar
│   │   ├── CodeEditor.jsx   # Python code editor with console
│   │   ├── ModelVisualization.jsx  # Interactive graph visualization
│   │   └── SplitPane.jsx    # Responsive split layout
│   ├── hooks/               # Custom React hooks
│   │   └── useModelParser.js # Model parsing logic
│   ├── App.jsx              # Main app component
│   ├── main.jsx             # React entry point
│   └── index.css            # Global styles (Tailwind + Animations)
├── package.json
├── vite.config.js
├── tailwind.config.js
├── ANIMATION_SYSTEM.md      # Animation patterns & customization
├── DESIGN_ENHANCEMENTS.md   # Design system documentation
├── ENHANCEMENT_SUMMARY.md   # Implementation summary
└── README.md
```

## 🎨 Design System

### Color Palette
```
Primary:    #3b82f6 (Blue)
Secondary:  #06b6d4 (Cyan)
Background: #0f0f0f (Dark)
Surface:    #111827 (Gray-900)
Border:     #374151 (Gray-700)
Success:    #10b981 (Green)
```

### Components
All components are fully modular and reusable:
- **Toolbar** - Navigation and action buttons with animations
- **CodeEditor** - Python code input with live console
- **ModelVisualization** - Interactive layer visualization with zoom and animations
- **SplitPane** - Draggable split-pane layout with smooth transitions
- **Custom Hooks** - useModelParser for business logic

## 🎬 Animation System

VizFlow features a comprehensive animation system powered by **Framer Motion** and **React Hot Toast**:

### Features
- ✅ **Entrance Animations** - Smooth component load transitions
- ✅ **Hover Effects** - Interactive feedback on all buttons
- ✅ **Toast Notifications** - Contextual user feedback
- ✅ **Continuous Animations** - Loading spinners and pulses
- ✅ **Accessibility** - Respects `prefers-reduced-motion` preference

### Animation Examples
```jsx
// Smooth hover feedback
whileHover={{ scale: 1.05 }}
whileTap={{ scale: 0.95 }}

// Staggered button appearance
staggerChildren: 0.1
delayChildren: 0.2

// Toast notification
toast.success('🚀 Running model analysis...')
```

For detailed animation documentation, see [ANIMATION_SYSTEM.md](./ANIMATION_SYSTEM.md)

## 📚 Usage Examples

### Parse and Visualize a PyTorch Model

```python
import torch
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

model = MyNet()
```

1. Paste the code above into the editor
2. Click the **Run** button
3. View the model architecture in the visualization pane
4. Click any layer to inspect its parameters

## 🔄 Roadmap

### ✅ Completed (Current)
- Modular component architecture
- Interactive model visualization
- Layer parameter calculation
- Professional UI/UX

### 📋 Sprint 1 (Weeks 1-2)
- [ ] Monaco Editor integration
- [ ] Language Server Protocol (LSP)
- [ ] Pyodide Python runtime
- [ ] Autocompletion & error detection

### 📊 Sprint 2 (Weeks 3-4)
- [ ] D3.js force-directed graphs
- [ ] Export to PNG/SVG/JSON
- [ ] Weight distribution histograms
- [ ] Framework auto-detection

### 🔧 Sprint 3 (Weeks 5-6)
- [ ] FastAPI backend
- [ ] WebSocket support
- [ ] User authentication
- [ ] Project persistence

See [ROADMAP.md](ROADMAP.md) for detailed sprint information.

## 🛠️ Development

### Available Scripts

```bash
# Start development server with hot reload
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Run ESLint
npm run lint
```

### Tech Stack

- **Frontend Framework:** React 19
- **Build Tool:** Vite (Rolldown-based)
- **Styling:** Tailwind CSS 4
- **Icons:** Lucide React
- **Code Editor:** Textarea (Monaco planned)
- **Visualization:** SVG (D3.js planned)

### Dependencies

```json
{
  "dependencies": {
    "react": "^19.2.0",
    "react-dom": "^19.2.0",
    "lucide-react": "^0.554.0",
    "tailwindcss": "^4.1.17"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^5.1.0",
    "vite": "npm:rolldown-vite@7.2.2",
    "eslint": "^9.39.1"
  }
}
```

## 🧪 Testing

### Manual Testing Checklist

- [ ] **Model Parsing**
  - Linear layers
  - Conv2D layers
  - ReLU activations
  - Complex architectures

- [ ] **UI/UX**
  - Responsive split pane
  - Layer click selection
  - Zoom controls
  - Console output

- [ ] **Performance**
  - Large models (100+ layers)
  - Smooth animations
  - Memory usage
  - Export speed

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style
- Use functional components with hooks
- Follow React best practices
- Keep components small and focused
- Use Tailwind CSS for styling
- Document complex logic with comments

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

- **Lead Developer:** [Your Name]
- **Design:** Professional dark theme with gradient accents
- **Architecture:** Modular components + custom hooks

## 🙏 Acknowledgments

- [React Documentation](https://react.dev)
- [Tailwind CSS](https://tailwindcss.com)
- [Vite Documentation](https://vitejs.dev)
- [Lucide Icons](https://lucide.dev)

## 📞 Support

For support, email support@vizflow.dev or open an issue on GitHub.

---

**Last Updated:** November 17, 2025  
**Current Version:** 0.1.0-beta  
**Status:** Active Development
