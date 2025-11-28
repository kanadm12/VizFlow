<div align="center">

# 🎨 VizFlow

### **AI/ML Model Architecture Visualizer & IDE**

<img src="https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=white" />
<img src="https://img.shields.io/badge/Node.js-20-339933?style=for-the-badge&logo=node.js&logoColor=white" />
<img src="https://img.shields.io/badge/MongoDB-7.0-47A248?style=for-the-badge&logo=mongodb&logoColor=white" />
<img src="https://img.shields.io/badge/Vite-5.0-646CFF?style=for-the-badge&logo=vite&logoColor=white" />

**A modern, full-stack IDE for visualizing neural network architectures with real-time code execution and intelligent terminal**

[Features](#-features) • [Quick Start](#-quick-start) • [Screenshots](#-screenshots) • [Tech Stack](#-tech-stack) • [API Docs](#-api-documentation)

</div>

---

## 🌟 Overview

VizFlow is a professional-grade web-based IDE that transforms how developers visualize and understand AI/ML model architectures. With an intuitive interface combining a code editor, interactive visualizer, and intelligent terminal, VizFlow makes complex neural networks easy to comprehend and debug.

## ✨ Features

### 🎯 Core IDE Features
- **Multi-View Workspace** - Switch between Full IDE, Editor+Visualizer, and Editor+Terminal modes
- **Live Code Editor** - Multi-file support with syntax highlighting and line numbers
- **Interactive Terminal** - Execute commands, view history, and navigate with arrow keys
- **Real-time Model Visualization** - Dynamic graph rendering with animated connections
- **Smart Layer Inspector** - Click any layer to see detailed parameter information
- **3D Animated Logo** - Eye-catching gradient effects with smooth animations

### 🔐 Authentication & User Management
- **Secure Login/Signup** - JWT-based authentication with bcrypt password hashing
- **Session Persistence** - Stay logged in across browser sessions
- **User Profiles** - Track AI provider preferences and usage statistics

### 💾 Cloud-Powered Workspace
- **Project Management** - Save, load, and organize multiple projects
- **Auto-Save** - Never lose your work with cloud synchronization
- **AI Provider Tracking** - Monitor which AI assistant (Claude, Gemini, ChatGPT) you use most
- **Execution History** - Review past runs with timestamps and outputs

### 🎨 Modern UI/UX
- **Glass-morphism Design** - Beautiful frosted glass effects throughout
- **Smooth Animations** - Framer Motion-powered transitions
- **Smart Notifications** - Context-aware toast messages
- **Dark Theme** - Easy on the eyes with vibrant accent colors
- **Responsive Layout** - Works seamlessly on all screen sizes

### 📊 Visualization Engine
- **Multiple Layout Algorithms** - DAG, Tree, Flowchart, and Force-directed graphs
- **Zoom & Pan Controls** - Navigate large architectures easily
- **Export Options** - Download as PNG, SVG, or HTML report
- **Connection Animations** - Flowing particles show data flow through layers
- **Layer Statistics** - Parameter counts, shapes, and types at a glance

## 🚀 Quick Start

### Prerequisites

Before you begin, ensure you have the following installed:
- **Node.js** 16+ ([Download](https://nodejs.org/))
- **MongoDB** 5.0+ ([Download](https://www.mongodb.com/try/download/community))
- **npm** or **yarn** (comes with Node.js)

### ⚡ Automated Setup (Recommended)

Run the included setup script for automatic installation:

```powershell
# Windows (PowerShell)
.\setup.ps1
```

```bash
# macOS/Linux
chmod +x setup.sh
./setup.sh
```

### 🔧 Manual Installation

#### Step 1: Clone the Repository

```bash
git clone https://github.com/kanadm12/VizFlow.git
cd VizFlow/VizFlow
```

#### Step 2: Setup Backend

```bash
# Navigate to server directory
cd server

# Install dependencies
npm install

# Create environment file
cp .env.example .env

# Edit .env and add your configuration:
# MONGO_URI=mongodb://localhost:27017/VizFlow
# PORT=5000
# JWT_SECRET=your-super-secret-jwt-key-here
```

#### Step 3: Setup Frontend

```bash
# Go back to root directory
cd ..

# Install frontend dependencies
npm install
```

#### Step 4: Start MongoDB

```bash
# Start MongoDB service
mongod

# Or if using MongoDB as a service:
# Windows: MongoDB runs automatically as a service
# macOS: brew services start mongodb-community
# Linux: sudo systemctl start mongod
```

#### Step 5: Launch Application

**Terminal 1 - Backend Server:**
```bash
cd server
npm run dev
# Server runs on http://localhost:5000
```

**Terminal 2 - Frontend App:**
```bash
npm run dev
# App runs on http://localhost:5174
```

### 🎉 You're Ready!

Open your browser and navigate to `http://localhost:5174`

**Default View:** You'll be greeted with the login/signup page. Create an account to start using VizFlow!

## 📁 Project Structure

```
VizFlow/
├── src/                          # Frontend source code
│   ├── components/               # React components
│   │   ├── AdvancedModelVisualization.jsx  # Main visualization engine
│   │   ├── AuthPage.jsx          # Login/signup interface
│   │   ├── CodeEditor.jsx        # Multi-file code editor
│   │   ├── Terminal.jsx          # Interactive terminal
│   │   ├── Toolbar.jsx           # Top navigation bar
│   │   └── SplitPane.jsx         # Resizable pane layout
│   ├── hooks/                    # Custom React hooks
│   │   └── useModelParser.js     # Model parsing logic
│   ├── utils/                    # Utility functions
│   │   ├── api.js                # API client (Axios)
│   │   ├── ExportUtils.js        # Export functionality
│   │   ├── GraphRenderer.js      # Graph rendering logic
│   │   └── LayoutDetector.js     # Layout algorithm detection
│   ├── App.jsx                   # Main application component
│   ├── main.jsx                  # React entry point
│   └── index.css                 # Global styles & animations
│
├── server/                       # Backend server
│   ├── controllers/              # Request handlers
│   │   ├── authController.js     # Authentication logic
│   │   ├── codeController.js     # Code execution & saving
│   │   └── userController.js     # User management
│   ├── models/                   # MongoDB schemas
│   │   ├── User.js               # User schema
│   │   └── Project.js            # Project schema
│   ├── routes/                   # API routes
│   │   ├── auth.js               # Auth endpoints
│   │   ├── code.js               # Code endpoints
│   │   └── user.js               # User endpoints
│   ├── middleware/               # Express middleware
│   │   └── auth.js               # JWT verification
│   ├── server.js                 # Express server setup
│   └── .env.example              # Environment template
│
├── public/                       # Static assets
├── package.json                  # Frontend dependencies
├── vite.config.js                # Vite configuration
├── tailwind.config.js            # Tailwind CSS config
└── README.md                     # This file
```

## 🛠️ Tech Stack

### Frontend
- **React 19** - UI framework with latest features
- **Vite 5** - Lightning-fast build tool
- **Tailwind CSS** - Utility-first CSS framework
- **Framer Motion** - Smooth animations
- **Axios** - HTTP client for API calls
- **React Hot Toast** - Beautiful notifications
- **Lucide React** - Modern icon library

### Backend
- **Node.js 20** - JavaScript runtime
- **Express.js** - Web application framework
- **MongoDB** - NoSQL database
- **Mongoose** - MongoDB object modeling
- **JWT** - JSON Web Tokens for auth
- **bcryptjs** - Password hashing
- **cors** - Cross-origin resource sharing
- **dotenv** - Environment variable management

## 🎨 Key Concepts

### View Modes
VizFlow offers three workspace layouts:
1. **Full IDE** - Editor + Visualizer (split) + Terminal (bottom)
2. **Editor + Visualizer** - Side-by-side code and graph
3. **Editor + Terminal** - Focus on coding and output

### Authentication Flow
```
User Sign Up → Password Hashed (bcrypt) → Stored in MongoDB
User Login → Credentials Verified → JWT Token Generated → Stored in localStorage
API Requests → Token Sent in Headers → Backend Validates → Returns Data
```

### Model Visualization
```
Python Code → Parser → Extract Layers → Detect Layout → Apply Algorithm → Render SVG
```

## 📚 Usage Guide

### Writing Model Code

VizFlow supports PyTorch and TensorFlow models:

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MyModel()
```

Click **Run** to visualize the architecture!

### Terminal Commands

The built-in terminal supports:
- `help` - Show available commands
- `clear` / `cls` - Clear terminal
- `ls` / `dir` - List files
- `pwd` - Show current directory
- `python --version` - Check Python version
- Arrow keys (↑/↓) - Navigate command history

### AI Provider Integration

Connect your preferred AI coding assistant:

1. Click the **Sparkles** (✨) icon in the toolbar
2. Select your AI provider (Claude, Gemini, or GitHub Copilot)
3. Enter your API key
4. Enable AI completion for code suggestions

VizFlow tracks which AI provider you use most frequently!

## 🔌 API Documentation

### Authentication Endpoints

#### Register User
```http
POST /api/auth/register
Content-Type: application/json

{
  "username": "johndoe",
  "email": "john@example.com",
  "password": "securepassword123"
}
```

#### Login User
```http
POST /api/auth/login
Content-Type: application/json

{
  "email": "john@example.com",
  "password": "securepassword123"
}
```

### Project Endpoints

#### Save Project
```http
POST /api/code/save
Authorization: Bearer <JWT_TOKEN>
Content-Type: application/json

{
  "projectId": "optional-existing-id",
  "name": "My Neural Network",
  "description": "CNN for image classification",
  "files": [
    {
      "name": "main.py",
      "content": "import torch...",
      "language": "python"
    }
  ],
  "aiProviderUsed": "claude"
}
```

#### Get User Projects
```http
GET /api/code/projects
Authorization: Bearer <JWT_TOKEN>
```

### User Endpoints

#### Update AI Provider
```http
PUT /api/user/ai-provider
Authorization: Bearer <JWT_TOKEN>
Content-Type: application/json

{
  "aiProvider": "claude",
  "aiApiKey": "sk-ant-..."
}
```

#### Get User Stats
```http
GET /api/user/stats
Authorization: Bearer <JWT_TOKEN>
```

## 🎯 Screenshots

### Main IDE Interface
![VizFlow IDE](https://via.placeholder.com/800x450/0f172a/3b82f6?text=VizFlow+IDE)

### Model Visualization
![Model Graph](https://via.placeholder.com/800x450/0f172a/06b6d4?text=Neural+Network+Visualization)

### Terminal Output
![Terminal](https://via.placeholder.com/800x450/0f172a/10b981?text=Interactive+Terminal)

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Ways to Contribute
- 🐛 Report bugs via [Issues](https://github.com/kanadm12/VizFlow/issues)
- 💡 Suggest new features
- 📝 Improve documentation
- 🔧 Submit pull requests

### Development Workflow

1. **Fork the repository**
```bash
git clone https://github.com/YOUR_USERNAME/VizFlow.git
cd VizFlow/VizFlow
```

2. **Create a feature branch**
```bash
git checkout -b feature/amazing-feature
```

3. **Make your changes**
```bash
# Install dependencies
npm install
cd server && npm install

# Start development
npm run dev
```

4. **Commit your changes**
```bash
git add .
git commit -m "Add amazing feature"
```

5. **Push to your fork**
```bash
git push origin feature/amazing-feature
```

6. **Open a Pull Request**

### Code Style
- Use ESLint for JavaScript/React
- Follow existing component patterns
- Write descriptive commit messages
- Add comments for complex logic

## 🐛 Troubleshooting

### Common Issues

**MongoDB Connection Error**
```bash
# Make sure MongoDB is running
mongod

# Check connection string in server/.env
MONGO_URI=mongodb://localhost:27017/VizFlow
```

**Port Already in Use**
```bash
# Frontend (5173)
npx kill-port 5173

# Backend (5000)
npx kill-port 5000
```

**JWT Token Invalid**
```bash
# Clear localStorage in browser console
localStorage.clear()

# Or delete specific items
localStorage.removeItem('token')
localStorage.removeItem('user')
```

**Module Not Found**
```bash
# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install

# Backend too
cd server
rm -rf node_modules package-lock.json
npm install
```

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 VizFlow

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## 🙏 Acknowledgments

- **React Team** - For the amazing framework
- **Vercel** - For Vite and incredible dev tools
- **Tailwind Labs** - For the best CSS framework
- **Framer** - For smooth animations
- **MongoDB** - For flexible data storage
- **OpenAI, Anthropic, Google** - For AI integrations

## 📞 Contact & Support

- **GitHub**: [@kanadm12](https://github.com/kanadm12)
- **Repository**: [VizFlow](https://github.com/kanadm12/VizFlow)
- **Issues**: [Report a Bug](https://github.com/kanadm12/VizFlow/issues)

## ⭐ Star History

If you find VizFlow helpful, please consider giving it a star on GitHub!

---

<div align="center">

**Built with ❤️ for the AI/ML community**

[⬆ Back to Top](#-vizflow)

</div>
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
