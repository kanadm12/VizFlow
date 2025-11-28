# VizFlow - Professional AI/ML Model Visualizer

## 🚀 Current Status: Beta Release with Modular Architecture

### ✅ Completed (Current Sprint 0)
- **Modular Component Architecture**
  - `Toolbar.jsx` - Enhanced navigation with Save/Share buttons
  - `CodeEditor.jsx` - Python code editor with console output
  - `ModelVisualization.jsx` - Interactive layer visualization with zoom & inspector
  - `SplitPane.jsx` - Responsive split layout component
  - `useModelParser.js` - Custom hook for model parsing logic

- **Professional UI/UX**
  - Dark theme with gradient accents (blue/cyan)
  - Responsive split-pane editor
  - Layer selection & inspection panel
  - Zoom controls for visualization
  - Professional spacing & transitions

- **Core Functionality**
  - Parse PyTorch model definitions
  - Visualize layer architecture
  - Display model statistics
  - Console output for execution feedback

---

## 🎯 Sprint 1: Advanced Editor (Sprint 1 - Weeks 1-2)

### Features to Implement
- [ ] **Monaco Editor Integration**
  - Replace textarea with full VS Code experience
  - Syntax highlighting for Python/PyTorch/TensorFlow
  - Line numbers, minimap, code folding
  - Multi-file support

- [ ] **Language Server Protocol (LSP)**
  - Autocompletion for PyTorch/TensorFlow APIs
  - Real-time error detection
  - Type hints & documentation
  - Go to definition

- [ ] **Code Execution**
  - Pyodide for client-side Python execution
  - Environment setup for ML frameworks
  - Real output capture & display
  - Interrupt/stop execution

### Dependencies to Install
```bash
npm install @monaco-editor/react pyodide
```

### Implementation Files
- `src/components/Editor.jsx` - New Monaco-based editor
- `src/services/pythonRuntime.js` - Pyodide execution engine
- `src/hooks/useEditor.js` - Editor state management

---

## 📊 Sprint 2: Advanced Visualization (Sprint 2 - Weeks 3-4)

### Features to Implement
- [ ] **D3.js Interactive Graphs**
  - Replace SVG with D3.js force-directed graph
  - Better layer positioning & flow
  - Smooth animations & transitions
  - Pan & zoom with mouse wheel

- [ ] **Layer Inspector Enhancements**
  - Click to inspect layer details
  - Weight distribution histogram
  - Activation patterns
  - Parameter breakdown

- [ ] **Export Functionality**
  - Export to PNG (use html2canvas)
  - Export to SVG (vector format)
  - Export to JSON (model architecture)
  - Copy as image feature

- [ ] **Framework Detection**
  - Auto-detect PyTorch vs TensorFlow vs Keras
  - Framework-specific parsing & visualization
  - Support for different layer types

### Dependencies to Install
```bash
npm install d3 html2canvas
```

### Implementation Files
- `src/components/D3Visualization.jsx` - D3.js graph
- `src/utils/frameworkDetector.js` - Auto-detection logic
- `src/hooks/useVisualization.js` - Visualization state

---

## 🔧 Sprint 3: Backend Integration (Sprint 3 - Weeks 5-6)

### Features to Implement
- [ ] **FastAPI Backend**
  - Model parsing service
  - Code validation & optimization
  - Framework detection service
  - Export generation service

- [ ] **WebSocket Communication**
  - Real-time code updates
  - Live model architecture sync
  - Collaborative editing foundation
  - Performance metrics streaming

- [ ] **Authentication & Storage**
  - User accounts (optional)
  - Save/load projects
  - Share with links
  - Version history

- [ ] **Advanced Features**
  - Model optimization suggestions
  - Performance profiling
  - Memory usage analysis
  - Batch size recommendations

### Backend Stack
```
FastAPI + Uvicorn
WebSocket support
SQLite/PostgreSQL for storage
```

### Implementation Files
- `backend/main.py` - FastAPI server
- `src/services/websocket.js` - WebSocket client
- `src/hooks/useBackend.js` - Backend integration

---

## 📁 Project Structure

```
VizFlow/
├── src/
│   ├── components/
│   │   ├── Toolbar.jsx              ✅ Done
│   │   ├── CodeEditor.jsx           ✅ Done
│   │   ├── ModelVisualization.jsx   ✅ Done
│   │   ├── SplitPane.jsx            ✅ Done
│   │   ├── Editor.jsx               📋 Sprint 1
│   │   └── D3Visualization.jsx      📋 Sprint 2
│   ├── hooks/
│   │   ├── useModelParser.js        ✅ Done
│   │   ├── useEditor.js             📋 Sprint 1
│   │   └── useBackend.js            📋 Sprint 3
│   ├── services/
│   │   ├── pythonRuntime.js         📋 Sprint 1
│   │   ├── frameworkDetector.js     📋 Sprint 2
│   │   └── websocket.js             📋 Sprint 3
│   ├── utils/
│   │   └── export.js                📋 Sprint 2
│   ├── App.jsx                      ✅ Done
│   ├── main.jsx                     ✅ Done
│   └── index.css                    ✅ Done
├── backend/                         📋 Sprint 3
│   ├── main.py
│   ├── requirements.txt
│   └── services/
├── public/
├── package.json
└── README.md

```

---

## 🔄 Component Flow

```
App.jsx
├── Toolbar
│   ├── Run Button → executeCode()
│   ├── Save Button → saveProject()
│   └── Share Button → generateLink()
├── SplitPane
│   ├── Left: CodeEditor
│   │   ├── Monaco Editor (Sprint 1)
│   │   └── Console Output
│   └── Right: ModelVisualization
│       ├── D3 Graph (Sprint 2)
│       └── Layer Inspector
└── useModelParser (custom hook)
    ├── parseModel()
    ├── executeCode()
    └── error handling
```

---

## 🎨 Design System

### Color Palette
- Primary: `#3b82f6` (Blue)
- Secondary: `#06b6d4` (Cyan)
- Background: `#0f0f0f` (Dark)
- Surface: `#111827` (Gray-900)
- Border: `#374151` (Gray-700)
- Success: `#10b981` (Green)

### Typography
- Font Family: System fonts (-apple-system, BlinkMacSystemFont, etc.)
- Code: Monospace (Monaco, Courier New)
- Sizes: 12px (xs), 14px (sm), 16px (base)

---

## 🚦 Next Steps

1. **Immediate (This Week)**
   - ✅ Deploy current modular version
   - Test with various PyTorch models
   - Gather user feedback

2. **Sprint 1 (Next 2 Weeks)**
   - Install Monaco Editor
   - Integrate Pyodide runtime
   - Build LSP integration

3. **Sprint 2 (Following 2 Weeks)**
   - Implement D3.js visualization
   - Add export functionality
   - Framework detection

4. **Sprint 3 (Final 2 Weeks)**
   - FastAPI backend setup
   - WebSocket implementation
   - Authentication & storage

---

## 📝 Development Notes

- All components are reusable and follow React best practices
- Hooks encapsulate business logic (parser, editor state, etc.)
- Tailwind CSS for styling consistency
- No external UI libraries (fully custom)
- Support for future framework additions

---

## 🤝 Contributing

When adding new features:
1. Create new component in `src/components/`
2. Create custom hooks in `src/hooks/`
3. Keep components small and focused
4. Document props and usage
5. Follow existing code style

---

**Last Updated:** November 17, 2025
**Status:** Beta Release - Ready for Sprint 1 Development
