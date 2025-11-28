# VizFlow Architecture Diagram

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        VizFlow Application                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
        ┌──────────────────┐  ┌──────────────────┐
        │   App.jsx        │  │  Tailwind CSS    │
        │  (Orchestrator)  │  │  (Dark Theme)    │
        └──────────────────┘  └──────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
   ┌────────┐  ┌────────┐  ┌────────┐
   │Toolbar │  │SplitPane  │  │  Data │
   │        │  │        │  │ Flow  │
   └────────┘  └────────┘  └────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
   ┌────────────┐           ┌─────────────────┐
   │  CodeEditor│           │ModelVisualization│
   │            │           │                 │
   │ • Input    │◄──────────► • Render        │
   │ • Syntax   │  Parser    │ • Interact     │
   │ • Console  │  Logic     │ • Zoom         │
   └────────────┘           │ • Inspector     │
                            └─────────────────┘
                                    │
                                    ▼
                            ┌──────────────────┐
                            │useModelParser    │
                            │                  │
                            │ • Parse layers   │
                            │ • Extract params │
                            │ • Map connections│
                            │ • Error handling │
                            └──────────────────┘
```

## 📊 Data Flow Diagram

```
User writes Python code
         │
         ▼
    CodeEditor
         │
         ▼ (onChange event)
    App.jsx state
         │
         ├──► [1. Display code]
         │
         └──► [2. On Run button]
              ▼
         useModelParser.executeCode()
              │
              ├──► Extract layer definitions
              │    (regex: nn.*)
              │
              ├──► Calculate parameters
              │    (Linear: in × out + out)
              │
              ├──► Parse forward method
              │    (trace layer calls)
              │
              ├──► Build connections
              │    (layer to layer flow)
              │
              └──► Return modelGraph
                   {layers: [], connections: []}
                        │
                        ▼
                   ModelVisualization
                        │
                        ├──► Render SVG
                        │    (layer boxes + connections)
                        │
                        ├──► Add interactivity
                        │    (click to select)
                        │
                        ├──► Show inspector
                        │    (parameters)
                        │
                        └──► Display statistics
                             (total params, layers, etc.)
                        │
                        ▼
                   User sees visual model!
```

## 🧩 Component Hierarchy

```
App.jsx
│
├── Toolbar
│   ├── Logo + Brand
│   ├── Run Button
│   ├── Save Button
│   ├── Share Button
│   ├── Download Button
│   └── Settings Button
│
├── SplitPane
│   │
│   ├── LEFT: CodeEditor
│   │   ├── File Tab (main.py)
│   │   ├── Textarea (code input)
│   │   └── Console Panel
│   │       └── Output Display
│   │
│   ├── DIVIDER (draggable)
│   │
│   └── RIGHT: ModelVisualization
│       ├── Header
│       │   ├── Title
│       │   └── Zoom Controls
│       ├── SVG Canvas
│       │   ├── Layer Boxes
│       │   └── Connection Lines
│       └── Inspector Panel
│           └── Layer Details
│
└── (Footer)
    └── Sprint Status
```

## 🔄 State Management Flow

```
                    App.jsx (Main State)
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
    [code]            [modelGraph]         [output]
    (string)          (object)             (string)
        │                  │                  │
        │                  │                  │
    ┌───▼──────┐       ┌───▼──────┐      ┌───▼──────┐
    │CodeEditor◄┼──────►│useModelParser  │Console   │
    │(onChange)│       │(parseModel)    │(display) │
    └──────────┘       └───┬──────┘      └──────────┘
                           │
                       ┌───▼──────────┐
                       │ModelVisualization
                       │(rendering)
                       └──────────────┘
```

## 📦 Module Dependencies

```
App.jsx
├── imports: Toolbar
├── imports: CodeEditor
├── imports: ModelVisualization
├── imports: SplitPane
└── imports: useModelParser

Toolbar.jsx
└── imports: lucide-react (icons)

CodeEditor.jsx
└── imports: lucide-react (FileCode icon)

ModelVisualization.jsx
└── imports: lucide-react (Grid, Info, ZoomIn, ZoomOut icons)

SplitPane.jsx
└── imports: lucide-react (Maximize2 icon)

useModelParser.js
└── imports: React (useState)

index.css
└── Tailwind CSS directives
```

## 🎨 Styling Architecture

```
Tailwind CSS (tailwind.config.js)
        │
        ├── Base Styles
        │   └── Reset + Typography
        │
        ├── Component Classes
        │   ├── .btn (buttons)
        │   ├── .card (cards)
        │   └── .input (inputs)
        │
        ├── Utility Classes
        │   ├── .bg-gray-900
        │   ├── .text-white
        │   ├── .rounded-lg
        │   └── .transition-all
        │
        └── Theme Colors
            ├── Primary (#3b82f6)
            ├── Secondary (#06b6d4)
            ├── Success (#10b981)
            └── Custom grays (#0f0f0f, #111827, etc.)
```

## 🔀 Event Flow

```
1. User Types Code
   └──► CodeEditor onChange
        └──► setCode(newCode)
             └──► Re-render

2. User Clicks Run
   └──► Toolbar onClick
        └──► handleRun()
             └──► executeCode()
                  └──► setModelGraph()
                       └──► ModelVisualization updates

3. User Clicks Layer
   └──► ModelVisualization onClick
        └──► setSelectedLayer()
             └──► Inspector appears
                  └──► Shows layer details

4. User Drags Divider
   └──► SplitPane onMouseDown
        └──► setIsDragging(true)
             └──► onMouseMove
                  └──► setSplitPos()
                       └──► Layout adjusts
```

## 🚀 Performance Optimization Points

```
Current Optimizations
├── SVG rendering (not DOM heavy)
├── Memoization ready
├── Efficient event handling
├── CSS gradients (GPU accelerated)
└── Minimal re-renders

Future Optimizations
├── Code splitting (Monaco)
├── Lazy loading (D3.js)
├── Web Workers (parsing)
├── Virtual scrolling
└── Progressive rendering
```

## 🔌 Extension Points

```
Sprint 1 - Monaco Editor
└── Replace CodeEditor.jsx
    └── Integrate @monaco-editor/react
        └── Add LSP support

Sprint 2 - D3 Visualization
└── Replace ModelVisualization SVG
    └── Build D3 force-directed graph
        └── Add animations

Sprint 3 - Backend
└── Add useBackend hook
    └── WebSocket integration
        └── API calls

Custom Frameworks
└── Extend useModelParser
    └── Add framework detection
        └── Support TensorFlow, Keras, etc.
```

## 📐 Responsive Breakpoints

```
Desktop (1200px+)
├── Toolbar: Full width
├── Split: 50/50 default
└── Console: 32px height

Tablet (768px - 1199px)
├── Toolbar: Compact
├── Split: 45/55 default
└── Console: 24px height

Mobile (<768px)
├── Stacked layout
├── Split: Tabs
└── Reduced console
```

---

This architecture ensures:
✅ Modularity - Each component handles one responsibility
✅ Maintainability - Clear dependencies and flow
✅ Scalability - Easy to add new features
✅ Performance - Optimized rendering and events
✅ Extensibility - Clear hooks for new functionality
