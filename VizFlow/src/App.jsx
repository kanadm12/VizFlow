import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Toaster } from 'react-hot-toast';
import './index.css';
import Toolbar from './components/Toolbar';
import CodeEditor from './components/CodeEditor';
import AdvancedModelVisualization from './components/AdvancedModelVisualization';
import SplitPane from './components/SplitPane';
import useModelParser from './hooks/useModelParser';

const App = () => {
  const [code, setCode] = useState(`import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    """Neural Network Architecture for Classification"""
    def __init__(self):
        super(SimpleNet, self).__init__()
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

model = SimpleNet()
`);

  const [output, setOutput] = useState('> Ready to execute code...');
  const [aiProvider, setAiProvider] = useState(null);
  const [aiApiKey, setAiApiKey] = useState('');
  const { modelGraph, executeCode, isLoading } = useModelParser();

  const handleRun = () => {
    setOutput('Executing code...\n');
    
    try {
      executeCode(code);
      
      // Simulate execution completion
      setTimeout(() => {
        if (modelGraph) {
          const totalParams = modelGraph.layers.reduce((sum, l) => sum + l.trainableParams, 0);
          setOutput(`✓ Model execution successful
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Model Statistics:
   • Layers: ${modelGraph.layers.length}
   • Connections: ${modelGraph.connections.length}
   • Total Parameters: ${totalParams.toLocaleString()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ Visualization updated | Click layers for details
`);
        }
      }, 1000);
    } catch (err) {
      setOutput(`❌ Execution Error: ${err}`);
    }
  };

  const handleSave = () => {
    console.log('Save functionality - Sprint 3 Backend Integration');
    setOutput('> Code saved (local storage)');
  };

  const handleShare = () => {
    console.log('Share functionality - Sprint 3 Backend Integration');
    setOutput('> Share link generated (coming soon)');
  };

  return (
    <div className="h-screen bg-gray-900 text-white flex flex-col overflow-hidden">
      {/* Toast Notifications */}
      <Toaster
        position="top-right"
        reverseOrder={false}
        gutter={8}
        toastOptions={{
          duration: 2000,
          style: {
            background: '#1a1a2e',
            color: '#fff',
            border: '1px solid #3b82f6',
            borderRadius: '0.5rem',
            boxShadow: '0 10px 25px rgba(59, 130, 246, 0.3)',
          },
          success: {
            style: {
              background: 'linear-gradient(135deg, #1a1a2e, #0f172a)',
              borderColor: '#10b981',
            },
          },
        }}
      />
      {/* Top Navigation */}
      <Toolbar 
        onRun={handleRun} 
        isRunning={isLoading}
        onSave={handleSave}
        onShare={handleShare}
        aiProvider={aiProvider}
        onAiProviderChange={setAiProvider}
        aiApiKey={aiApiKey}
        onAiApiKeyChange={setAiApiKey}
      />

      {/* Main Content Area */}
      <SplitPane
        left={
          <CodeEditor 
            code={code} 
            onChange={setCode}
            output={output}
            aiProvider={aiProvider}
            aiApiKey={aiApiKey}
          />
        }
        right={
          <AdvancedModelVisualization modelGraph={modelGraph} />
        }
        defaultSplit={45}
        minSize={30}
        maxSize={70}
      />
    </div>
  );
};

export default App;