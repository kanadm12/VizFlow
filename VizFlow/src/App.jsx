import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Toaster } from 'react-hot-toast';
import toast from 'react-hot-toast';
import './index.css';
import Toolbar from './components/Toolbar';
import CodeEditor from './components/CodeEditor';
import AdvancedModelVisualization from './components/AdvancedModelVisualization';
import SplitPane from './components/SplitPane';
import Terminal from './components/Terminal';
import AuthPage from './components/AuthPage';
import useModelParser from './hooks/useModelParser';
import { codeAPI, userAPI } from './utils/api';

const App = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);
  const [currentProject, setCurrentProject] = useState(null);
  const [terminalMinimized, setTerminalMinimized] = useState(false);
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
  const fileInputRef = React.useRef(null);

  // Check authentication on mount
  useEffect(() => {
    const token = localStorage.getItem('token');
    const savedUser = localStorage.getItem('user');
    
    if (token && savedUser) {
      setIsAuthenticated(true);
      setUser(JSON.parse(savedUser));
    }
  }, []);

  const handleLogin = (userData, token) => {
    setIsAuthenticated(true);
    setUser(userData);
    setAiProvider(userData.aiProvider);
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setIsAuthenticated(false);
    setUser(null);
    toast.success('Logged out successfully');
  };

  const handleRun = async () => {
    setOutput('Executing code...\n');
    
    try {
      executeCode(code);
      
      // Simulate execution completion
      setTimeout(async () => {
        if (modelGraph) {
          const totalParams = modelGraph.layers.reduce((sum, l) => sum + l.trainableParams, 0);
          const outputText = `✓ Model execution successful
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Model Statistics:
   • Layers: ${modelGraph.layers.length}
   • Connections: ${modelGraph.connections.length}
   • Total Parameters: ${totalParams.toLocaleString()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ Visualization updated | Click layers for details
`;
          setOutput(outputText);

          // Save execution to backend
          if (currentProject && isAuthenticated) {
            try {
              await codeAPI.saveExecution({
                projectId: currentProject._id,
                output: outputText,
                success: true
              });
            } catch (error) {
              console.error('Failed to save execution:', error);
            }
          }
        }
      }, 1000);
    } catch (err) {
      const errorOutput = `❌ Execution Error: ${err}`;
      setOutput(errorOutput);
      
      // Save error to backend
      if (currentProject && isAuthenticated) {
        try {
          await codeAPI.saveExecution({
            projectId: currentProject._id,
            output: errorOutput,
            success: false
          });
        } catch (error) {
          console.error('Failed to save execution:', error);
        }
      }
    }
  };

  const handleSave = async () => {
    if (!isAuthenticated) {
      toast.error('Please login to save your work');
      return;
    }

    try {
      const projectData = {
        projectId: currentProject?._id,
        name: currentProject?.name || 'Untitled Project',
        description: currentProject?.description || '',
        files: [{ name: 'main.py', content: code, language: 'python' }],
        aiProviderUsed: aiProvider || 'none'
      };

      const response = await codeAPI.saveProject(projectData);
      setCurrentProject(response.data.project);
      toast.success('Project saved successfully!');
      setOutput('> Project saved to cloud');
    } catch (error) {
      toast.error('Failed to save project');
      console.error('Save error:', error);
    }
  };

  const handleShare = () => {
    console.log('Share functionality - Sprint 3 Backend Integration');
    setOutput('> Share link generated (coming soon)');
  };

  const handleUpload = () => {
    fileInputRef.current.click();
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target.result;
        setCode(content);
        toast.success(`File "${file.name}" loaded successfully!`);
      };
      reader.readAsText(file);
    }
  };

  const handleAiProviderChange = async (provider) => {
    setAiProvider(provider);
    
    if (isAuthenticated) {
      try {
        await userAPI.updateAiProvider({ aiProvider: provider });
      } catch (error) {
        console.error('Failed to update AI provider:', error);
      }
    }
  };

  const handleAiApiKeyChange = async (key) => {
    setAiApiKey(key);
    
    if (isAuthenticated && aiProvider) {
      try {
        await userAPI.updateAiProvider({ aiProvider, aiApiKey: key });
      } catch (error) {
        console.error('Failed to update API key:', error);
      }
    }
  };

  // Show auth page if not authenticated
  if (!isAuthenticated) {
    return <AuthPage onLogin={handleLogin} />;
  }

  return (
    <div className="h-screen bg-transparent text-white flex flex-col overflow-hidden">
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
        accept=".py,.js,.txt"
      />
      {/* Toast Notifications */}
      <Toaster
        position="top-right"
        reverseOrder={false}
        gutter={8}
        toastOptions={{
          duration: 2000,
          style: {
            background: 'rgba(255, 255, 255, 0.1)',
            backdropFilter: 'blur(10px)',
            color: '#fff',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '0.5rem',
            boxShadow: '0 10px 25px rgba(0, 0, 0, 0.2)',
          },
          success: {
            style: {
              background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(22, 163, 74, 0.3))',
              borderColor: 'rgba(52, 211, 153, 0.5)',
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
        onUpload={handleUpload}
        onLogout={handleLogout}
        user={user}
        aiProvider={aiProvider}
        onAiProviderChange={handleAiProviderChange}
        aiApiKey={aiApiKey}
        onAiApiKeyChange={handleAiApiKeyChange}
      />

      {/* Main Content Area with Terminal */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-1 overflow-hidden">
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
        
        {/* Terminal Window */}
        <div className={`${terminalMinimized ? 'h-0' : 'h-64'} transition-all duration-300`}>
          <Terminal
            output={output}
            onClear={() => setOutput('')}
            isMinimized={terminalMinimized}
            onToggleMinimize={() => setTerminalMinimized(!terminalMinimized)}
          />
        </div>
      </div>
    </div>
  );
};

export default App;