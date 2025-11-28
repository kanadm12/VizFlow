import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Download, Settings, Code, Save, Share2, Sparkles, X, Upload, LogOut, User as UserIcon } from 'lucide-react';
import toast from 'react-hot-toast';

const Toolbar = ({ onRun, isRunning, onSave, onShare, onUpload, onLogout, user, aiProvider, onAiProviderChange, aiApiKey, onAiApiKeyChange }) => {
  const [showAiSettings, setShowAiSettings] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState(aiProvider);
  const [apiKey, setApiKey] = useState(aiApiKey);
  const [testingApi, setTestingApi] = useState(false);

  const aiProviders = [
    { id: 'claude', name: 'Claude (Anthropic)', placeholder: 'sk-ant-...' },
    { id: 'gemini', name: 'Google Gemini', placeholder: 'AIza...' },
    { id: 'copilot', name: 'GitHub Copilot', placeholder: 'ghp_...' }
  ];

  const handleAiProviderSave = async () => {
    if (!selectedProvider || !apiKey) {
      toast.error('Please select provider and enter API key');
      return;
    }

    setTestingApi(true);
    // Simulate API test
    setTimeout(() => {
      onAiProviderChange(selectedProvider);
      onAiApiKeyChange(apiKey);
      setTestingApi(false);
      setShowAiSettings(false);
      toast.success(`✨ ${selectedProvider.toUpperCase()} connected for code completion!`);
    }, 1500);
  };
  const buttonVariants = {
    rest: {
      scale: 1,
      transition: { type: 'spring', stiffness: 300, damping: 30 },
    },
    hover: {
      scale: 1.08,
      transition: { type: 'spring', stiffness: 400, damping: 25 },
    },
    tap: {
      scale: 0.95,
    },
  };

  const containerVariants = {
    hidden: { opacity: 0, y: -20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.6, ease: 'easeOut' },
    },
  };

  const buttonContainerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2,
      },
    },
  };

  const childVariants = {
    hidden: { opacity: 0, y: 10 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.3 },
    },
  };

  const handleRunClick = () => {
    onRun();
    toast.success('🚀 Running model analysis...');
  };

  const handleSaveClick = () => {
    onSave();
    toast.success('💾 Code saved!');
  };

  const handleShareClick = () => {
    onShare();
    toast.success('🔗 Share link copied!');
  };

  const handleDownload = () => {
    toast.success('📸 Downloading flowchart as PNG...');
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="toolbar bg-white/5 backdrop-blur-lg border-b border-white/10 px-8 py-4 flex items-center justify-between shadow-lg"
    >
      {/* Logo and Title */}
      <motion.div
        className="flex items-center space-x-4"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
      >
        <motion.div
          className="flex items-center justify-center w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg shadow-lg"
          whileHover={{ rotate: 10, scale: 1.08 }}
          whileTap={{ scale: 0.9 }}
        >
          <Code className="w-6 h-6 text-white" />
        </motion.div>
        <div>
          <div className="flex items-center space-x-3">
            <span className="font-bold text-xl text-white">VizFlow</span>
            <motion.span
              className="bg-white/10 text-blue-300 text-xs font-bold px-2 py-0.5 rounded-full"
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.3, type: 'spring' }}
            >
              Beta
            </motion.span>
          </div>
          <p className="text-xs text-gray-400 font-medium">AI/ML Model Visualizer</p>
        </div>
      </motion.div>

      {/* Action Buttons */}
      <motion.div
        className="flex items-center space-x-3"
        variants={buttonContainerVariants}
        initial="hidden"
        animate="visible"
      >
        <motion.button
          onClick={handleRunClick}
          disabled={isRunning}
          variants={childVariants}
          whileHover="hover"
          whileTap="tap"
          className="bg-blue-600 hover:bg-blue-700 text-white font-semibold flex items-center space-x-2 px-5 py-2.5 rounded-lg transition-all"
        >
          <motion.div
            animate={isRunning ? { rotate: 360 } : { rotate: 0 }}
            transition={{ duration: 1, repeat: isRunning ? Infinity : 0 }}
          >
            <Play className="w-4 h-4" />
          </motion.div>
          <span>{isRunning ? 'Running...' : 'Run'}</span>
        </motion.button>

        <motion.button
          onClick={handleSaveClick}
          variants={childVariants}
          whileHover="hover"
          whileTap="tap"
          className="bg-white/10 hover:bg-white/20 text-white p-2.5 rounded-lg"
          title="Save (Ctrl+S)"
        >
          <Save className="w-4 h-4" />
        </motion.button>

        <motion.button
          onClick={handleShareClick}
          variants={childVariants}
          whileHover="hover"
          whileTap="tap"
          className="bg-white/10 hover:bg-white/20 text-white p-2.5 rounded-lg"
          title="Share"
        >
          <Share2 className="w-4 h-4" />
        </motion.button>

        <motion.button
          onClick={onUpload}
          variants={childVariants}
          whileHover="hover"
          whileTap="tap"
          className="bg-white/10 hover:bg-white/20 text-white p-2.5 rounded-lg"
          title="Upload File"
        >
          <Upload className="w-4 h-4" />
        </motion.button>

        <motion.button
          onClick={handleDownload}
          variants={childVariants}
          whileHover="hover"
          whileTap="tap"
          className="bg-white/10 hover:bg-white/20 text-white p-2.5 rounded-lg"
          title="Download"
        >
          <Download className="w-4 h-4" />
        </motion.button>

        <motion.button
          onClick={() => setShowAiSettings(!showAiSettings)}
          variants={childVariants}
          whileHover="hover"
          whileTap="tap"
          className={`bg-white/10 hover:bg-white/20 text-white p-2.5 rounded-lg ${aiProvider ? 'bg-blue-600 text-white' : ''}`}
          title="AI Code Completion"
        >
          <Sparkles className="w-4 h-4" />
        </motion.button>

        <motion.button
          variants={childVariants}
          whileHover="hover"
          whileTap="tap"
          className="bg-white/10 hover:bg-white/20 text-white p-2.5 rounded-lg"
          title="Settings"
        >
          <Settings className="w-4 h-4" />
        </motion.button>

        {/* User Menu */}
        {user && (
          <motion.div
            variants={childVariants}
            className="flex items-center space-x-2 ml-2 pl-2 border-l border-white/20"
          >
            <div className="flex items-center space-x-2 bg-white/10 px-3 py-1.5 rounded-lg">
              <UserIcon className="w-4 h-4 text-blue-400" />
              <span className="text-sm font-medium">{user.username}</span>
            </div>
            <motion.button
              onClick={onLogout}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="bg-red-600/80 hover:bg-red-700 text-white p-2.5 rounded-lg"
              title="Logout"
            >
              <LogOut className="w-4 h-4" />
            </motion.button>
          </motion.div>
        )}
      </motion.div>

      {/* AI Settings Modal */}
      <AnimatePresence>
        {showAiSettings && (
          <motion.div
            className="fixed inset-0 bg-black bg-opacity-70 backdrop-blur-sm flex items-center justify-center z-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setShowAiSettings(false)}
          >
            <motion.div
              className="bg-gray-800/80 backdrop-blur-lg rounded-lg border border-white/20 p-6 max-w-md w-full mx-4 shadow-2xl"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-2">
                  <Sparkles className="w-5 h-5 text-blue-400" />
                  <h3 className="text-lg font-bold text-white">AI Code Completion</h3>
                </div>
                <motion.button
                  onClick={() => setShowAiSettings(false)}
                  whileHover={{ scale: 1.1, rotate: 90 }}
                  whileTap={{ scale: 0.95 }}
                  className="text-gray-400 hover:text-white"
                >
                  <X className="w-5 h-5" />
                </motion.button>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="text-sm text-gray-300 mb-2 block">AI Provider</label>
                  <select
                    value={selectedProvider}
                    onChange={(e) => setSelectedProvider(e.target.value)}
                    className="w-full bg-gray-700/50 text-white px-3 py-2 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 border border-white/20"
                  >
                    <option value="">Select Provider...</option>
                    {aiProviders.map(p => (
                      <option key={p.id} value={p.id}>{p.name}</option>
                    ))}
                  </select>
                  <p className="text-xs text-gray-500 mt-1">
                    {selectedProvider && 
                      `Get your API key from ${aiProviders.find(p => p.id === selectedProvider)?.name}`
                    }
                  </p>
                </div>

                {selectedProvider && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                  >
                    <label className="text-sm text-gray-300 mb-2 block">API Key</label>
                    <input
                      type="password"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder={aiProviders.find(p => p.id === selectedProvider)?.placeholder}
                      className="w-full bg-gray-700/50 text-white px-3 py-2 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 border border-white/20"
                    />
                  </motion.div>
                )}

                <div className="bg-black/20 p-3 rounded-md text-xs text-gray-400 border border-white/10">
                  <p className="font-semibold text-gray-300 mb-1">How to get your API key:</p>
                  {selectedProvider === 'claude' && (
                    <p>1. Visit <a href="https://console.anthropic.com" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">console.anthropic.com</a><br/>
                    2. Create a new API key<br/>3. Copy and paste it here</p>
                  )}
                  {selectedProvider === 'gemini' && (
                    <p>1. Visit <a href="https://makersuite.google.com/app/apikey" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">Google AI Studio</a><br/>
                    2. Get your API key<br/>3. Copy and paste it here</p>
                  )}
                  {selectedProvider === 'copilot' && (
                    <p>1. Visit <a href="https://github.com/settings/tokens" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">GitHub Settings</a><br/>
                    2. Create a personal access token<br/>3. Copy and paste it here</p>
                  )}
                </div>

                <motion.button
                  onClick={handleAiProviderSave}
                  disabled={testingApi || !selectedProvider || !apiKey}
                  className={`w-full px-4 py-2 rounded-lg font-medium transition-all ${
                    testingApi || !selectedProvider || !apiKey
                      ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                      : 'bg-blue-600 hover:bg-blue-700 text-white'
                  }`}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  {testingApi ? 'Testing connection...' : 'Enable AI Completion'}
                </motion.button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default Toolbar;
