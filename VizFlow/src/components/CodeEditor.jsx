import React, { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { FileCode, Plus, ChevronDown, Sparkles } from 'lucide-react';

const CodeEditor = ({ code, onChange, output, aiProvider, aiApiKey }) => {
  const [files, setFiles] = useState([{ id: 1, name: 'main.py', content: code, active: true }]);
  const [activeFileId, setActiveFileId] = useState(1);
  const [showFileMenu, setShowFileMenu] = useState(false);
  const [newFileName, setNewFileName] = useState('');
  const [showAiOptions, setShowAiOptions] = useState(false);
  const textareaRef = useRef(null);

  const activeFile = files.find(f => f.id === activeFileId) || files[0];

  const handleAddFile = () => {
    if (newFileName.trim()) {
      const newFile = {
        id: Math.max(...files.map(f => f.id), 0) + 1,
        name: newFileName,
        content: '',
        active: false
      };
      setFiles([...files, newFile]);
      setActiveFileId(newFile.id);
      setNewFileName('');
      setShowFileMenu(false);
    }
  };

  const handleCodeChange = (e) => {
    const newContent = e.target.value;
    const updatedFiles = files.map(f =>
      f.id === activeFileId ? { ...f, content: newContent } : f
    );
    setFiles(updatedFiles);
    onChange(newContent);
  };

  const handleSwitchFile = (fileId) => {
    setActiveFileId(fileId);
    const file = files.find(f => f.id === fileId);
    onChange(file.content);
  };

  const getLineNumbers = () => {
    return activeFile.content.split('\n').map((_, i) => i + 1);
  };

  const containerVariants = {
    hidden: { opacity: 0, x: -20 },
    visible: {
      opacity: 1,
      x: 0,
      transition: { duration: 0.5, ease: 'easeOut' },
    },
  };

  return (
    <motion.div
      className="flex flex-col h-full bg-gray-900"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* File Tabs */}
      <motion.div
        className="bg-gray-800 px-4 py-2 border-b border-gray-700 flex items-center space-x-2 overflow-x-auto"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05 }}
      >
        {files.map(file => (
          <motion.button
            key={file.id}
            onClick={() => handleSwitchFile(file.id)}
            className={`px-3 py-1 rounded text-sm font-medium transition-all whitespace-nowrap ${
              activeFileId === file.id
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {file.name}
          </motion.button>
        ))}
        
        <motion.div className="relative">
          <motion.button
            onClick={() => setShowFileMenu(!showFileMenu)}
            className="px-2 py-1 rounded bg-gray-700 text-gray-300 hover:bg-gray-600 transition-all"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Plus className="w-4 h-4" />
          </motion.button>
          
          {showFileMenu && (
            <motion.div
              className="absolute top-full left-0 mt-2 bg-gray-800 rounded-lg border border-gray-600 p-3 min-w-max z-50"
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <input
                type="text"
                value={newFileName}
                onChange={(e) => setNewFileName(e.target.value)}
                placeholder="filename.py"
                className="bg-gray-700 text-white px-2 py-1 rounded text-sm w-48 focus:outline-none focus:ring-2 focus:ring-blue-500"
                onKeyPress={(e) => e.key === 'Enter' && handleAddFile()}
              />
              <motion.button
                onClick={handleAddFile}
                className="mt-2 w-full bg-blue-600 hover:bg-blue-700 text-white px-2 py-1 rounded text-sm transition-all"
                whileHover={{ scale: 1.02 }}
              >
                Create File
              </motion.button>
            </motion.div>
          )}
        </motion.div>
      </motion.div>

      {/* Editor Header with AI Options */}
      <motion.div
        className="bg-gray-800 px-4 py-3 text-sm border-b border-gray-700 flex items-center justify-between"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div className="flex items-center space-x-2">
          <motion.div whileHover={{ scale: 1.1, rotate: 5 }}>
            <FileCode className="w-4 h-4 text-blue-400" />
          </motion.div>
          <span className="text-gray-300 font-medium">{activeFile.name}</span>
          <span className="text-xs text-gray-500">UTF-8 • Python</span>
        </div>

        {/* AI Completion Toggle */}
        {aiProvider && (
          <motion.div className="relative">
            <motion.button
              onClick={() => setShowAiOptions(!showAiOptions)}
              className="flex items-center space-x-1 px-3 py-1 rounded bg-blue-600 hover:bg-blue-700 text-white text-xs transition-all"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Sparkles className="w-3 h-3" />
              <span>{aiProvider.toUpperCase()}</span>
              <ChevronDown className="w-3 h-3" />
            </motion.button>

            {showAiOptions && (
              <motion.div
                className="absolute top-full right-0 mt-2 bg-gray-800 rounded-lg border border-gray-600 p-3 min-w-max z-50 text-xs"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <div className="text-gray-400 mb-2">API Key: {aiApiKey ? '✓ Configured' : '✗ Not set'}</div>
                <div className="text-gray-500 text-xs">Auto-completion ready</div>
              </motion.div>
            )}
          </motion.div>
        )}
      </motion.div>

      {/* Code Editor with Line Numbers */}
      <motion.div
        className="flex-1 flex bg-gray-900 overflow-hidden"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.15 }}
      >
        {/* Line Numbers */}
        <div className="bg-gray-950 border-r border-gray-700 px-3 py-4 select-none overflow-hidden">
          {getLineNumbers().map((lineNum) => (
            <div
              key={lineNum}
              className="text-right text-xs text-gray-600 font-mono"
              style={{ lineHeight: '1.6', height: '1.6em' }}
            >
              {lineNum}
            </div>
          ))}
        </div>

        {/* Textarea */}
        <motion.textarea
          ref={textareaRef}
          value={activeFile.content}
          onChange={handleCodeChange}
          className="flex-1 bg-gray-900 text-gray-100 p-4 font-mono text-sm resize-none focus:outline-none w-full border-none focus:ring-0"
          style={{
            lineHeight: '1.6',
            tabSize: 4,
            backgroundColor: '#111827',
            color: '#f3f4f6'
          }}
          spellCheck={false}
        />
      </motion.div>

      {/* Console Output */}
      <motion.div
        className="console-output border-t border-gray-700 p-4 h-32 overflow-auto w-full"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.25 }}
      >
        <div className="flex items-center space-x-2 mb-2">
          <motion.div
            className="w-2 h-2 bg-green-500 rounded-full"
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          />
          <span className="text-xs text-gray-500 font-semibold">CONSOLE OUTPUT</span>
        </div>
        <motion.div
          className="text-xs font-mono text-green-400 whitespace-pre-wrap font-medium"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.35 }}
        >
          {output || '> Ready to execute code...'}
        </motion.div>
      </motion.div>
    </motion.div>
  );
};

export default CodeEditor;
