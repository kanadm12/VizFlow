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
      className="flex flex-col h-full bg-white/5 backdrop-blur-xl"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* File Tabs */}
      <motion.div
        className="bg-black/20 px-4 py-2 border-b border-white/10 flex items-center space-x-2 overflow-x-auto"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05 }}
      >
        {files.map(file => (
          <motion.button
            key={file.id}
            onClick={() => handleSwitchFile(file.id)}
            className={`px-3 py-1 rounded-md text-sm font-medium transition-all whitespace-nowrap ${
              activeFileId === file.id
                ? 'bg-blue-600 text-white'
                : 'bg-white/10 text-gray-300 hover:bg-white/20'
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
            className="px-2 py-1 rounded-md bg-white/10 text-gray-300 hover:bg-white/20 transition-all"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Plus className="w-4 h-4" />
          </motion.button>
          
          {showFileMenu && (
            <motion.div
              className="absolute top-full left-0 mt-2 bg-gray-800/80 backdrop-blur-lg rounded-lg border border-white/20 p-3 min-w-max z-50"
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
            >
              <input
                type="text"
                value={newFileName}
                onChange={(e) => setNewFileName(e.target.value)}
                placeholder="New file name..."
                className="w-full bg-gray-700/50 text-white px-3 py-2 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 border border-white/20"
                onKeyPress={(e) => e.key === 'Enter' && handleAddFile()}
              />
              <button
                onClick={handleAddFile}
                className="w-full mt-2 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-sm font-medium"
              >
                Add File
              </button>
            </motion.div>
          )}
        </motion.div>
      </motion.div>

      {/* Editor */}
      <div className="flex-grow flex relative">
        <div className="w-10 text-right pr-2 pt-2 text-gray-500 text-sm select-none">
          {getLineNumbers().map(n => (
            <div key={n}>{n}</div>
          ))}
        </div>
        <textarea
          ref={textareaRef}
          value={activeFile.content}
          onChange={handleCodeChange}
          className="flex-grow bg-transparent text-white p-2 font-mono text-sm focus:outline-none resize-none"
          spellCheck="false"
        />
      </div>

      {/* Output Console */}
      <motion.div
        className="bg-black/30 border-t border-white/10 p-4"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-bold text-gray-300">Output</h3>
          {aiProvider && (
            <motion.div className="relative">
              <motion.button
                onClick={() => setShowAiOptions(!showAiOptions)}
                className="flex items-center space-x-1 text-xs text-blue-400 hover:text-blue-300"
                whileHover={{ scale: 1.05 }}
              >
                <Sparkles className="w-3 h-3" />
                <span>AI Actions</span>
                <ChevronDown className="w-3 h-3" />
              </motion.button>
              {showAiOptions && (
                <motion.div
                  className="absolute bottom-full right-0 mb-2 bg-gray-800/80 backdrop-blur-lg rounded-lg border border-white/20 p-2 min-w-max z-50"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <button className="w-full text-left px-3 py-1.5 text-sm text-gray-300 hover:bg-white/10 rounded-md">Explain Code</button>
                  <button className="w-full text-left px-3 py-1.5 text-sm text-gray-300 hover:bg-white/10 rounded-md">Optimize Code</button>
                  <button className="w-full text-left px-3 py-1.5 text-sm text-gray-300 hover:bg-white/10 rounded-md">Find Bugs</button>
                </motion.div>
              )}
            </motion.div>
          )}
        </div>
        <pre className="text-xs text-gray-400 whitespace-pre-wrap font-mono h-24 overflow-y-auto">
          {output}
        </pre>
      </motion.div>
    </motion.div>
  );
};

export default CodeEditor;
