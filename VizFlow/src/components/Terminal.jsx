import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Terminal as TerminalIcon, X, Trash2, Copy, ChevronUp } from 'lucide-react';
import toast from 'react-hot-toast';

const Terminal = ({ output, onClear, isMinimized, onToggleMinimize }) => {
  const terminalRef = useRef(null);
  const [commandHistory, setCommandHistory] = useState([]);

  useEffect(() => {
    if (terminalRef.current && !isMinimized) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [output, isMinimized]);

  useEffect(() => {
    if (output) {
      setCommandHistory(prev => [...prev, { timestamp: new Date(), output }]);
    }
  }, [output]);

  const handleCopy = () => {
    navigator.clipboard.writeText(output);
    toast.success('Output copied to clipboard!');
  };

  const handleClearHistory = () => {
    setCommandHistory([]);
    onClear();
    toast.success('Terminal cleared!');
  };

  if (isMinimized) {
    return (
      <motion.div
        initial={{ y: 100 }}
        animate={{ y: 0 }}
        className="fixed bottom-4 right-4 z-50"
      >
        <motion.button
          onClick={onToggleMinimize}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="bg-gray-800/90 backdrop-blur-lg border border-white/20 rounded-lg px-4 py-3 shadow-lg flex items-center space-x-2 text-white"
        >
          <TerminalIcon className="w-5 h-5" />
          <span className="font-medium">Terminal</span>
          <ChevronUp className="w-4 h-4" />
        </motion.button>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="h-full flex flex-col bg-gray-900/95 backdrop-blur-lg border-t border-white/10"
    >
      {/* Terminal Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-black/30 border-b border-white/10">
        <div className="flex items-center space-x-2">
          <TerminalIcon className="w-4 h-4 text-green-400" />
          <span className="text-sm font-semibold text-gray-300">Terminal</span>
          <span className="text-xs text-gray-500">
            ({commandHistory.length} {commandHistory.length === 1 ? 'execution' : 'executions'})
          </span>
        </div>
        <div className="flex items-center space-x-2">
          <motion.button
            onClick={handleCopy}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            className="p-1.5 hover:bg-white/10 rounded text-gray-400 hover:text-white transition-colors"
            title="Copy output"
          >
            <Copy className="w-4 h-4" />
          </motion.button>
          <motion.button
            onClick={handleClearHistory}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            className="p-1.5 hover:bg-white/10 rounded text-gray-400 hover:text-white transition-colors"
            title="Clear terminal"
          >
            <Trash2 className="w-4 h-4" />
          </motion.button>
          <motion.button
            onClick={onToggleMinimize}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            className="p-1.5 hover:bg-white/10 rounded text-gray-400 hover:text-white transition-colors"
            title="Minimize"
          >
            <X className="w-4 h-4" />
          </motion.button>
        </div>
      </div>

      {/* Terminal Output */}
      <div
        ref={terminalRef}
        className="flex-1 overflow-y-auto p-4 font-mono text-sm text-gray-300 bg-black/20"
        style={{ scrollbarWidth: 'thin', scrollbarColor: '#4B5563 #1F2937' }}
      >
        {commandHistory.length === 0 ? (
          <div className="text-gray-500 italic">
            <p>➜ VizFlow Terminal Ready</p>
            <p className="mt-2">Run your code to see output here...</p>
          </div>
        ) : (
          <div className="space-y-4">
            {commandHistory.map((entry, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                className="border-l-2 border-blue-500/50 pl-3"
              >
                <div className="text-xs text-gray-500 mb-1">
                  {entry.timestamp.toLocaleTimeString()}
                </div>
                <pre className="whitespace-pre-wrap text-green-400">
                  {entry.output}
                </pre>
              </motion.div>
            ))}
          </div>
        )}
      </div>

      {/* Terminal Footer */}
      <div className="px-4 py-2 bg-black/30 border-t border-white/10 text-xs text-gray-500">
        <span>Status: Ready</span>
        <span className="mx-2">|</span>
        <span>Python 3.x</span>
      </div>
    </motion.div>
  );
};

export default Terminal;
