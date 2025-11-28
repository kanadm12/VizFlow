import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Terminal as TerminalIcon, X, Trash2, Copy, ChevronUp, ChevronDown, Maximize2, Minimize2 } from 'lucide-react';
import toast from 'react-hot-toast';

const Terminal = ({ output, onClear, isMinimized, onToggleMinimize }) => {
  const terminalRef = useRef(null);
  const inputRef = useRef(null);
  const [commandHistory, setCommandHistory] = useState([]);
  const [currentCommand, setCurrentCommand] = useState('');
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [executedCommands, setExecutedCommands] = useState([]);

  useEffect(() => {
    if (terminalRef.current && !isMinimized) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [output, isMinimized, commandHistory]);

  useEffect(() => {
    if (output) {
      setCommandHistory(prev => [...prev, { 
        type: 'output',
        timestamp: new Date(), 
        content: output 
      }]);
    }
  }, [output]);

  const handleCopy = () => {
    const allOutput = commandHistory.map(entry => entry.content).join('\n');
    navigator.clipboard.writeText(allOutput);
    toast.success('Output copied to clipboard!');
  };

  const handleClearHistory = () => {
    setCommandHistory([]);
    setExecutedCommands([]);
    onClear();
    toast.success('Terminal cleared!');
  };

  const executeCommand = (cmd) => {
    const trimmedCmd = cmd.trim();
    if (!trimmedCmd) return;

    // Add command to history
    setCommandHistory(prev => [...prev, {
      type: 'command',
      timestamp: new Date(),
      content: trimmedCmd
    }]);

    setExecutedCommands(prev => [...prev, trimmedCmd]);

    // Simulate command execution
    let response = '';
    const lowerCmd = trimmedCmd.toLowerCase();

    if (lowerCmd === 'clear' || lowerCmd === 'cls') {
      handleClearHistory();
      return;
    } else if (lowerCmd === 'help') {
      response = `Available commands:
  clear / cls    - Clear terminal
  help           - Show this help message
  pwd            - Print working directory
  ls / dir       - List files
  python --version - Show Python version
  node --version - Show Node version
  npm --version  - Show npm version
  
Note: This is a simulated terminal for demonstration.
Use the Run button to execute your code.`;
    } else if (lowerCmd === 'pwd') {
      response = 'C:\\Users\\Kanad\\Desktop\\VizFlow\\VizFlow';
    } else if (lowerCmd === 'ls' || lowerCmd === 'dir') {
      response = `src/
server/
package.json
README.md
vite.config.js`;
    } else if (lowerCmd.includes('python')) {
      if (lowerCmd.includes('--version') || lowerCmd.includes('-v')) {
        response = 'Python 3.11.0';
      } else {
        response = 'Use the Run button to execute Python code in the editor.';
      }
    } else if (lowerCmd.includes('node')) {
      if (lowerCmd.includes('--version') || lowerCmd.includes('-v')) {
        response = 'v20.10.0';
      } else {
        response = 'Node.js runtime available.';
      }
    } else if (lowerCmd.includes('npm')) {
      if (lowerCmd.includes('--version') || lowerCmd.includes('-v')) {
        response = '10.2.3';
      } else {
        response = 'npm package manager available.';
      }
    } else {
      response = `Command not recognized: ${trimmedCmd}\nType 'help' for available commands.`;
    }

    // Add response to history
    if (response) {
      setCommandHistory(prev => [...prev, {
        type: 'output',
        timestamp: new Date(),
        content: response
      }]);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      executeCommand(currentCommand);
      setCurrentCommand('');
      setHistoryIndex(-1);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (executedCommands.length > 0) {
        const newIndex = historyIndex + 1;
        if (newIndex < executedCommands.length) {
          setHistoryIndex(newIndex);
          setCurrentCommand(executedCommands[executedCommands.length - 1 - newIndex]);
        }
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIndex > 0) {
        const newIndex = historyIndex - 1;
        setHistoryIndex(newIndex);
        setCurrentCommand(executedCommands[executedCommands.length - 1 - newIndex]);
      } else if (historyIndex === 0) {
        setHistoryIndex(-1);
        setCurrentCommand('');
      }
    }
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
        onClick={() => inputRef.current?.focus()}
      >
        {commandHistory.length === 0 ? (
          <div className="text-gray-500">
            <p className="text-green-400">➜ VizFlow Terminal v1.0</p>
            <p className="mt-2">Type 'help' for available commands or use the Run button to execute code.</p>
          </div>
        ) : (
          <div className="space-y-2">
            {commandHistory.map((entry, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.2 }}
              >
                {entry.type === 'command' ? (
                  <div className="flex items-start space-x-2">
                    <span className="text-green-400">➜</span>
                    <span className="text-white">{entry.content}</span>
                  </div>
                ) : (
                  <pre className="whitespace-pre-wrap text-gray-300 ml-4">
{entry.content}
                  </pre>
                )}
              </motion.div>
            ))}
          </div>
        )}
      </div>

      {/* Terminal Input */}
      <div className="px-4 py-2 bg-black/30 border-t border-white/10 flex items-center space-x-2">
        <span className="text-green-400 font-mono text-sm">➜</span>
        <input
          ref={inputRef}
          type="text"
          value={currentCommand}
          onChange={(e) => setCurrentCommand(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a command..."
          className="flex-1 bg-transparent text-white font-mono text-sm outline-none placeholder-gray-600"
          autoFocus
        />
      </div>

      {/* Terminal Footer */}
      <div className="px-4 py-1.5 bg-black/40 border-t border-white/5 text-xs text-gray-500 flex justify-between">
        <span>Status: Ready</span>
        <span>{commandHistory.length} entries</span>
      </div>
    </motion.div>
  );
};

export default Terminal;
