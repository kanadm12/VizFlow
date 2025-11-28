import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Maximize2 } from 'lucide-react';

const SplitPane = ({ left, right, defaultSplit = 50, minSize = 20, maxSize = 80 }) => {
  const [splitPos, setSplitPos] = useState(defaultSplit);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef(null);

  const handleMouseDown = () => setIsDragging(true);

  const handleMouseMove = (e) => {
    if (!isDragging || !containerRef.current) return;

    const container = containerRef.current.getBoundingClientRect();
    const newPos = ((e.clientX - container.left) / container.width) * 100;
    setSplitPos(Math.max(minSize, Math.min(maxSize, newPos)));
  };

  const handleMouseUp = () => setIsDragging(false);

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging]);

  return (
    <div
      ref={containerRef}
      className="flex-1 flex overflow-hidden w-full"
      style={{ cursor: isDragging ? 'col-resize' : 'default' }}
    >
      {/* Left Pane */}
      <motion.div
        style={{ width: `${splitPos}%`, minWidth: 0 }}
        className="flex flex-col"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        {left}
      </motion.div>

      {/* Draggable Divider */}
      <motion.div
        className="divider w-0.5 bg-gray-700 hover:bg-blue-500 cursor-col-resize transition-all duration-200 relative group flex-shrink-0"
        onMouseDown={handleMouseDown}
        whileHover={{ scaleY: 1.1 }}
      >
        <div className="absolute inset-y-0 -left-1.5 -right-1.5" />
        <motion.div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity duration-200"
          animate={isDragging ? { scale: 1.2 } : { scale: 1 }}
        >
          <motion.div
            className="bg-blue-500 rounded-full p-1"
            whileHover={{ scale: 1.15 }}
          >
            <Maximize2 className="w-3 h-3 text-white" />
          </motion.div>
        </motion.div>
      </motion.div>

      {/* Right Pane */}
      <motion.div
        style={{ width: `${100 - splitPos}%`, minWidth: 0 }}
        className="flex flex-col"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.1 }}
      >
        {right}
      </motion.div>
    </div>
  );
};

export default SplitPane;
