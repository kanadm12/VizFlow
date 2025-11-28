import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Grid, Info, ZoomIn, ZoomOut } from 'lucide-react';
import toast from 'react-hot-toast';

const ModelVisualization = ({ modelGraph }) => {
  const [selectedLayer, setSelectedLayer] = useState(null);
  const [zoom, setZoom] = useState(1);

  const handleLayerClick = (layer) => {
    const isSelected = selectedLayer?.id === layer.id;
    setSelectedLayer(isSelected ? null : layer);
    if (!isSelected) {
      toast.success(`📊 ${layer.name} selected`, {
        style: {
          background: '#1a1a2e',
          color: '#fff',
          border: '1px solid #3b82f6',
        },
        duration: 1500,
      });
    }
  };

  const handleZoomIn = () => {
    const newZoom = Math.min(2, zoom + 0.1);
    setZoom(newZoom);
    toast.success(`🔍 ${Math.round(newZoom * 100)}%`, {
      style: {
        background: '#1a1a2e',
        color: '#fff',
        border: '1px solid #3b82f6',
      },
      duration: 800,
    });
  };

  const handleZoomOut = () => {
    const newZoom = Math.max(0.5, zoom - 0.1);
    setZoom(newZoom);
    toast.success(`🔍 ${Math.round(newZoom * 100)}%`, {
      style: {
        background: '#1a1a2e',
        color: '#fff',
        border: '1px solid #3b82f6',
      },
      duration: 800,
    });
  };

  if (!modelGraph || modelGraph.layers.length === 0) {
    return (
      <motion.div
        className="flex flex-col items-center justify-center h-full text-gray-400"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="text-center">
          <motion.div
            animate={{ y: [-5, 5, -5] }}
            transition={{ duration: 3, repeat: Infinity }}
          >
            <Grid className="w-16 h-16 mx-auto mb-4 opacity-50" />
          </motion.div>
          <p className="text-lg font-semibold">No Model Loaded</p>
          <p className="text-sm mt-2 text-gray-500">Run code to visualize your model architecture</p>
        </div>
      </motion.div>
    );
  }

  const { layers, connections } = modelGraph;
  const layerHeight = 80;
  const spacing = 120;
  const svgHeight = Math.max(400, layers.length * spacing + 200);

  const renderGraph = () => {
    return (
      <svg
        className="w-full"
        style={{
          height: svgHeight,
          minHeight: '400px',
          transform: `scale(${zoom})`,
          transformOrigin: '0 0',
          transition: 'transform 200ms ease'
        }}
      >
        <defs>
          <linearGradient id="layerGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.9" />
            <stop offset="100%" stopColor="#1e40af" stopOpacity="0.95" />
          </linearGradient>
          <linearGradient id="selectedGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#10b981" stopOpacity="0.9" />
            <stop offset="100%" stopColor="#047857" stopOpacity="0.95" />
          </linearGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <filter id="shadow">
            <feDropShadow dx="2" dy="2" stdDeviation="3" floodOpacity="0.3" />
          </filter>
        </defs>

        {/* Draw connections */}
        {connections.map((conn, idx) => {
          const y1 = 80 + conn.from * spacing + layerHeight / 2;
          const y2 = 80 + conn.to * spacing + layerHeight / 2;

          return (
            <g key={`conn-${idx}`}>
              <line
                x1={150}
                y1={y1}
                x2={150}
                y2={y2}
                stroke="#60a5fa"
                strokeWidth="2"
                strokeDasharray="5,5"
                opacity="0.7"
              />
              <circle cx={150} cy={y2} r="4" fill="#3b82f6" filter="url(#glow)" />
            </g>
          );
        })}

        {/* Draw layers */}
        {layers.map((layer, idx) => {
          const y = 80 + idx * spacing;
          const x = 50;
          const isSelected = selectedLayer?.id === layer.id;

          return (
            <g
              key={layer.id}
              className="cursor-pointer hover:opacity-100 transition-opacity"
              opacity={selectedLayer && !isSelected ? 0.5 : 1}
              onClick={() => handleLayerClick(layer)}
            >
              {/* Layer Box */}
              <rect
                x={x}
                y={y}
                width="200"
                height={layerHeight}
                rx="8"
                fill={isSelected ? 'url(#selectedGradient)' : 'url(#layerGradient)'}
                stroke={isSelected ? '#10b981' : '#3b82f6'}
                strokeWidth={isSelected ? '3' : '2'}
                filter="url(#shadow)"
              />

              {/* Layer Name */}
              <text
                x={x + 100}
                y={y + 25}
                textAnchor="middle"
                className="fill-white font-bold"
                fontSize="14"
              >
                {layer.name}
              </text>

              {/* Layer Type */}
              <text
                x={x + 100}
                y={y + 45}
                textAnchor="middle"
                className="fill-blue-200"
                fontSize="11"
                opacity="0.9"
              >
                {layer.type}
              </text>

              {/* Parameters Box */}
              <rect
                x={x + 10}
                y={y + 55}
                width="180"
                height="18"
                rx="4"
                fill={isSelected ? 'rgba(16, 185, 129, 0.2)' : 'rgba(30, 58, 138, 0.4)'}
                stroke={isSelected ? '#10b981' : '#1e40af'}
                strokeWidth="1"
                opacity="0.7"
              />

              {/* Parameters Text */}
              <text
                x={x + 100}
                y={y + 67}
                textAnchor="middle"
                className="fill-white"
                fontSize="10"
                fontWeight="500"
              >
                {layer.inputSize} → {layer.outputSize} | {layer.trainableParams.toLocaleString()} params
              </text>
            </g>
          );
        })}
      </svg>
    );
  };

  return (
    <div className="flex flex-col h-full bg-gray-850">
      {/* Visualization Header */}
      <div className="bg-gray-800 px-4 py-3 text-sm border-b border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Grid className="w-4 h-4 text-blue-400" />
            <span className="text-gray-300 font-medium">Model Architecture</span>
            <span className="text-xs text-gray-500">
              {modelGraph.layers.length} layers • {modelGraph.connections.length} connections
            </span>
          </div>

          {/* Zoom Controls */}
          <motion.div
            className="flex items-center space-x-1 bg-gray-900 rounded-lg p-1"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4, delay: 0.2 }}
          >
            <motion.button
              onClick={handleZoomOut}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
              className="p-1 hover:bg-gray-700 rounded transition-colors text-gray-400 hover:text-white"
              title="Zoom Out"
            >
              <ZoomOut className="w-4 h-4" />
            </motion.button>
            <span className="text-xs text-gray-500 w-10 text-center">{Math.round(zoom * 100)}%</span>
            <motion.button
              onClick={handleZoomIn}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
              className="p-1 hover:bg-gray-700 rounded transition-colors text-gray-400 hover:text-white"
              title="Zoom In"
            >
              <ZoomIn className="w-4 h-4" />
            </motion.button>
          </motion.div>
        </div>
      </div>

      {/* Visualization Area */}
      <div className="flex-1 overflow-auto p-6 w-full">
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-700">
          {renderGraph()}
        </div>
      </div>

      {/* Layer Inspector */}
      {selectedLayer && (
        <motion.div
          className="bg-gray-800 border-t border-gray-700 p-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 20 }}
          transition={{ duration: 0.3 }}
        >
          <div className="flex items-start space-x-3">
            <Info className="w-4 h-4 text-blue-400 mt-0.5 flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <motion.div
                className="text-sm font-semibold text-white mb-2"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.1 }}
              >
                {selectedLayer.name}
              </motion.div>
              <motion.div
                className="grid grid-cols-2 gap-2 text-xs text-gray-400"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.15 }}
              >
                <div>
                  <span className="text-gray-500">Type:</span> {selectedLayer.type}
                </div>
                <div>
                  <span className="text-gray-500">Input:</span> {selectedLayer.inputSize}
                </div>
                <div>
                  <span className="text-gray-500">Output:</span> {selectedLayer.outputSize}
                </div>
                <div>
                  <span className="text-gray-500">Parameters:</span> {selectedLayer.trainableParams.toLocaleString()}
                </div>
              </motion.div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default ModelVisualization;
