import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { Grid, Info, ZoomIn, ZoomOut, Maximize2, TreePine, GitBranch, Workflow, Network, Download, FileJson } from 'lucide-react';
import toast from 'react-hot-toast';
import { selectLayout, layoutStrategies } from '../utils/GraphRenderer';
import { detectBestLayout, getLayoutRecommendation } from '../utils/LayoutDetector';
import { exportSVGToPNG, exportAsSVG, generateReport } from '../utils/ExportUtils';

const AdvancedModelVisualization = ({ modelGraph }) => {
  const [selectedLayer, setSelectedLayer] = useState(null);
  const [zoom, setZoom] = useState(1);
  const [layoutType, setLayoutType] = useState(null);
  const [autoDetectedLayout, setAutoDetectedLayout] = useState(null);
  const [layoutData, setLayoutData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showLayoutInfo, setShowLayoutInfo] = useState(false);
  const [exporting, setExporting] = useState(false);
  const svgRef = useRef(null);

  // Auto-detect layout on model change
  useEffect(() => {
    if (!modelGraph?.layers?.length) return;
    
    const detected = detectBestLayout(modelGraph);
    setAutoDetectedLayout(detected);
    setLayoutType(detected); // Set initial layout to auto-detected
  }, [modelGraph]);

  // Compute layout when graph or layout type changes
  useEffect(() => {
    if (!modelGraph?.layers?.length || !layoutType) return;

    const computeLayout = async () => {
      setLoading(true);
      try {
        const nodes = modelGraph.layers.map(layer => ({
          id: layer.id,
          name: layer.name,
          type: layer.type,
          properties: layer
        }));

        const edges = (modelGraph.connections || []).map(conn => ({
          source: conn.from,
          target: conn.to,
          type: 'connection'
        }));

        const layout = await selectLayout(nodes, edges, layoutType);
        setLayoutData(layout);
      } catch (error) {
        console.error('Layout computation error:', error);
        toast.error('Layout computation failed');
      }
      setLoading(false);
    };

    computeLayout();
  }, [modelGraph, layoutType]);

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

  const handleLayoutChange = (newType) => {
    setLayoutType(newType);
    const isAutoDetected = newType === autoDetectedLayout;
    toast.success(`📐 ${newType.toUpperCase()}${isAutoDetected ? ' (auto-detected)' : ''}`, {
      style: {
        background: '#1a1a2e',
        color: '#fff',
        border: `1px solid ${isAutoDetected ? '#06b6d4' : '#10b981'}`,
      },
      duration: 1000,
    });
  };

  const handleDownloadPNG = async () => {
    if (!svgRef.current) {
      toast.error('Visualization not ready');
      return;
    }

    setExporting(true);
    try {
      await exportSVGToPNG(svgRef.current, `flowchart-${Date.now()}.png`);
      toast.success('✅ Flowchart downloaded as PNG!', {
        style: {
          background: '#1a1a2e',
          color: '#fff',
          border: '1px solid #10b981',
        },
      });
    } catch (error) {
      toast.error(`Export failed: ${error.message}`);
    } finally {
      setExporting(false);
    }
  };

  const handleDownloadSVG = async () => {
    if (!svgRef.current) {
      toast.error('Visualization not ready');
      return;
    }

    try {
      exportAsSVG(svgRef.current, `flowchart-${Date.now()}.svg`);
      toast.success('✅ Flowchart downloaded as SVG!', {
        style: {
          background: '#1a1a2e',
          color: '#fff',
          border: '1px solid #10b981',
        },
      });
    } catch (error) {
      toast.error(`Export failed: ${error.message}`);
    }
  };

  const handleGenerateReport = async () => {
    if (!svgRef.current) {
      toast.error('Visualization not ready');
      return;
    }

    setExporting(true);
    try {
      await generateReport(modelGraph, svgRef.current, 'html');
      toast.success('✅ Report generated and downloaded!', {
        style: {
          background: '#1a1a2e',
          color: '#fff',
          border: '1px solid #10b981',
        },
      });
    } catch (error) {
      toast.error(`Report generation failed: ${error.message}`);
    } finally {
      setExporting(false);
    }
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

  return (
    <motion.div
      className="flex flex-col h-full bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      {/* Toolbar */}
      <motion.div
        className="flex items-center justify-between p-4 border-b border-slate-700 bg-slate-800/50 backdrop-blur-sm"
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
      >
        <div className="flex items-center space-x-2">
          <h3 className="text-sm font-semibold text-white">Diagram Type:</h3>
          
          {/* Layout type buttons */}
          <div className="flex space-x-2">
            <motion.button
              onClick={() => handleLayoutChange('dag')}
              className={`p-2 rounded-lg transition-all ${
                layoutType === 'dag'
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
              }`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              title="Directed Acyclic Graph (Best for neural networks)"
            >
              <Workflow className="w-4 h-4" />
            </motion.button>

            <motion.button
              onClick={() => handleLayoutChange('tree')}
              className={`p-2 rounded-lg transition-all ${
                layoutType === 'tree'
                  ? 'bg-green-600 text-white shadow-lg'
                  : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
              }`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              title="Tree Structure"
            >
              <TreePine className="w-4 h-4" />
            </motion.button>

            <motion.button
              onClick={() => handleLayoutChange('flowchart')}
              className={`p-2 rounded-lg transition-all ${
                layoutType === 'flowchart'
                  ? 'bg-cyan-600 text-white shadow-lg'
                  : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
              }`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              title="Flowchart Layout"
            >
              <GitBranch className="w-4 h-4" />
            </motion.button>

            <motion.button
              onClick={() => handleLayoutChange('graph')}
              className={`p-2 rounded-lg transition-all ${
                layoutType === 'graph'
                  ? 'bg-purple-600 text-white shadow-lg'
                  : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
              }`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              title="Force-directed Graph"
            >
              <Network className="w-4 h-4" />
            </motion.button>
          </div>
        </div>

        {/* Zoom controls */}
        <div className="flex items-center space-x-2">
          <motion.button
            onClick={handleZoomOut}
            className="p-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-gray-300 transition-all"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            title="Zoom out"
          >
            <ZoomOut className="w-4 h-4" />
          </motion.button>

          <span className="text-sm text-gray-400 w-12 text-center">
            {Math.round(zoom * 100)}%
          </span>

          <motion.button
            onClick={handleZoomIn}
            className="p-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-gray-300 transition-all"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            title="Zoom in"
          >
            <ZoomIn className="w-4 h-4" />
          </motion.button>

          <div className="w-px h-6 bg-slate-600 mx-1" />

          {/* Download and Export */}
          <motion.button
            onClick={handleDownloadPNG}
            disabled={exporting}
            className="p-2 rounded-lg bg-green-600 hover:bg-green-700 text-white transition-all disabled:opacity-50"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            title="Download as PNG"
          >
            <Download className="w-4 h-4" />
          </motion.button>

          <motion.button
            onClick={handleDownloadSVG}
            disabled={exporting}
            className="p-2 rounded-lg bg-purple-600 hover:bg-purple-700 text-white transition-all disabled:opacity-50"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            title="Download as SVG"
          >
            <FileJson className="w-4 h-4" />
          </motion.button>

          <motion.button
            onClick={handleGenerateReport}
            disabled={exporting}
            className="p-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white transition-all disabled:opacity-50"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            title="Generate Report (HTML)"
          >
            <Info className="w-4 h-4" />
          </motion.button>

          <div className="w-px h-6 bg-slate-600 mx-1" />

          <span className="text-xs text-gray-400">
            {modelGraph.layers.length} layers • {modelGraph.connections?.length || 0} connections
          </span>
        </div>
      </motion.div>

      {/* Visualization Area */}
      <motion.div
        className="flex-1 overflow-auto relative bg-slate-900"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
      >
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/30 backdrop-blur-sm z-50">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <Grid className="w-8 h-8 text-blue-500" />
            </motion.div>
          </div>
        )}

        {layoutData ? (
          <VisualizationCanvas
            layoutData={layoutData}
            zoom={zoom}
            selectedLayer={selectedLayer}
            onLayerClick={handleLayerClick}
          />
        ) : (
          <div className="flex items-center justify-center h-full text-gray-500">
            Computing layout...
          </div>
        )}
      </motion.div>

      {/* Info Panel */}
      {selectedLayer && (
        <motion.div
          className="p-4 border-t border-slate-700 bg-slate-800/50 backdrop-blur-sm max-h-40 overflow-y-auto"
          initial={{ y: 100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: 100, opacity: 0 }}
        >
          <div className="space-y-2">
            <h4 className="font-semibold text-white flex items-center space-x-2">
              <Info className="w-4 h-4 text-blue-500" />
              <span>{selectedLayer.name}</span>
            </h4>
            <div className="grid grid-cols-2 gap-2 text-xs text-gray-300">
              <div><span className="text-gray-500">Type:</span> {selectedLayer.type}</div>
              {selectedLayer.properties?.params && (
                <div><span className="text-gray-500">Parameters:</span> {selectedLayer.properties.params}</div>
              )}
              {selectedLayer.properties?.output && (
                <div><span className="text-gray-500">Output:</span> {selectedLayer.properties.output}</div>
              )}
            </div>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};

/**
 * SVG Canvas for rendering the graph
 */
const VisualizationCanvas = ({ layoutData, zoom, selectedLayer, onLayerClick }) => {
  const { nodes, links, type } = layoutData;

  if (!nodes || nodes.length === 0) {
    return <div className="text-gray-500 p-4">No layout data available</div>;
  }

  // Calculate bounds
  const padding = 50;
  const minX = Math.min(...nodes.map(n => n.x)) - padding;
  const maxX = Math.max(...nodes.map(n => n.x)) + padding;
  const minY = Math.min(...nodes.map(n => n.y)) - padding;
  const maxY = Math.max(...nodes.map(n => n.y)) + padding;

  const viewBox = `${minX} ${minY} ${maxX - minX} ${maxY - minY}`;

  return (
    <svg
      viewBox={viewBox}
      className="w-full h-full"
      style={{
        backgroundColor: '#0f172a',
      }}
    >
      <defs>
        <linearGradient id="nodeGradient" x1="0%" y1="0%" x2="0%" y2="100%">
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
        <marker
          id="arrowhead"
          markerWidth="10"
          markerHeight="10"
          refX="9"
          refY="3"
          orient="auto"
        >
          <polygon points="0 0, 10 3, 0 6" fill="#60a5fa" fillOpacity="0.6" />
        </marker>
      </defs>

      {/* Draw connections/edges with curves */}
      {links.map((link, idx) => {
        const sourceNode = nodes.find(n => n.id === link.source);
        const targetNode = nodes.find(n => n.id === link.target);

        if (!sourceNode || !targetNode) return null;

        // Calculate curved path (quadratic Bezier curve)
        const x1 = sourceNode.x;
        const y1 = sourceNode.y;
        const x2 = targetNode.x;
        const y2 = targetNode.y;
        
        // Control point for curve (middle point offset)
        const dx = x2 - x1;
        const dy = y2 - y1;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const curveAmount = Math.min(distance * 0.3, 80); // Curve intensity
        
        // Perpendicular offset for curve
        const perpX = -dy / distance * curveAmount;
        const perpY = dx / distance * curveAmount;
        
        const cx = (x1 + x2) / 2 + perpX;
        const cy = (y1 + y2) / 2 + perpY;
        
        // Quadratic Bezier path
        const pathData = `M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`;

        return (
          <motion.g key={`edge-${idx}`}>
            {/* Connection line with glow effect */}
            <motion.path
              d={pathData}
              fill="none"
              stroke="#60a5fa"
              strokeWidth="3"
              strokeOpacity="0.2"
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 0.2 }}
              transition={{ duration: 1, delay: idx * 0.05 }}
              style={{ filter: 'drop-shadow(0 0 4px #3b82f6)' }}
            />
            
            {/* Main connection line */}
            <motion.path
              d={pathData}
              fill="none"
              stroke="#60a5fa"
              strokeWidth="2"
              strokeOpacity="0.7"
              markerEnd="url(#arrowhead)"
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 0.7 }}
              transition={{ duration: 0.8, delay: idx * 0.05 }}
              className="hover:stroke-cyan-400"
              style={{ transition: 'stroke 200ms' }}
            />
            
            {/* Animated flow pulse */}
            <motion.circle
              r="3"
              fill="#06b6d4"
              opacity="0.8"
              initial={{ offsetDistance: '0%' }}
              animate={{ offsetDistance: '100%' }}
              transition={{
                duration: 2,
                delay: idx * 0.1,
                repeat: Infinity,
                ease: 'linear'
              }}
              style={{
                offsetPath: `path('${pathData}')`,
                offsetRotate: '0deg'
              }}
            />
          </motion.g>
        );
      })}

      {/* Draw nodes */}
      {nodes.map((node, idx) => {
        const isSelected = selectedLayer?.id === node.id;
        const nodeWidth = node.width || 140;
        const nodeHeight = node.height || 80;

        return (
          <motion.g
            key={node.id}
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.5, delay: idx * 0.02 }}
            onClick={() => onLayerClick(node)}
            style={{ cursor: 'pointer' }}
          >
            {/* Node background */}
            <motion.rect
              x={node.x - nodeWidth / 2}
              y={node.y - nodeHeight / 2}
              width={nodeWidth}
              height={nodeHeight}
              rx="8"
              ry="8"
              fill={isSelected ? 'url(#selectedGradient)' : 'url(#nodeGradient)'}
              stroke={isSelected ? '#10b981' : '#3b82f6'}
              strokeWidth={isSelected ? '3' : '2'}
              filter={isSelected ? 'url(#glow)' : undefined}
              whileHover={{
                scale: 1.08,
                boxShadow: '0 0 20px rgba(59, 130, 246, 0.6)'
              }}
            />

            {/* Node text - simplified */}
            <text
              x={node.x}
              y={node.y - 5}
              textAnchor="middle"
              fontSize="12"
              fill="white"
              fontWeight="600"
              pointerEvents="none"
            >
              {node.name}
            </text>

            {/* Node type badge */}
            {node.type && (
              <text
                x={node.x}
                y={node.y + 12}
                textAnchor="middle"
                fontSize="10"
                fill="#a0aec0"
                pointerEvents="none"
              >
                {node.type}
              </text>
            )}

            {/* Selection indicator */}
            {isSelected && (
              <motion.circle
                cx={node.x}
                cy={node.y}
                r={nodeWidth / 2 + 15}
                fill="none"
                stroke="#10b981"
                strokeWidth="2"
                strokeDasharray="5,5"
                animate={{ rotate: 360 }}
                transition={{ duration: 4, repeat: Infinity, ease: 'linear' }}
              />
            )}
          </motion.g>
        );
      })}
    </svg>
  );
};

export default AdvancedModelVisualization;
