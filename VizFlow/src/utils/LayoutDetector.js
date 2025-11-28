/**
 * Automatically detects the best layout type based on model architecture
 */

export const detectBestLayout = (modelGraph) => {
  if (!modelGraph || !modelGraph.layers || modelGraph.layers.length === 0) {
    return 'dag';
  }

  const layers = modelGraph.layers;
  const connections = modelGraph.connections || [];
  
  // Count different layer types
  const layerTypes = {};
  layers.forEach(layer => {
    layerTypes[layer.type] = (layerTypes[layer.type] || 0) + 1;
  });

  // Calculate graph characteristics
  const totalLayers = layers.length;
  const totalConnections = connections.length;
  const avgConnections = totalLayers > 0 ? totalConnections / totalLayers : 0;
  
  // Check for skip connections (residual networks)
  const hasSkipConnections = connections.some(conn => {
    const sourceIndex = layers.findIndex(l => l.id === conn.from);
    const targetIndex = layers.findIndex(l => l.id === conn.to);
    return sourceIndex >= 0 && targetIndex >= 0 && targetIndex - sourceIndex > 1;
  });

  // Check for recurrent connections (RNNs, LSTMs)
  const hasRecurrentConnections = connections.some(conn => {
    const sourceIndex = layers.findIndex(l => l.id === conn.from);
    const targetIndex = layers.findIndex(l => l.id === conn.to);
    return sourceIndex >= 0 && targetIndex >= 0 && targetIndex < sourceIndex;
  });

  // Check if it's a sequential/linear architecture
  const isSequential = totalConnections === totalLayers - 1 && 
                       avgConnections <= 1.1 && 
                       !hasSkipConnections && 
                       !hasRecurrentConnections;

  // Check if it's a tree-like structure
  const isTree = connections.every(conn => {
    const targetIndegree = connections.filter(c => c.to === conn.to).length;
    return targetIndegree === 1;
  });

  // Check for complex/graph-like structures
  const isComplex = hasSkipConnections || hasRecurrentConnections || avgConnections > 1.5;

  // Detect model type from layer types
  const hasConvLayers = layerTypes['Conv2d'] || layerTypes['Conv1d'] || layerTypes['Conv3d'];
  const hasLSTMLayers = layerTypes['LSTM'] || layerTypes['GRU'] || layerTypes['RNN'];
  const hasAttentionLayers = layerTypes['MultiheadAttention'] || layerTypes['Attention'];
  const hasPoolingLayers = layerTypes['MaxPool2d'] || layerTypes['AvgPool2d'];
  
  // Decision logic
  if (isSequential && !hasSkipConnections) {
    return 'flowchart'; // Simple sequential models
  } else if (isTree && !hasRecurrentConnections) {
    return 'tree'; // Tree-like hierarchical structures
  } else if (isComplex || hasSkipConnections || hasRecurrentConnections) {
    return 'graph'; // Complex models with skip/recurrent connections
  } else if (hasConvLayers && hasPoolingLayers && totalLayers < 20) {
    return 'dag'; // CNNs and small networks
  } else {
    return 'dag'; // Default to DAG for most cases
  }
};

/**
 * Get layout recommendation with explanation
 */
export const getLayoutRecommendation = (modelGraph) => {
  const layout = detectBestLayout(modelGraph);
  
  const recommendations = {
    dag: {
      name: 'DAG (Directed Acyclic Graph)',
      icon: 'Workflow',
      description: 'Best for hierarchical neural networks and feedforward architectures',
      advantages: ['Clear flow', 'Hierarchical', 'Good for most models']
    },
    tree: {
      name: 'Tree Structure',
      icon: 'TreePine',
      description: 'Best for tree-like hierarchical models',
      advantages: ['Level-based', 'Clear hierarchy', 'Minimal overlap']
    },
    flowchart: {
      name: 'Flowchart',
      icon: 'GitBranch',
      description: 'Best for sequential and linear models',
      advantages: ['Left-to-right flow', 'Simple', 'Easy to follow']
    },
    graph: {
      name: 'Force-Directed Graph',
      icon: 'Network',
      description: 'Best for complex models with skip connections or recurrent patterns',
      advantages: ['Shows all relationships', 'Handles complexity', 'Interactive']
    }
  };

  return {
    recommended: layout,
    recommendation: recommendations[layout],
    alternatives: Object.keys(recommendations).filter(k => k !== layout)
  };
};
