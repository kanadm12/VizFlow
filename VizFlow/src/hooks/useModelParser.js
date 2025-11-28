import { useState, useCallback } from 'react';

/**
 * Enhanced model parser that generates proper graph structures
 * Supports: PyTorch, TensorFlow, Keras models
 */
const useModelParser = () => {
  const [modelGraph, setModelGraph] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const executeCode = useCallback((code) => {
    setIsLoading(true);

    try {
      // Parse PyTorch model
      const torchMatch = code.match(/class\s+(\w+)\s*\(\s*nn\.Module\s*\)/);
      if (torchMatch) {
        const modelName = torchMatch[1];
        const graph = parseNeuralNetwork(code, modelName);
        setModelGraph(graph);
        setIsLoading(false);
        return;
      }

      // Parse TensorFlow/Keras model
      const tfMatch = code.match(/Sequential\(\[|Model\(inputs/);
      if (tfMatch) {
        const graph = parseTensorFlowModel(code);
        setModelGraph(graph);
        setIsLoading(false);
        return;
      }

      // Parse simple Python objects as flowchart
      const graph = parseAsFlowchart(code);
      if (graph.layers.length > 0) {
        setModelGraph(graph);
      } else {
        setModelGraph({
          layers: [],
          connections: []
        });
      }
    } catch (error) {
      console.error('Parse error:', error);
    }
    setIsLoading(false);
  }, []);

  return { modelGraph, executeCode, isLoading };
};

/**
 * Parse PyTorch model architecture
 */
function parseNeuralNetwork(code, modelName) {
  const layers = [];
  const connections = [];
  let layerId = 0;

  // Extract all layers
  const layerPattern = /self\.(\w+)\s*=\s*nn\.(\w+)\s*\((.*?)\)/g;
  let match;
  const layerMap = new Map();

  while ((match = layerPattern.exec(code)) !== null) {
    const [fullMatch, varName, layerType, params] = match;
    
    const layer = {
      id: `layer_${layerId}`,
      name: varName,
      type: layerType,
      params: parseLayerParams(layerType, params),
      trainableParams: estimateParams(layerType, params),
      output: `${layerType}(${params.split(',')[0]})`
    };

    layers.push(layer);
    layerMap.set(varName, `layer_${layerId}`);
    layerId++;
  }

  // Extract forward pass connections
  const forwardPattern = /def forward\(self,\s*(\w+)\)([\s\S]*?)(?=def|\Z)/;
  const forwardMatch = code.match(forwardPattern);

  if (forwardMatch) {
    const forwardBody = forwardMatch[2];
    
    // Parse assignments like x = self.fc1(x)
    const assignPattern = /(\w+)\s*=\s*self\.(\w+)\((.*?)\)/g;
    let prevVar = 'x'; // Input variable
    let prevLayerId = null;

    while ((match = assignPattern.exec(forwardBody)) !== null) {
      const [, outputVar, layerName, inputVar] = match;
      
      if (layerMap.has(layerName)) {
        const currentLayerId = layerMap.get(layerName);
        
        if (prevLayerId) {
          connections.push({
            from: prevLayerId,
            to: currentLayerId,
            label: ''
          });
        }
        
        prevLayerId = currentLayerId;
        prevVar = outputVar;
      }
    }

    // If no connections found, create linear chain
    if (connections.length === 0 && layers.length > 1) {
      for (let i = 0; i < layers.length - 1; i++) {
        connections.push({
          from: layers[i].id,
          to: layers[i + 1].id,
          label: ''
        });
      }
    }
  }

  return { layers, connections };
}

/**
 * Parse TensorFlow/Keras model
 */
function parseTensorFlowModel(code) {
  const layers = [];
  const connections = [];
  let layerId = 0;

  // Look for Dense, Conv, etc. layer definitions
  const layerPattern = /layers\.(\w+)\s*\((.*?)\)|\.add\((\w+)\((.*?)\)\)/g;
  let match;

  while ((match = layerPattern.exec(code)) !== null) {
    const layerType = match[1] || match[3] || 'Layer';
    const params = match[2] || match[4] || '';

    const layer = {
      id: `layer_${layerId}`,
      name: `${layerType}_${layerId}`,
      type: layerType,
      params: params.split(',')[0],
      trainableParams: estimateParams(layerType, params),
      output: `${layerType}(...)`
    };

    layers.push(layer);
    layerId++;
  }

  // Create linear chain
  for (let i = 0; i < layers.length - 1; i++) {
    connections.push({
      from: layers[i].id,
      to: layers[i + 1].id,
      label: ''
    });
  }

  return { layers, connections };
}

/**
 * Parse any Python code as a flowchart
 */
function parseAsFlowchart(code) {
  const layers = [];
  const connections = [];
  let nodeId = 0;

  // Look for class definitions
  const classPattern = /class\s+(\w+)/g;
  const funcPattern = /def\s+(\w+)\s*\((.*?)\)/g;
  const processPattern = /^[\s]*([\w.]+)\s*=\s*([\w.()[\]]+)/gm;

  let match;
  const seenNodes = new Set();

  // Add class definitions as nodes
  while ((match = classPattern.exec(code)) !== null) {
    const className = match[1];
    if (!seenNodes.has(className)) {
      layers.push({
        id: `node_${nodeId}`,
        name: className,
        type: 'Class',
        params: 'Definition',
        trainableParams: 0,
        output: ''
      });
      seenNodes.add(className);
      nodeId++;
    }
  }

  // Add function definitions as nodes
  while ((match = funcPattern.exec(code)) !== null) {
    const funcName = match[1];
    const params = match[2];
    if (!seenNodes.has(funcName)) {
      layers.push({
        id: `node_${nodeId}`,
        name: funcName,
        type: 'Function',
        params: params || 'no params',
        trainableParams: 0,
        output: ''
      });
      seenNodes.add(funcName);
      nodeId++;
    }
  }

  // Add assignments as nodes
  while ((match = processPattern.exec(code)) !== null) {
    const [, varName, value] = match;
    if (!seenNodes.has(varName) && varName.length < 50) {
      layers.push({
        id: `node_${nodeId}`,
        name: varName,
        type: 'Variable',
        params: value,
        trainableParams: 0,
        output: value.substring(0, 30)
      });
      seenNodes.add(varName);
      nodeId++;
    }
  }

  // Create sequential connections
  for (let i = 0; i < layers.length - 1; i++) {
    connections.push({
      from: layers[i].id,
      to: layers[i + 1].id,
      label: ''
    });
  }

  return { layers, connections };
}

/**
 * Parse layer parameters from string
 */
function parseLayerParams(layerType, paramStr) {
  const parts = paramStr.split(',')[0].trim();
  return parts || layerType;
}

/**
 * Estimate number of trainable parameters
 */
function estimateParams(layerType, paramStr) {
  try {
    const nums = paramStr.match(/\d+/g) || [];
    if (nums.length === 0) return 0;

    const num1 = parseInt(nums[0]);
    const num2 = parseInt(nums[1] || nums[0]);

    const paramMap = {
      Linear: num1 * num2 + num2,
      Dense: num1 * num2 + num2,
      Conv1d: num1 * 3 * num2 + num2,
      Conv2d: num1 * 3 * 3 * num2 + num2,
      Conv3d: num1 * 3 * 3 * 3 * num2 + num2,
      LSTM: 4 * num1 * num2 + 4 * num2 * num2 + 3 * num2,
      GRU: 3 * num1 * num2 + 3 * num2 * num2 + 2 * num2,
      BatchNorm1d: 2 * num1,
      BatchNorm2d: 2 * num1,
      Embedding: num1 * num2,
    };

    return paramMap[layerType] || num1 * num2;
  } catch {
    return 0;
  }
}

export default useModelParser;
