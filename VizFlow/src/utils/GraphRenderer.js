/**
 * Advanced Graph Rendering Engine
 * Supports: Flowcharts, Trees, DAGs, Neural Networks
 */

import * as d3 from 'd3';
import dagre from 'dagre';

/**
 * Layout strategies for different diagram types
 */
export const layoutStrategies = {
  flowchart: 'flowchart',
  tree: 'tree',
  dag: 'dag',
  neural_network: 'neural_network',
  graph: 'graph'
};

/**
 * Hierarchical (Tree) Layout using D3
 */
export const treeLayout = (nodes, edges) => {
  const root = buildHierarchy(nodes, edges);
  
  const width = 1200;
  const height = 800;
  
  const tree = d3.tree().size([width, height]);
  const layout = tree(root);
  
  const treeNodes = [];
  const treeLinks = [];
  
  layout.each(node => {
    treeNodes.push({
      id: node.data.id,
      name: node.data.name,
      x: node.x,
      y: node.y,
      type: node.data.type,
      properties: node.data.properties
    });
  });
  
  layout.links().forEach(link => {
    treeLinks.push({
      source: link.source.data.id,
      target: link.target.data.id,
      type: 'tree'
    });
  });
  
  return { nodes: treeNodes, links: treeLinks };
};

/**
 * Dagre-based Layout (DAG) - Best for neural networks and flowcharts
 */
export const dagreLayout = (nodes, edges, rankdir = 'TB') => {
  const g = new dagre.graphlib.Graph({ compound: true });
  
  g.setGraph({
    rankdir: rankdir, // TB, LR, RL, BT
    nodesep: 100,
    ranksep: 100,
    marginx: 20,
    marginy: 20
  });
  
  g.setDefaultEdgeLabel(() => ({}));
  
  // Add nodes
  nodes.forEach(node => {
    g.setNode(node.id, {
      label: node.name,
      width: 140,
      height: 80,
      ...node
    });
  });
  
  // Add edges
  edges.forEach(edge => {
    g.setEdge(edge.source, edge.target, { label: edge.label || '' });
  });
  
  // Run layout
  dagre.layout(g);
  
  // Extract positioned nodes and edges
  const layoutNodes = [];
  const layoutLinks = [];
  
  g.nodes().forEach(nodeId => {
    const dagNode = g.node(nodeId);
    const originalNode = nodes.find(n => n.id === nodeId);
    
    layoutNodes.push({
      id: nodeId,
      name: dagNode.label || originalNode?.name,
      x: dagNode.x,
      y: dagNode.y,
      width: dagNode.width,
      height: dagNode.height,
      type: originalNode?.type || 'default',
      properties: originalNode?.properties || {}
    });
  });
  
  g.edges().forEach(edge => {
    const dagEdge = g.edge(edge);
    layoutLinks.push({
      source: edge.v,
      target: edge.w,
      points: dagEdge.points || [],
      label: dagEdge.label || ''
    });
  });
  
  return { nodes: layoutNodes, links: layoutLinks };
};

/**
 * Force-directed layout (for general graphs)
 */
export const forceLayout = (nodes, edges, width = 1200, height = 800) => {
  return new Promise((resolve) => {
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(edges)
        .id(d => d.id)
        .distance(120)
        .strength(0.5)
      )
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(60))
      .on('end', () => {
        const layoutNodes = nodes.map(n => ({
          ...n,
          x: n.x,
          y: n.y
        }));
        
        const layoutLinks = edges.map(e => ({
          source: e.source.id || e.source,
          target: e.target.id || e.target,
          type: e.type || 'default'
        }));
        
        resolve({ nodes: layoutNodes, links: layoutLinks });
      });
  });
};

/**
 * Circular/Radial layout for graphs
 */
export const circularLayout = (nodes, edges) => {
  const radius = 300;
  const angleSlice = (Math.PI * 2) / nodes.length;
  const centerX = 600;
  const centerY = 400;
  
  const layoutNodes = nodes.map((node, i) => ({
    ...node,
    x: centerX + radius * Math.cos(i * angleSlice - Math.PI / 2),
    y: centerY + radius * Math.sin(i * angleSlice - Math.PI / 2)
  }));
  
  return { nodes: layoutNodes, links: edges };
};

/**
 * Build hierarchy from flat node/edge list
 */
function buildHierarchy(nodes, edges) {
  const nodeMap = new Map(nodes.map(n => [n.id, { ...n, children: [] }]));
  
  // Find root (node with no incoming edges)
  const hasIncoming = new Set(edges.map(e => e.target));
  let rootId = nodes[0]?.id;
  
  for (const node of nodes) {
    if (!hasIncoming.has(node.id)) {
      rootId = node.id;
      break;
    }
  }
  
  // Build tree structure
  const edgeMap = new Map();
  edges.forEach(edge => {
    if (!edgeMap.has(edge.source)) {
      edgeMap.set(edge.source, []);
    }
    edgeMap.get(edge.source).push(edge.target);
  });
  
  function buildNode(id) {
    const node = nodeMap.get(id);
    const children = edgeMap.get(id) || [];
    node.children = children.map(childId => buildNode(childId));
    return node;
  }
  
  return d3.hierarchy(buildNode(rootId));
}

/**
 * Detect layout type automatically
 */
export const detectLayoutType = (nodes, edges) => {
  // Check for tree structure (single root, no cycles)
  const hasIncoming = new Set(edges.map(e => e.target));
  const roots = nodes.filter(n => !hasIncoming.has(n.id));
  
  if (roots.length === 1 && isAcyclic(nodes, edges)) {
    return layoutStrategies.tree;
  }
  
  // Check for DAG structure
  if (isAcyclic(nodes, edges)) {
    return layoutStrategies.dag;
  }
  
  // Default to force layout for general graphs
  return layoutStrategies.graph;
};

/**
 * Check if graph is acyclic
 */
function isAcyclic(nodes, edges) {
  const visited = new Set();
  const recStack = new Set();
  
  const adjacency = new Map();
  nodes.forEach(n => adjacency.set(n.id, []));
  edges.forEach(e => {
    if (!adjacency.has(e.source)) adjacency.set(e.source, []);
    adjacency.get(e.source).push(e.target);
  });
  
  function hasCycle(node) {
    visited.add(node);
    recStack.add(node);
    
    const neighbors = adjacency.get(node) || [];
    for (const neighbor of neighbors) {
      if (!visited.has(neighbor)) {
        if (hasCycle(neighbor)) return true;
      } else if (recStack.has(neighbor)) {
        return true;
      }
    }
    
    recStack.delete(node);
    return false;
  }
  
  for (const node of nodes) {
    if (!visited.has(node.id)) {
      if (hasCycle(node.id)) return false;
    }
  }
  
  return true;
}

/**
 * Layout selector - returns appropriate layout function
 */
export const selectLayout = async (nodes, edges, preferredType = null) => {
  const type = preferredType || detectLayoutType(nodes, edges);
  
  switch (type) {
    case layoutStrategies.tree:
      return { ...treeLayout(nodes, edges), type };
    case layoutStrategies.dag:
      return { ...dagreLayout(nodes, edges, 'TB'), type };
    case layoutStrategies.flowchart:
      return { ...dagreLayout(nodes, edges, 'TB'), type };
    case layoutStrategies.neural_network:
      return { ...dagreLayout(nodes, edges, 'LR'), type };
    case layoutStrategies.graph:
    default:
      return { ...await forceLayout(nodes, edges), type };
  }
};
