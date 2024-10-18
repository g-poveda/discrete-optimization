// src/algos/coloring.rs

use pyo3::prelude::*;
use petgraph::Undirected;
use petgraph::graph::{Graph, NodeIndex};
use std::collections::{HashMap, HashSet};
use petgraph::algo::dijkstra;
#[pyclass]
#[derive(Debug)]
pub struct PyGraph {
    graph: Graph<usize, f32, Undirected>, // The graph with nodes and edges
}

#[pymethods]
impl PyGraph {
    #[new]
    pub fn new() -> Self {
        PyGraph {
            graph: Graph::new_undirected(),
        }
    }

    pub fn create_graph_from_edges(&mut self, edges: Vec<(usize, usize)>) {
        let mut node_indices: HashMap<usize, NodeIndex> = HashMap::new();
        let mut unique_nodes = HashSet::new(); // To ensure unique node values

        // Collect unique nodes from the edges
        for (u, v) in &edges {
            unique_nodes.insert(*u);
            unique_nodes.insert(*v);
        }

        // Sort the unique nodes to maintain a consistent order
        let mut sorted_nodes: Vec<_> = unique_nodes.iter().copied().collect();
        sorted_nodes.sort_unstable();

        // Add nodes to the graph in the order of their values
        for &node_value in &sorted_nodes {
            node_indices.insert(node_value, self.graph.add_node(node_value));
        }

        // Add edges between nodes using the previously established indices
        for (u, v) in edges {
            let u_index = *node_indices.get(&u).unwrap();
            let v_index = *node_indices.get(&v).unwrap();

            // Add an edge between u and v (the undirected nature means one edge suffices)
            self.graph.add_edge(u_index, v_index, 1.0); // Using weight 1 for simplicity
        }
    }

    pub fn create_graph_from_edges_with_weights(&mut self, edges: Vec<(usize, usize, f32)>) {
        let mut node_indices: HashMap<usize, NodeIndex> = HashMap::new();

        for (u, v, w) in edges {
            // Add nodes for u and v if they haven't been added yet
            let u_index = *node_indices.entry(u).or_insert_with(|| self.graph.add_node(u));
            let v_index = *node_indices.entry(v).or_insert_with(|| self.graph.add_node(v));

            // Add an edge between u and v
            self.graph.add_edge(u_index, v_index, w); // Using weight 1 for simplicity
            self.graph.add_edge(v_index, u_index, w); // In case
        }
    }


    pub fn greedy_coloring(&self) -> Vec<usize> {
        let mut colors = vec![usize::MAX; self.graph.node_count()]; // Use usize::MAX for uncolored
        let mut available_colors = vec![true; self.graph.node_count()]; // Track available colors

        // Iterate through all nodes
        for node in self.graph.node_indices() {
            // Reset available colors
            available_colors.fill(true);

            // Mark colors of adjacent nodes as unavailable
            for neighbor in self.graph.neighbors(node) {
                if colors[neighbor.index()] != usize::MAX { // Check if neighbor is colored
                    available_colors[colors[neighbor.index()]] = false; // Mark that color as unavailable
                }
            }

            // Assign the first available color
            let mut color_to_assign = 0;
            while color_to_assign < available_colors.len() && !available_colors[color_to_assign] {
                color_to_assign += 1; // Find the next available color
            }

            // Assign the found color to the current node
            colors[node.index()] = color_to_assign;
        }

        colors // Return the colors vector
    }

    pub fn shortest_path(&self, source: usize, target: usize) -> (Option<f32>, Vec<usize>) {
        let source_index = self.graph.node_indices().find(|&n| self.graph[n] == source);
        let target_index = self.graph.node_indices().find(|&n| self.graph[n] == target);

        if source_index.is_none() || target_index.is_none() {
            return (None, Vec::new()); // Return None distance and empty path if nodes are not found
        }

        let result = dijkstra(
            &self.graph,
            source_index.unwrap(),
            None,
            |e| *e.weight(),
        );

        // Retrieve the distance to the target node
        let distance = result.get(&target_index.unwrap()).cloned();

        // Reconstruct the path
        let mut path = Vec::new();
        let mut current = target_index.unwrap();

        // Use a parent map to backtrack the path
        let mut parent_map: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        for (node, &dist) in result.iter() {
            for neighbor in self.graph.neighbors(*node) {
                if result.get(&neighbor).is_some() && dist + *self.graph.edge_weight(self.graph.find_edge(*node, neighbor).unwrap()).unwrap() == result[&neighbor] {
                    parent_map.insert(neighbor, *node); // Store the parent for path reconstruction
                }
            }
        }
        path.push(current.index());
        // Backtrack from target to source using the parent map
        while let Some(&parent) = parent_map.get(&current) {
            path.push(parent.index());
            current = parent;
            if current == source_index.unwrap() {
                break;
            }
        }

        path.reverse(); // Reverse the path to get it from source to target
        (distance, path)
    }
}

#[pymodule]
pub fn coloring_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    Ok(())
}
