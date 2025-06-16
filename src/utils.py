import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import torch
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score
import numpy as np

def create_ticket_graph(tickets: List[Dict], relationships: List[Dict]) -> nx.Graph:
    """
    Create a NetworkX graph from tickets and their relationships.
    
    Args:
        tickets: List of ticket dictionaries
        relationships: List of predicted relationships
    """
    G = nx.Graph()
    
    # Add nodes
    for ticket in tickets:
        G.add_node(ticket['id'], 
                  title=ticket['title'],
                  priority=ticket.get('Priority', 'Medium'))
    
    # Add edges
    for rel in relationships:
        G.add_edge(rel['ticket1'], rel['ticket2'], 
                  weight=rel['confidence'])
    
    return G

def visualize_ticket_graph(G: nx.Graph, save_path: str = None):
    """
    Visualize the ticket relationship graph.
    
    Args:
        G: NetworkX graph of tickets
        save_path: Optional path to save the visualization
    """
    plt.figure(figsize=(12, 8))
    
    # Create layout
    pos = nx.spring_layout(G)
    
    # Draw nodes with different colors based on priority
    priority_colors = {'High': 'red', 'Medium': 'yellow', 'Low': 'green'}
    node_colors = [priority_colors[G.nodes[node]['priority']] for node in G.nodes()]
    
    # Draw the graph
    nx.draw(G, pos, 
            node_color=node_colors,
            with_labels=True,
            node_size=1000,
            font_size=8)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def evaluate_predictions(true_edges, pred_edges, num_nodes=None):
    """
    Evaluate edge predictions using precision, recall, and F1 score.
    
    Args:
        true_edges: Ground truth edges as tensor of shape [2, num_edges]
        pred_edges: Predicted edges as tensor of shape [2, num_edges]
        num_nodes: Total number of nodes (optional)
        
    Returns:
        precision, recall, f1 scores
    """
    if num_nodes is None:
        num_nodes = max(
            torch.max(true_edges).item() + 1,
            torch.max(pred_edges).item() + 1
        )
    
    # Convert edge indices to adjacency matrices
    true_adj = torch.zeros(num_nodes, num_nodes)
    pred_adj = torch.zeros(num_nodes, num_nodes)
    
    true_adj[true_edges[0], true_edges[1]] = 1
    pred_adj[pred_edges[0], pred_edges[1]] = 1
    
    # Flatten adjacency matrices for sklearn metrics
    true_flat = true_adj.numpy().flatten()
    pred_flat = pred_adj.numpy().flatten()
    
    # Calculate metrics
    precision = precision_score(true_flat, pred_flat, zero_division=0)
    recall = recall_score(true_flat, pred_flat, zero_division=0)
    f1 = f1_score(true_flat, pred_flat, zero_division=0)
    
    return precision, recall, f1

def save_ticket_graph(G: nx.Graph, path: str):
    """Save the graph structure to JSON format."""
    graph_data = {
        'nodes': [{'id': node, 'data': G.nodes[node]} for node in G.nodes()],
        'edges': [{'source': u, 'target': v, 'weight': d['weight']} 
                 for u, v, d in G.edges(data=True)]
    }
    
    import json
    with open(path, 'w') as f:
        json.dump(graph_data, f, indent=2)