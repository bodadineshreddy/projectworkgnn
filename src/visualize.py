import networkx as nx
import matplotlib.pyplot as plt
import json
from pathlib import Path

def load_data():
    # Load predicted relationships
    with open('data/predicted_relationships.json', 'r') as f:
        relationships = json.load(f)
    
    # Load original tickets
    with open('data/tickets.json', 'r') as f:
        tickets = json.load(f)
    
    return tickets, relationships

def create_ticket_graph(tickets, relationships):
    G = nx.Graph()
    
    # Create ticket lookup for quick access
    ticket_lookup = {t['id']: t for t in tickets}
    
    # Add nodes with attributes
    for ticket in tickets:
        G.add_node(ticket['id'], 
                  title=ticket['title'],
                  priority=ticket['Priority'])
    
    # Add edges from predictions
    for rel in relationships:
        G.add_edge(rel['ticket1'], rel['ticket2'], 
                  weight=rel['confidence'])
    
    return G

def visualize_graph(G, save_path='data/ticket_graph.png'):
    plt.figure(figsize=(15, 10))
    
    # Create layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes with different colors based on priority
    priority_colors = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
    node_colors = [priority_colors[G.nodes[node]['priority']] for node in G.nodes()]
    
    # Draw network
    nx.draw(G, pos,
            node_color=node_colors,
            with_labels=True,
            node_size=1000,
            font_size=8,
            font_weight='bold',
            edge_color='gray',
            width=0.5)
    
    # Add edge labels (weights/distances)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    
    # Add title
    plt.title("Ticket Relationship Graph\nColors: Red=High Priority, Orange=Medium, Green=Low", 
              pad=20, fontsize=12)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_communities(G):
    # Detect communities using Louvain method
    communities = nx.community.louvain_communities(G)
    
    # Print community analysis
    print("\nCommunity Analysis:")
    for i, community in enumerate(communities):
        print(f"\nCommunity {i+1}:")
        for node in community:
            print(f"- {node}: {G.nodes[node]['title']}")

def main():
    # Load data
    tickets, relationships = load_data()
    
    # Create and visualize graph
    G = create_ticket_graph(tickets, relationships)
    visualize_graph(G)
    
    # Analyze graph structure
    print("\nGraph Statistics:")
    print(f"Number of tickets: {G.number_of_nodes()}")
    print(f"Number of relationships: {G.number_of_edges()}")
    
    # Find central tickets
    centrality = nx.betweenness_centrality(G)
    most_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nMost central tickets:")
    for ticket_id, score in most_central:
        print(f"- {ticket_id}: {G.nodes[ticket_id]['title']} (centrality: {score:.3f})")
    
    # Analyze communities
    analyze_communities(G)

if __name__ == '__main__':
    main()