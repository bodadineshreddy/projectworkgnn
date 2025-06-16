import torch
from data_processor import TicketDataProcessor
from model import TicketGraphAutoencoder
import json
import numpy as np

# Test tickets with different scenarios
new_tickets = [
    {
        "id": "SEC-1001",
        "title": "OAuth2 implementation for auth service",
        "text": "Implement OAuth2 authentication flow",
        "Summary": "OAuth2 authentication implementation",
        "Description": "Add OAuth2 support to enhance security of the authentication system",
        "Priority": "High"
    },
    {
        "id": "PERF-1002",
        "title": "Redis caching for auth tokens",
        "text": "Implement Redis caching",
        "Summary": "Token caching implementation",
        "Description": "Use Redis to cache authentication tokens for better performance",
        "Priority": "Medium"
    },
    {
        "id": "UI-1003",
        "title": "OAuth login button styling",
        "text": "Update UI for OAuth buttons",
        "Summary": "OAuth UI improvements",
        "Description": "Style OAuth login buttons according to design guidelines",
        "Priority": "Low"
    }
]

def load_model(model_path, input_dim):
    """Load the trained model."""
    model = TicketGraphAutoencoder(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def main():
    # Load original tickets for combined prediction
    with open('data/tickets.json', 'r') as f:
        original_tickets = json.load(f)
    
    # Combine original and new tickets
    all_tickets = original_tickets + new_tickets
    
    # Process all tickets
    processor = TicketDataProcessor()
    data = processor.process_tickets(all_tickets)
    
    # Load trained model
    model = load_model('data/model_checkpoints/best_model.pt', data.x.size(1))
    
    # Get predictions
    edge_index, confidence_scores = model.predict_relationships(data, threshold=0.3)
    
    # Convert to list of relationships
    relationships = []
    for i in range(edge_index.size(1)):
        source_idx = edge_index[0, i].item()
        target_idx = edge_index[1, i].item()
        confidence = confidence_scores[i].item()
        
        relationships.append({
            'ticket1': all_tickets[source_idx]['id'],
            'ticket2': all_tickets[target_idx]['id'],
            'confidence': confidence
        })
    
    # Sort by confidence
    relationships = sorted(relationships, key=lambda x: x['confidence'], reverse=True)
    
    # Print relationship analysis
    print("\nDiscovered relationships (confidence > 0.3):")
    for rel in relationships:
        print(f"{rel['ticket1']} -> {rel['ticket2']} (confidence: {rel['confidence']:.3f})")
    
    # Save relationships
    with open('data/predicted_relationships.json', 'w') as f:
        json.dump(relationships, f, indent=2)
    
    print(f"\nTotal relationships found: {len(relationships)}")
    
    if relationships:
        confidences = [r['confidence'] for r in relationships]
        print(f"\nConfidence statistics:")
        print(f"Min confidence: {min(confidences):.3f}")
        print(f"Max confidence: {max(confidences):.3f}")
        print(f"Mean confidence: {np.mean(confidences):.3f}")
        print(f"Median confidence: {np.median(confidences):.3f}")

if __name__ == '__main__':
    main()