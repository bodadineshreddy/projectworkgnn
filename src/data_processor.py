from typing import List, Dict
import torch
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
import numpy as np

class TicketDataProcessor:
    def __init__(self):
        """Initialize the data processor with a BERT model."""
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.priority_map = {'Low': 0, 'Medium': 1, 'High': 2}
        
    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """Get BERT embedding for text."""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()
    
    def _encode_priority(self, priority: str) -> torch.Tensor:
        """One-hot encode priority."""
        priority_idx = self.priority_map.get(priority, 1)  # Default to Medium
        return torch.eye(3)[priority_idx]
    
    def process_tickets(self, tickets: List[Dict]) -> Data:
        """
        Process ticket data into a PyTorch Geometric Data object.
        
        Args:
            tickets: List of ticket dictionaries containing id, title, text, etc.
            
        Returns:
            PyTorch Geometric Data object with node features and edge index
        """
        num_tickets = len(tickets)
        
        # Get text embeddings and priority encodings
        features = []
        for ticket in tickets:
            # Combine title, text and description
            text = f"{ticket['title']} {ticket['text']} {ticket.get('Description', '')}"
            text_embedding = self._get_text_embedding(text)
            priority_encoding = self._encode_priority(ticket.get('Priority', 'Medium'))
            
            # Concatenate features
            combined_features = torch.cat([text_embedding, priority_encoding])
            features.append(combined_features)
        
        # Stack features into node feature matrix
        x = torch.stack(features)
        
        # Create initial edge index (no edges)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index)