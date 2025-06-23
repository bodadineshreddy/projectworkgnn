import torch
import faiss
import json
import numpy as np
from typing import List, Dict
from data_processor import TicketDataProcessor
from model import TicketGraphAutoencoder

class TicketRecommender:
    def __init__(self, model_path: str):
        """Initialize recommender with trained model"""
        # Load the trained model
        checkpoint = torch.load(model_path)
        self.input_dim = checkpoint['input_dim']
        
        self.model = TicketGraphAutoencoder(self.input_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize FAISS index
        self.index = None
        self.ticket_lookup = {}
        
    def build_index(self, tickets: List[Dict]):
        """Build search index from existing tickets"""
        processor = TicketDataProcessor()
        data = processor.process_tickets(tickets)
        
        # Get embeddings from trained model
        with torch.no_grad():
            embeddings = self.model(data.x).cpu().numpy()
        
        # Initialize and populate FAISS index
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        
        # Store ticket lookup
        self.ticket_lookup = {i: ticket for i, ticket in enumerate(tickets)}
    
    def find_similar_tickets(self, new_ticket: Dict, k: int = 5) -> List[Dict]:
        """Find similar tickets for a new ticket"""
        # Process new ticket
        processor = TicketDataProcessor()
        query_data = processor.process_single_ticket(new_ticket)
        
        # Get embedding for new ticket
        with torch.no_grad():
            query_embedding = self.model(query_data.x).cpu().numpy()
        
        # Search similar tickets
        distances, indices = self.index.search(query_embedding, k)
        
        # Format results
        similar_tickets = []
        for idx, distance in zip(indices[0], distances[0]):
            ticket = self.ticket_lookup[idx]
            similarity_score = 1 / (1 + distance)  # Convert distance to similarity
            similar_tickets.append({
                'ticket': ticket,
                'similarity_score': float(similarity_score),
                'resolution': ticket.get('resolution', '')
            })
        
        return similar_tickets
    
    def get_resolution_synthesis(self, similar_tickets: List[Dict], new_ticket: Dict) -> str:
        """Get synthesized resolution based on similar tickets"""
        # This is a placeholder - integrate with your preferred LLM here
        context = "\n".join([
            f"Similar ticket {i+1}:\n"
            f"Title: {t['ticket']['title']}\n"
            f"Resolution: {t['ticket']['resolution']}\n"
            for i, t in enumerate(similar_tickets)
        ])
        
        prompt = f"""
        New ticket: {new_ticket['title']}
        
        Based on these similar tickets and their resolutions:
        {context}
        
        Suggest a possible resolution approach.
        """
        
        # Replace this with your LLM integration
        return prompt

def load_recommender(model_path: str, tickets_path: str) -> TicketRecommender:
    """Utility function to load and initialize the recommender"""
    # Initialize recommender
    recommender = TicketRecommender(model_path)
    
    # Load existing tickets
    with open(tickets_path, 'r') as f:
        existing_tickets = json.load(f)
    
    # Build search index
    recommender.build_index(existing_tickets)
    
    return recommender