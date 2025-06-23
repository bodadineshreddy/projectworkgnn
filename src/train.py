import torch
from data_processor import TicketDataProcessor
from model import TicketGraphAutoencoder
import json
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def load_tickets(file_path: str) -> List[Dict]:
    """Load ticket data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def train_model(tickets: List[Dict], epochs: int = 300, save_path: str = None):
    """
    Train the GNN model on ticket data with improved training strategy.
    
    Args:
        tickets: List of ticket dictionaries
        epochs: Number of training epochs
        save_path: Path to save the model checkpoints
    """
    # Process tickets into graph data
    processor = TicketDataProcessor()
    data = processor.process_tickets(tickets)
    
    # Initialize model with improved architecture
    input_dim = data.x.size(1)
    model = TicketGraphAutoencoder(input_dim)
    
    # Create save directory if it doesn't exist
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    # Training loop with early stopping
    losses = []
    best_loss = float('inf')
    patience = 10
    no_improve_count = 0
    progress_bar = tqdm(range(epochs), desc='Training')
    
    for epoch in progress_bar:
        # Multiple training steps per epoch for better convergence
        epoch_losses = []
        for _ in range(5):  # 5 steps per epoch
            loss = model.train_step(data)
            if loss > 0:  # Only consider valid loss values
                epoch_losses.append(loss)
        
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Save best model and check early stopping
            if save_path and avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss': avg_loss,
                    'input_dim': input_dim
                }, os.path.join(save_path, 'best_model.pt'))
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Early stopping
            if no_improve_count >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
            
            # Save checkpoint every 50 epochs
            if save_path and (epoch + 1) % 50 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss': avg_loss,
                    'losses': losses,
                    'input_dim': input_dim
                }, os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pt'))
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if save_path:
        plt.savefig(os.path.join(save_path, 'training_curve.png'))
    plt.close()
    
    return model, losses

def predict_relationships(model, data, tickets):
    """Predict relationships between tickets."""
    model.eval()
    edge_index, confidence_scores = model.predict_relationships(data)
    
    relationships = []
    for i in range(edge_index.size(1)):
        source_idx = edge_index[0, i].item()
        target_idx = edge_index[1, i].item()
        confidence = confidence_scores[i].item()
        
        relationships.append({
            'ticket1': tickets[source_idx]['id'],
            'ticket2': tickets[target_idx]['id'],
            'confidence': float(confidence)
        })
    
    return relationships

def main():
    # Load and train
    tickets = load_tickets('data/tickets.json')
    save_dir = 'data/model_checkpoints'
    model, losses = train_model(tickets, epochs=300, save_path=save_dir)
    
    # Process tickets again for prediction
    processor = TicketDataProcessor()
    data = processor.process_tickets(tickets)
    
    # Get predictions
    relationships = predict_relationships(model, data, tickets)
    
    # Save predictions
    with open('data/predicted_relationships.json', 'w') as f:
        json.dump(relationships, f, indent=2)
    
    print(f"Training completed. Model and predictions saved in {save_dir}")

if __name__ == '__main__':
    main()