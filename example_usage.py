from src.train import train_model, load_tickets
from src.recommender import load_recommender
import json

def main():
    # 1. Train the model
    tickets = load_tickets('data/tickets.json')
    save_dir = 'data/model_checkpoints'
    print("Training model...")
    model, losses = train_model(tickets, epochs=300, save_path=save_dir)
    print("Training completed!")

    # 2. Initialize recommender system
    print("\nInitializing recommender system...")
    recommender = load_recommender('data/model_checkpoints/best_model.pt', 'data/tickets.json')
    
    # 3. Example: Find similar tickets for a new ticket
    new_ticket = {
        "id": "TEST-001",
        "title": "Application crashes during startup",
        "text": "Users are reporting that the application crashes immediately after launching",
        "Priority": "High"
    }
    
    print("\nFinding similar tickets...")
    similar_tickets = recommender.find_similar_tickets(new_ticket, k=5)
    
    # 4. Display results
    print("\nTop 5 similar tickets:")
    for i, result in enumerate(similar_tickets, 1):
        ticket = result['ticket']
        score = result['similarity_score']
        print(f"\n{i}. Ticket: {ticket['id']}")
        print(f"   Title: {ticket['title']}")
        print(f"   Similarity Score: {score:.3f}")
        if 'resolution' in ticket:
            print(f"   Resolution: {ticket['resolution']}")

if __name__ == "__main__":
    main()