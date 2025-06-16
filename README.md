# Ticket Relationship Analysis using GNN

This project implements an unsupervised Graph Neural Network (GNN) model to automatically identify relationships between tickets in an issue tracking system. Instead of relying on manually specified links, the system learns to discover meaningful connections between tickets based on their content and metadata.

## Features

- Unsupervised learning of ticket relationships using Graph Neural Networks
- Text embedding of ticket content using Sentence Transformers
- Consideration of ticket priorities in relationship analysis
- Visualization of discovered ticket relationships
- Evaluation metrics for relationship prediction quality

## Installation

1. Ensure you have Python 3.8+ installed
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `src/`
  - `data_processor.py`: Processes ticket data into graph format
  - `model.py`: GNN model implementation
  - `train.py`: Training script
  - `utils.py`: Utility functions for visualization and evaluation
- `tests/`: Unit tests
- `data/`: Directory for input data and results
- `notebooks/`: Jupyter notebooks for exploration and analysis

## Usage

1. Prepare your ticket data in JSON format:
```json
[
    {
        "id": "TICKET-1",
        "title": "Issue title",
        "text": "Issue description",
        "Description": "Detailed description",
        "Priority": "High"
    },
    ...
]
```

2. Train the model:
```python
from src.train import train_model, load_tickets

# Load your ticket data
tickets = load_tickets('data/tickets.json')

# Train the model
model, losses = train_model(tickets, epochs=100)
```

3. Get predicted relationships:
```python
from src.train import predict_relationships
from src.data_processor import TicketDataProcessor

# Process tickets
processor = TicketDataProcessor()
data = processor.process_tickets(tickets)

# Get predictions
relationships = predict_relationships(model, data, tickets)
```

4. Visualize the relationships:
```python
from src.utils import create_ticket_graph, visualize_ticket_graph

# Create and visualize graph
G = create_ticket_graph(tickets, relationships)
visualize_ticket_graph(G, save_path='ticket_graph.png')
```

## Model Architecture

The system uses a Graph Auto-Encoder (GAE) architecture with the following components:

1. Text Embedding:
   - Sentence Transformer for encoding ticket text
   - Priority encoding as additional features

2. Graph Neural Network:
   - Two Graph Convolutional layers
   - Dropout for regularization
   - Hidden dimension of 64

3. Edge Prediction:
   - Inner product decoder
   - Threshold-based relationship prediction

## Testing

Run the tests using pytest:
```bash
pytest tests/
```

## Citation

If you use this project in your research, please cite:

```bibtex
@software{ticket_gnn,
    title={Ticket Relationship Analysis using GNN},
    year={2025},
    description={An unsupervised GNN approach to discover relationships between tickets}
}
```