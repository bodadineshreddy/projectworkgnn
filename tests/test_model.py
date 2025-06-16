import pytest
import torch
import torch_geometric
from torch_geometric.data import Data
from src.model import TicketGNN, TicketGraphAutoencoder
from src.data_processor import TicketDataProcessor
from src.utils import evaluate_predictions

@pytest.fixture
def sample_tickets():
    return [
        {
            "id": "TEST-1",
            "title": "Test ticket one",
            "text": "This is a test ticket",
            "Description": "Test description one",
            "Priority": "High"
        },
        {
            "id": "TEST-2",
            "title": "Test ticket two",
            "text": "Another test ticket",
            "Description": "Test description two",
            "Priority": "Medium"
        },
        {
            "id": "TEST-3",
            "title": "Related to ticket one",
            "text": "This ticket is related to test one",
            "Description": "Test description three",
            "Priority": "High"
        }
    ]

def test_data_processor(sample_tickets):
    processor = TicketDataProcessor()
    data = processor.process_tickets(sample_tickets)
    
    assert data.x is not None
    assert data.edge_index is not None
    assert data.x.size(0) == len(sample_tickets)  # Number of nodes
    assert data.x.size(1) > 0  # Feature dimension

def test_gnn_model():
    input_dim = 768 + 3  # BERT embedding dim + priority encoding
    model = TicketGNN(input_dim)
    
    # Create dummy data
    x = torch.randn(3, input_dim)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    
    # Test forward pass
    out = model(x, edge_index)
    assert out is not None
    assert out.size(0) == 3  # Number of nodes
    assert out.size(1) == 64  # Hidden dimension

def test_ticket_gnn_initialization():
    input_dim = 32
    hidden_dim = 64
    model = TicketGNN(input_dim, hidden_dim)
    
    assert isinstance(model, TicketGNN)
    assert model.encoder.conv1.in_channels == input_dim
    assert model.encoder.conv1.out_channels == hidden_dim * 2
    assert model.encoder.conv2.out_channels == hidden_dim

def test_forward_pass():
    # Create dummy data
    input_dim = 32
    num_nodes = 10
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, 15))  # 15 random edges
    
    # Initialize model
    model = TicketGNN(input_dim)
    
    # Forward pass
    output = model(x, edge_index)
    
    # Check output shape
    assert output.shape == (num_nodes, 64)  # default hidden_dim=64
    assert not torch.isnan(output).any()  # No NaN values

def test_autoencoder_training(sample_tickets):
    # Process real ticket data
    processor = TicketDataProcessor()
    data = processor.process_tickets(sample_tickets)
    
    input_dim = data.x.size(1)
    model = TicketGraphAutoencoder(input_dim)
    
    # Test single training step
    loss = model.train_step(data)
    assert isinstance(loss, float)
    assert not torch.isnan(torch.tensor(loss))
    
    # Test edge prediction
    pred_edges = model.get_edge_predictions(data)
    assert pred_edges.size(0) == 2  # Edge index format [2, num_edges]

def test_edge_prediction():
    # Create dummy data
    input_dim = 32
    num_nodes = 10
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, 15))
    data = Data(x=x, edge_index=edge_index)
    
    # Initialize autoencoder
    autoencoder = TicketGraphAutoencoder(input_dim)
    
    # Get predictions
    pred_edges = autoencoder.get_edge_predictions(data)
    
    assert isinstance(pred_edges, torch.Tensor)
    assert pred_edges.dim() == 2
    assert pred_edges.shape[0] == 2  # Edge list format (2 x num_edges)

def test_evaluation_metrics():
    # Create sample true and predicted edges
    true_edges = torch.tensor([[0, 1, 1], [1, 2, 0]], dtype=torch.long)
    pred_edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    
    precision, recall, f1 = evaluate_predictions(true_edges, pred_edges)
    
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1

def test_end_to_end(sample_tickets):
    # Test complete pipeline
    processor = TicketDataProcessor()
    data = processor.process_tickets(sample_tickets)
    
    input_dim = data.x.size(1)
    model = TicketGraphAutoencoder(input_dim)
    
    # Train for a few epochs
    for _ in range(5):
        loss = model.train_step(data)
    
    # Get predictions
    edge_pred = model.get_edge_predictions(data)
    
    # Verify predictions format
    assert edge_pred.size(0) == 2
    assert edge_pred.dtype == torch.long