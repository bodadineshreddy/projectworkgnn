import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index):
        x = x.float()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class EdgePredictor(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, z, edge_index):
        # Extract node pairs
        row, col = edge_index
        edge_features = torch.cat([z[row], z[col]], dim=1)
        
        # Apply MLPs
        x = F.relu(self.fc1(edge_features))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


class TicketGraphAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Initialize with smaller weights
        self.init_range = 0.1
        
        # Encoder layers with strict normalization
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256, eps=1e-6),  # Increased epsilon for stability
            nn.ReLU(),
            nn.Dropout(0.1),  # Reduced dropout
            nn.Linear(256, 128),
            nn.LayerNorm(128, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Edge prediction layers
        self.edge_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64, eps=1e-6),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
        
        # Use AdamW with lower learning rate and higher eps
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-5,  # Much lower learning rate
            weight_decay=0.01,
            eps=1e-8
        )
        self.max_grad_norm = 0.5  # Lower gradient clipping threshold
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, -self.init_range, self.init_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Add small epsilon to prevent zero values
        x = x + 1e-8
        node_embeddings = self.encoder(x)
        return node_embeddings
    
    def predict_relationships(self, data):
        self.eval()
        with torch.no_grad():
            node_embeddings = self(data.x)
            
            # Generate all possible pairs of nodes
            num_nodes = node_embeddings.size(0)
            node_pairs = torch.combinations(torch.arange(num_nodes), r=2)
            
            # Get embeddings for each pair
            embeddings_1 = node_embeddings[node_pairs[:, 0]]
            embeddings_2 = node_embeddings[node_pairs[:, 1]]
            
            # Concatenate embeddings and predict relationship
            pair_features = torch.cat([embeddings_1, embeddings_2], dim=1)
            confidence_scores = self.edge_predictor(pair_features)
            
            # Only keep edges with confidence above threshold
            threshold = 0.5
            mask = confidence_scores.squeeze() > threshold
            selected_pairs = node_pairs[mask]
            selected_scores = confidence_scores[mask]
            
            if len(selected_pairs) > 0:
                edge_index = selected_pairs.t()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                
            return edge_index, selected_scores

    def train_step(self, data):
        self.train()
        self.optimizer.zero_grad()
        
        try:
            # Skip empty graphs
            if data.edge_index.size(1) == 0:
                return 0.0
            
            # Forward pass with gradient scaling and error checking
            with torch.autograd.detect_anomaly():
                node_embeddings = self(data.x)
                
                # Early return if embeddings contain NaN
                if torch.isnan(node_embeddings).any():
                    print("Warning: NaN detected in embeddings")
                    return 0.0
                
                # Calculate edge predictions
                src_embeddings = node_embeddings[data.edge_index[0]]
                dst_embeddings = node_embeddings[data.edge_index[1]]
                edge_features = torch.cat([src_embeddings, dst_embeddings], dim=1)
                edge_pred = self.edge_predictor(edge_features)
                
                # Generate negative samples with proper validation
                num_nodes = data.x.size(0)
                neg_edge_index = self._get_negative_edges(data.edge_index, num_nodes)
                
                if neg_edge_index is not None and neg_edge_index.size(1) > 0:
                    neg_src_embeddings = node_embeddings[neg_edge_index[0]]
                    neg_dst_embeddings = node_embeddings[neg_edge_index[1]]
                    neg_edge_features = torch.cat([neg_src_embeddings, neg_dst_embeddings], dim=1)
                    neg_edge_pred = self.edge_predictor(neg_edge_features)
                    
                    # Calculate loss with positive and negative samples
                    pos_loss = F.binary_cross_entropy_with_logits(
                        edge_pred, 
                        torch.ones(edge_pred.size(0), device=edge_pred.device)
                    )
                    neg_loss = F.binary_cross_entropy_with_logits(
                        neg_edge_pred,
                        torch.zeros(neg_edge_pred.size(0), device=neg_edge_pred.device)
                    )
                    loss = pos_loss + neg_loss
                else:
                    # Fallback to only positive samples if no negative samples
                    loss = F.binary_cross_entropy_with_logits(
                        edge_pred,
                        torch.ones(edge_pred.size(0), device=edge_pred.device)
                    )
                
                # Apply loss scaling and clipping
                loss = loss * 0.5  # Scale down loss to prevent explosion
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                return loss.item()
                
        except RuntimeError as e:
            print(f"Warning: Runtime error in training step: {e}")
            return 0.0
        
    def _get_negative_edges(self, pos_edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if num_nodes < 2:
            return torch.zeros((2, 0), dtype=torch.long)
        
        # Create all possible edges
        rows, cols = [], []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                rows.append(i)
                cols.append(j)
        
        all_edges = torch.tensor([rows, cols], dtype=torch.long)
        
        # Convert positive edges to set for faster lookup
        pos_edges = set(map(tuple, pos_edge_index.t().tolist()))
        
        # Filter out existing edges
        neg_edges = []
        for i, (row, col) in enumerate(all_edges.t()):
            if (row.item(), col.item()) not in pos_edges:
                neg_edges.append(i)
        
        if not neg_edges:
            return torch.zeros((2, 0), dtype=torch.long)
        
        neg_edge_index = all_edges[:, neg_edges]
        
        # Randomly sample the same number of negative edges as positive edges
        num_neg = pos_edge_index.size(1)
        perm = torch.randperm(neg_edge_index.size(1))[:num_neg]
        return neg_edge_index[:, perm]