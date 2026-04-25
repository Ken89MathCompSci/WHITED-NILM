import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class LSTMModel(nn.Module):
    """LSTM model for NILM"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        if self.bidirectional:
            # Concatenate the last outputs from both directions
            last_out = lstm_out[:, -1, :]
        else:
            # Just take the last output
            last_out = lstm_out[:, -1, :]
        
        # Linear layer
        output = self.fc(last_out)
        
        return output

class GRUModel(nn.Module):
    """GRU model for NILM"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=False):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
    
    def forward(self, x):
        # GRU forward pass
        gru_out, _ = self.gru(x)
        
        # Take the output from the last time step
        last_out = gru_out[:, -1, :]
        
        # Linear layer
        output = self.fc(last_out)
        
        return output

class TCNBlock(nn.Module):
    """Temporal Convolutional Network block"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size-1) * dilation // 2,
            dilation=dilation
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class TCNModel(nn.Module):
    """Temporal Convolutional Network model for NILM"""
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2):
        super(TCNModel, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout)
            )
        
        self.tcn_layers = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # Permute to (batch_size, input_size, sequence_length) for 1D convolution
        x = x.permute(0, 2, 1)
        
        # TCN layers
        x = self.tcn_layers(x)
        
        # Global average pooling
        x = torch.mean(x, dim=2)
        
        # Linear layer
        x = self.fc(x)
        
        return x

class LiquidTimeLayer(nn.Module):
    """Liquid Time-Constant Neural Network Layer"""
    def __init__(self, input_size, hidden_size, dt=0.1):
        super(LiquidTimeLayer, self).__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Time constants (initialized to 1.0)
        self.tau = nn.Parameter(torch.ones(hidden_size))
        
        # Recurrent weights
        self.rec_weights = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)
        
        # Activation function
        self.tanh = nn.Tanh()
    
    def forward(self, x, hidden=None):
        """
        Forward pass with Euler integration
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            hidden: Hidden state tensor of shape (batch_size, hidden_size)
            
        Returns:
            New hidden state
        """
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Input projection
        input_proj = self.input_proj(x)
        
        # Recurrent projection
        rec_proj = torch.matmul(hidden, self.rec_weights)
        
        # ODE integration (Euler method)
        # dh/dt = -h/tau + f(Wx + Uh)
        f_t = self.tanh(input_proj + rec_proj)
        dh = (-hidden / self.tau.unsqueeze(0) + f_t) * self.dt
        
        # Update hidden state
        new_hidden = (hidden + dh).clamp(-10.0, 10.0)  # bound hidden state to prevent BPTT explosion
        
        return new_hidden

class LiquidNetworkModel(nn.Module):
    """Liquid Neural Network for NILM"""
    def __init__(self, input_size, hidden_size, output_size, dt=0.1):
        super(LiquidNetworkModel, self).__init__()
        self.hidden_size = hidden_size
        
        # Liquid time layer
        self.liquid_layer = LiquidTimeLayer(input_size, hidden_size, dt)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor
        """
        # x shape: (batch_size, sequence_length, input_size)
        batch_size, seq_length, _ = x.size()
        
        # Initialize hidden state
        hidden = None
        
        # Process each time step
        for t in range(seq_length):
            x_t = x[:, t, :]
            hidden = self.liquid_layer(x_t, hidden)
        
        # Use the final hidden state for prediction
        output = self.fc(hidden)
        
        return output

# More sophisticated Liquid Neural Network implementation
class AdvancedLiquidTimeLayer(nn.Module):
    """Advanced Liquid Time-Constant Neural Network Layer with adaptive time constants"""
    def __init__(self, input_size, hidden_size, dt=0.1):
        super(AdvancedLiquidTimeLayer, self).__init__()
        self.hidden_size = hidden_size
        self.dt = dt

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Base time constants — stored in log-space so softplus keeps them positive
        self.tau_base = nn.Parameter(torch.ones(hidden_size))

        # Adaptive time constants modulation
        self.tau_mod = nn.Linear(input_size, hidden_size)

        # Recurrent weights
        self.rec_weights = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)

        # Gate for controlling information flow
        self.gate = nn.Linear(input_size + hidden_size, hidden_size)

        # Activation functions
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden=None):
        batch_size = x.size(0)

        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Input projection
        input_proj = self.input_proj(x)

        # Recurrent projection
        rec_proj = torch.matmul(hidden, self.rec_weights)

        # Adaptive time constants — softplus ensures tau_base > 0, clamp prevents near-zero
        tau_base = torch.nn.functional.softplus(self.tau_base).unsqueeze(0)
        tau_mod = self.sigmoid(self.tau_mod(x))
        tau = (tau_base * tau_mod).clamp(min=self.dt)  # Euler stability: tau >= dt

        # Input-dependent gate
        combined = torch.cat([x, hidden], dim=1)
        gate = self.sigmoid(self.gate(combined))

        # ODE integration with gating
        f_t = self.tanh(input_proj + rec_proj)
        dh = ((-hidden / tau) + gate * f_t) * self.dt

        new_hidden = (hidden + dh).clamp(-10.0, 10.0)  # bound hidden state to prevent BPTT explosion

        return new_hidden

class AdvancedLiquidNetworkModel(nn.Module):
    """Advanced Liquid Neural Network for NILM"""
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dt=0.1):
        super(AdvancedLiquidNetworkModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Multiple liquid time layers
        self.liquid_layers = nn.ModuleList([
            AdvancedLiquidTimeLayer(
                input_size if i == 0 else hidden_size,
                hidden_size,
                dt
            ) for i in range(num_layers)
        ])

        # LayerNorm between stacked layers to prevent hidden state explosion
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        hidden_states = [None] * self.num_layers

        for t in range(seq_length):
            x_t = x[:, t, :]

            for i in range(self.num_layers):
                inp = x_t if i == 0 else self.layer_norms[i - 1](hidden_states[i - 1])
                hidden_states[i] = self.liquid_layers[i](inp, hidden_states[i])

        output = self.fc(self.layer_norms[-1](hidden_states[-1]))

        return output

class AdvancedLiquidNetworkModelTwo(nn.Module):
    """Advanced LNN v2 — adds intra-layer LayerNorm (normalises h_{t-1} each timestep)
    in addition to the inter-layer LayerNorm between stacked layers."""
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dt=0.1):
        super(AdvancedLiquidNetworkModelTwo, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.liquid_layers = nn.ModuleList([
            AdvancedLiquidTimeLayer(
                input_size if i == 0 else hidden_size,
                hidden_size,
                dt
            ) for i in range(num_layers)
        ])

        # Inter-layer LayerNorm: normalises hidden state crossing layer boundaries
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

        # Intra-layer LayerNorm: normalises h_{t-1} before each timestep recurrence
        self.intra_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        hidden_states = [None] * self.num_layers

        for t in range(seq_length):
            x_t = x[:, t, :]

            for i in range(self.num_layers):
                # Inter: normalise signal coming from previous layer
                inp = x_t if i == 0 else self.layer_norms[i - 1](hidden_states[i - 1])
                # Intra: normalise h_{t-1} before feeding back into same layer
                h_prev = (self.intra_norms[i](hidden_states[i])
                          if hidden_states[i] is not None else None)
                hidden_states[i] = self.liquid_layers[i](inp, h_prev)

        output = self.fc(self.layer_norms[-1](hidden_states[-1]))

        return output

class ResidualBlock(nn.Module):
    """Residual block for ResNet architecture"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=5, 
            stride=stride, 
            padding=2
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(
            out_channels, 
            out_channels, 
            kernel_size=5, 
            padding=2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection (shortcut)
        self.shortcut = nn.Sequential()
        # If dimensions change, we need to adjust the shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        # Store the input for the skip connection
        residual = x
        
        # First conv block
        out = self.relu(self.bn1(self.conv1(x)))
        
        # Second conv block (no activation yet)
        out = self.bn2(self.conv2(out))
        
        # Add skip connection
        out += self.shortcut(residual)
        
        # Apply ReLU activation after addition
        out = self.relu(out)
        
        return out

class ResNetModel(nn.Module):
    """ResNet model for NILM"""
    def __init__(self, input_size, output_size, layers=[2, 2, 2], base_width=32):
        super(ResNetModel, self).__init__()
        
        self.in_channels = base_width
        
        # Initial convolutional layer
        self.conv1 = nn.Conv1d(input_size, self.in_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(base_width, layers[0])
        self.layer2 = self._make_layer(base_width*2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_width*4, layers[2], stride=2)
        
        # Global average pooling and final fully connected layer
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_width*4, output_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, out_channels, num_blocks, stride=1):
        layers = []
        
        # First block may have stride > 1 to reduce spatial dimensions
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        
        # Update in_channels for subsequent blocks
        self.in_channels = out_channels
        
        # Add remaining blocks (with stride=1)
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # Permute for 1D convolution: (batch_size, input_size, sequence_length)
        x = x.permute(0, 2, 1)
        
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global average pooling
        x = self.avgpool(x)
        
        # Flatten and pass through fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class SimpleMultiHeadAttention(nn.Module):
    """Simple multi-head attention mechanism for lightweight transformer"""
    def __init__(self, embed_dim, num_heads=4):
        super(SimpleMultiHeadAttention, self).__init__()
        
        # Ensure embed_dim is divisible by num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections for query, key, value
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Scaling factor for dot-product attention
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)
        batch_size, seq_length, _ = x.size()
        
        # Linear projections for Q, K, V
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Reshape for multi-head attention:
        # (batch_size, seq_length, embed_dim) -> (batch_size, num_heads, seq_length, head_dim)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape back to original dimensions:
        # (batch_size, num_heads, seq_length, head_dim) -> (batch_size, seq_length, embed_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        
        # Final output projection
        output = self.out_proj(context)
        
        return output

class TransformerEncoderLayer(nn.Module):
    """Simplified Transformer encoder layer"""
    def __init__(self, embed_dim, num_heads=4, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-head attention
        self.self_attn = SimpleMultiHeadAttention(embed_dim, num_heads)
        
        # Feed-forward network
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Activation function
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Self-attention block with residual connection and layer norm
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x)
        x = residual + self.dropout(x)
        
        # Feed-forward block with residual connection and layer norm
        residual = x
        x = self.norm2(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = residual + self.dropout(x)
        
        return x

class SimplePositionalEncoding(nn.Module):
    """Simple positional encoding for transformer models"""
    def __init__(self, embed_dim, max_len=5000):
        super(SimplePositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        
        # Sine for even indices, cosine for odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)
        # Add positional encoding to the input
        return x + self.pe[:, :x.size(1), :]

class SimpleTransformerModel(nn.Module):
    """Simplified Transformer model for NILM"""
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, num_heads=4, dropout=0.1):
        super(SimpleTransformerModel, self).__init__()
        
        # Input embedding
        self.input_embedding = nn.Linear(input_size, hidden_size)
        
        # Positional encoding
        self.pos_encoding = SimplePositionalEncoding(hidden_size)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_size, 
                num_heads=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        
        # Input embedding
        x = self.input_embedding(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Global average pooling over the sequence dimension
        x = torch.mean(x, dim=1)

        # Output layer
        x = self.output_layer(x)

        return x


# ============================================================================
# ATTENTION-ENHANCED LIQUID NEURAL NETWORKS
# ============================================================================

class SelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism for temporal sequences.
    Helps the model focus on relevant time steps in the input sequence.
    """
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.out = nn.Linear(hidden_size, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = x.size()

        # Save residual connection
        residual = x

        # Linear projections and split into multiple heads
        # Shape: (batch_size, seq_len, hidden_size) -> (batch_size, num_heads, seq_len, head_dim)
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate heads
        # Shape: (batch_size, seq_len, hidden_size)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

        # Final linear projection
        output = self.out(attention_output)
        output = self.dropout(output)

        # Add residual connection and layer normalization
        output = self.layer_norm(output + residual)

        return output


class LiquidODECell(nn.Module):
    """
    Liquid ODE Cell for continuous-time recurrent computation.

    This cell implements a simple liquid time-constant RNN cell using ODE integration.
    It's designed to work with the AttentionLiquidNetworkModel.
    """
    def __init__(self, input_size, hidden_size, dt=0.1):
        super(LiquidODECell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Time constants (initialized to 1.0)
        self.tau = nn.Parameter(torch.ones(hidden_size))

        # Recurrent weights
        self.rec_weights = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)

        # Activation function
        self.tanh = nn.Tanh()

    def forward(self, x, hidden):
        """
        Forward pass with Euler integration.

        Args:
            x: Input tensor of shape (batch_size, input_size)
            hidden: Hidden state tensor of shape (batch_size, hidden_size)

        Returns:
            New hidden state of shape (batch_size, hidden_size)
        """
        # Input projection
        input_proj = self.input_proj(x)

        # Recurrent projection
        rec_proj = torch.matmul(hidden, self.rec_weights)

        # ODE integration (Euler method)
        # dh/dt = -h/tau + f(Wx + Uh)
        f_t = self.tanh(input_proj + rec_proj)
        dh = (-hidden / self.tau.unsqueeze(0) + f_t) * self.dt

        # Update hidden state
        new_hidden = (hidden + dh).clamp(-10.0, 10.0)  # bound hidden state to prevent BPTT explosion

        return new_hidden


class AttentionLiquidNetworkModel(nn.Module):
    """
    Attention-Enhanced Liquid Neural Network for NILM.

    Architecture:
        Input -> Liquid Layer (ODE) -> Self-Attention -> FC -> Output

    The self-attention mechanism helps the model focus on relevant time steps
    in the sequence, improving performance on appliance state transitions.
    """
    def __init__(self, input_size, hidden_size, output_size, dt=0.1, num_heads=4, dropout=0.1):
        super(AttentionLiquidNetworkModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dt = dt

        # Input projection
        self.input_layer = nn.Linear(input_size, hidden_size)

        # Single liquid layer
        self.liquid = LiquidODECell(hidden_size, hidden_size, dt=dt)

        # Self-attention mechanism
        self.attention = SelfAttention(hidden_size, num_heads=num_heads, dropout=dropout)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size, seq_len, _ = x.size()

        # Project input to hidden dimension
        x = self.input_layer(x)  # (batch_size, seq_len, hidden_size)

        # Process through liquid layer
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        liquid_outputs = []

        for t in range(seq_len):
            h = self.liquid(x[:, t, :], h)
            liquid_outputs.append(h)

        # Stack outputs: (batch_size, seq_len, hidden_size)
        liquid_outputs = torch.stack(liquid_outputs, dim=1)

        # Apply self-attention
        attended = self.attention(liquid_outputs)  # (batch_size, seq_len, hidden_size)

        # Use the last time step output
        output = attended[:, -1, :]  # (batch_size, hidden_size)

        # Final prediction
        output = self.fc(output)  # (batch_size, output_size)

        return output


class CNNEncoderLiquidNetworkModel(nn.Module):
    """
    CNN Encoder + Liquid Neural Network for NILM.

    Architecture:
        Input -> CNN Encoder (feature extraction) -> Liquid Layer (ODE) -> FC -> Output

    The CNN encoder extracts local temporal patterns and features,
    which are then processed by the liquid layer for temporal dynamics.
    """
    def __init__(self, input_size, hidden_size, output_size, dt=0.1,
                 num_conv_layers=3, kernel_size=5, dropout=0.1):
        super(CNNEncoderLiquidNetworkModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dt = dt

        # CNN Encoder
        self.conv_layers = nn.ModuleList()
        in_channels = input_size
        out_channels = hidden_size // 4

        for i in range(num_conv_layers):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            in_channels = out_channels
            out_channels = min(out_channels * 2, hidden_size)

        # Project CNN output to hidden size
        self.encoder_projection = nn.Linear(in_channels, hidden_size)

        # Liquid layer
        self.liquid = LiquidODECell(hidden_size, hidden_size, dt=dt)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size, seq_len, _ = x.size()

        # CNN expects (batch_size, channels, seq_len)
        x = x.transpose(1, 2)  # (batch_size, input_size, seq_len)

        # Apply CNN encoder
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Back to (batch_size, seq_len, features)
        x = x.transpose(1, 2)  # (batch_size, seq_len, features)

        # Project to hidden size
        x = self.encoder_projection(x)  # (batch_size, seq_len, hidden_size)

        # Process through liquid layer
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        for t in range(seq_len):
            h = self.liquid(x[:, t, :], h)

        # Final prediction
        output = self.fc(h)  # (batch_size, output_size)

        return output


class TCNLiquidNetworkModel(nn.Module):
    """
    TCN Encoder + Basic Liquid Neural Network for NILM.

    Architecture:
        Input -> TCN Blocks (dilated conv, feature extraction) -> Linear projection
              -> LiquidODECell (ODE recurrence) -> FC -> Output

    TCN handles local pattern extraction with exponentially growing receptive field.
    The basic LNN ODE cell integrates temporal dynamics over the sequence.
    """
    def __init__(self, input_size, hidden_size, output_size, dt=0.1,
                 num_channels=None, kernel_size=3, dropout=0.2):
        super(TCNLiquidNetworkModel, self).__init__()
        self.hidden_size = hidden_size
        self.dt = dt

        if num_channels is None:
            num_channels = [32, 64, 128]

        # TCN Encoder: stack of dilated TCNBlocks
        tcn_layers = []
        in_ch = input_size
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            tcn_layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.tcn_encoder = nn.Sequential(*tcn_layers)

        # Project TCN output channels to hidden_size
        self.encoder_projection = nn.Linear(num_channels[-1], hidden_size)

        # Basic LNN ODE cell
        self.liquid = LiquidODECell(hidden_size, hidden_size, dt=dt)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size, seq_len, _ = x.size()

        # TCN expects (batch_size, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.tcn_encoder(x)                  # (batch_size, num_channels[-1], seq_len)
        x = x.transpose(1, 2)                    # (batch_size, seq_len, num_channels[-1])

        # Project to hidden size
        x = self.encoder_projection(x)           # (batch_size, seq_len, hidden_size)

        # Step through LNN ODE cell
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(seq_len):
            h = self.liquid(x[:, t, :], h)

        return self.fc(h)


class TCNAdvancedLiquidNetworkModel(nn.Module):
    """
    TCN Encoder + AdvancedLiquidNetworkModel for NILM.

    Architecture:
        Input -> TCN Blocks -> Linear projection
              -> 2x AdvancedLiquidTimeLayer (adaptive tau, input gate,
                 inter-layer LayerNorm only) -> FC -> Output
    """
    def __init__(self, input_size, hidden_size, output_size, dt=0.1,
                 num_channels=None, kernel_size=3, dropout=0.2, num_layers=2):
        super(TCNAdvancedLiquidNetworkModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if num_channels is None:
            num_channels = [32, 64, 128]

        # TCN Encoder
        tcn_layers = []
        in_ch = input_size
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            tcn_layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.tcn_encoder = nn.Sequential(*tcn_layers)

        self.encoder_projection = nn.Linear(num_channels[-1], hidden_size)

        # Stacked AdvancedLiquidTimeLayer with inter-layer LayerNorm only
        self.liquid_layers = nn.ModuleList([
            AdvancedLiquidTimeLayer(
                hidden_size if i == 0 else hidden_size,
                hidden_size,
                dt
            ) for i in range(num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        x = x.transpose(1, 2)
        x = self.tcn_encoder(x)
        x = x.transpose(1, 2)
        x = self.encoder_projection(x)

        hidden_states = [None] * self.num_layers

        for t in range(seq_len):
            x_t = x[:, t, :]
            for i in range(self.num_layers):
                inp = x_t if i == 0 else self.layer_norms[i - 1](hidden_states[i - 1])
                hidden_states[i] = self.liquid_layers[i](inp, hidden_states[i])

        return self.fc(self.layer_norms[-1](hidden_states[-1]))


class TCNAdvancedLiquidNetworkMultiHeadModel(nn.Module):
    """
    Shared TCN Encoder + Separate LNN Head per Appliance for multi-task NILM.

    Architecture:
        Input -> Shared TCN Blocks -> Shared Linear projection
              -> Per-appliance stacked AdvancedLiquidTimeLayer (inter-LayerNorm only)
              -> Per-appliance FC -> Output (batch, num_appliances)

    The shared encoder learns general temporal features from the mains signal,
    while each appliance head models its own specific consumption dynamics.

    Args:
        input_size:      feature dim per timestep (1 for scalar power)
        hidden_size:     hidden size for projection and all LNN heads
        num_appliances:  number of appliance heads (default 4)
        dt:              LNN Euler integration step
        num_channels:    TCN channel sizes per block (default [32, 64, 128])
        kernel_size:     TCN kernel size
        dropout:         TCN dropout
        num_layers:      number of AdvancedLiquidTimeLayer per head
    """
    def __init__(self, input_size, hidden_size, num_appliances=4, dt=0.1,
                 num_channels=None, kernel_size=3, dropout=0.2, num_layers=2):
        super(TCNAdvancedLiquidNetworkMultiHeadModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_appliances = num_appliances

        if num_channels is None:
            num_channels = [32, 64, 128]

        # Shared TCN encoder
        tcn_layers = []
        in_ch = input_size
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            tcn_layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.tcn_encoder = nn.Sequential(*tcn_layers)
        self.encoder_projection = nn.Linear(num_channels[-1], hidden_size)

        # Per-appliance LNN heads: each has its own liquid layers + layer norms + FC
        self.liquid_layers = nn.ModuleList([
            nn.ModuleList([
                AdvancedLiquidTimeLayer(hidden_size, hidden_size, dt)
                for _ in range(num_layers)
            ])
            for _ in range(num_appliances)
        ])
        self.layer_norms = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(hidden_size)
                for _ in range(num_layers)
            ])
            for _ in range(num_appliances)
        ])
        self.fc_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_appliances)
        ])

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            (batch_size, num_appliances)
        """
        batch_size, seq_len, _ = x.size()

        # Shared TCN encoding
        x = x.transpose(1, 2)
        x = self.tcn_encoder(x)
        x = x.transpose(1, 2)
        x = self.encoder_projection(x)  # (batch, seq_len, hidden_size)

        # Per-appliance LNN heads
        outputs = []
        for a in range(self.num_appliances):
            hidden_states = [None] * self.num_layers
            for t in range(seq_len):
                x_t = x[:, t, :]
                for i in range(self.num_layers):
                    inp = x_t if i == 0 else self.layer_norms[a][i - 1](hidden_states[i - 1])
                    hidden_states[i] = self.liquid_layers[a][i](inp, hidden_states[i])
            out = self.fc_heads[a](self.layer_norms[a][-1](hidden_states[-1]))
            outputs.append(out)

        return torch.cat(outputs, dim=1)  # (batch, num_appliances)


class TCNAdvancedLiquidNetworkModelTwo(nn.Module):
    """
    TCN Encoder + AdvancedLiquidNetworkModelTwo for NILM.

    Architecture:
        Input -> TCN Blocks (dilated conv, feature extraction) -> Linear projection
              -> 2x AdvancedLiquidTimeLayer (adaptive tau, input gate,
                 inter-layer + intra-layer LayerNorm) -> FC -> Output
    """
    def __init__(self, input_size, hidden_size, output_size, dt=0.1,
                 num_channels=None, kernel_size=3, dropout=0.2, num_layers=2):
        super(TCNAdvancedLiquidNetworkModelTwo, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if num_channels is None:
            num_channels = [32, 64, 128]

        # TCN Encoder: stack of dilated TCNBlocks
        tcn_layers = []
        in_ch = input_size
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            tcn_layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.tcn_encoder = nn.Sequential(*tcn_layers)

        # Project TCN output channels to hidden_size
        self.encoder_projection = nn.Linear(num_channels[-1], hidden_size)

        # Stacked AdvancedLiquidTimeLayer (same as AdvancedLiquidNetworkModelTwo)
        self.liquid_layers = nn.ModuleList([
            AdvancedLiquidTimeLayer(
                hidden_size if i == 0 else hidden_size,
                hidden_size,
                dt
            ) for i in range(num_layers)
        ])

        # Inter-layer LayerNorm: normalises hidden state crossing layer boundaries
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

        # Intra-layer LayerNorm: normalises h_{t-1} before each timestep recurrence
        self.intra_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # TCN expects (batch_size, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.tcn_encoder(x)               # (batch_size, num_channels[-1], seq_len)
        x = x.transpose(1, 2)                 # (batch_size, seq_len, num_channels[-1])

        # Project to hidden size
        x = self.encoder_projection(x)        # (batch_size, seq_len, hidden_size)

        # Step through stacked advanced LNN layers
        hidden_states = [None] * self.num_layers

        for t in range(seq_len):
            x_t = x[:, t, :]

            for i in range(self.num_layers):
                inp = x_t if i == 0 else self.layer_norms[i - 1](hidden_states[i - 1])
                h_prev = (self.intra_norms[i](hidden_states[i])
                          if hidden_states[i] is not None else None)
                hidden_states[i] = self.liquid_layers[i](inp, h_prev)

        return self.fc(self.layer_norms[-1](hidden_states[-1]))


class LSTMLiquidNetworkModel(nn.Module):
    """
    Bidirectional LSTM Encoder + Liquid Neural Network for NILM.

    Architecture:
        Input -> BiLSTM (variable-length dependency extraction)
              -> Linear projection -> LiquidODECell (ODE recurrence) -> FC -> Output

    Unlike TCN's fixed receptive field, LSTM captures variable-length dependencies
    from both directions. The LNN then refines the temporal dynamics with its
    continuous-time ODE recurrence.

    Args:
        input_size:    feature dim per timestep (1 for scalar power)
        lstm_hidden:   hidden size of the LSTM (each direction)
        hidden_size:   hidden size for the LNN and projection
        output_size:   number of outputs (1 for single appliance)
        num_lstm_layers: number of stacked LSTM layers
        dropout:       dropout between LSTM layers (only applied if num_lstm_layers > 1)
        dt:            LNN Euler integration step
    """
    def __init__(self, input_size, hidden_size, output_size,
                 lstm_hidden=64, num_lstm_layers=2, dropout=0.2, dt=0.1):
        super(LSTMLiquidNetworkModel, self).__init__()
        self.hidden_size = hidden_size
        self.dt = dt

        # Bidirectional LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0
        )

        # Project BiLSTM output (lstm_hidden * 2) to hidden_size for LNN
        self.encoder_projection = nn.Linear(lstm_hidden * 2, hidden_size)

        # LiquidODECell
        self.liquid = LiquidODECell(hidden_size, hidden_size, dt=dt)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            (batch_size, output_size)
        """
        batch_size, seq_len, _ = x.size()

        # BiLSTM encode: (batch, seq_len, lstm_hidden * 2)
        lstm_out, _ = self.lstm(x)

        # Project to hidden_size: (batch, seq_len, hidden_size)
        x = self.encoder_projection(lstm_out)

        # Step through LNN ODE cell
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(seq_len):
            h = self.liquid(x[:, t, :], h)

        return self.fc(h)


class GRULiquidNetworkModel(nn.Module):
    """
    Bidirectional GRU Encoder + Liquid Neural Network for NILM.

    Architecture:
        Input -> BiGRU (variable-length dependency extraction)
              -> Linear projection -> LiquidODECell (ODE recurrence) -> FC -> Output

    GRU is lighter than LSTM (no separate cell state) while still capturing
    variable-length context from both directions before the LNN refines dynamics.

    Args:
        input_size:    feature dim per timestep (1 for scalar power)
        gru_hidden:    hidden size of the GRU (each direction)
        hidden_size:   hidden size for the LNN and projection
        output_size:   number of outputs (1 for single appliance)
        num_gru_layers: number of stacked GRU layers
        dropout:       dropout between GRU layers (only applied if num_gru_layers > 1)
        dt:            LNN Euler integration step
    """
    def __init__(self, input_size, hidden_size, output_size,
                 gru_hidden=64, num_gru_layers=2, dropout=0.2, dt=0.1):
        super(GRULiquidNetworkModel, self).__init__()
        self.hidden_size = hidden_size
        self.dt = dt

        # Bidirectional GRU encoder
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=gru_hidden,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_gru_layers > 1 else 0.0
        )

        # Project BiGRU output (gru_hidden * 2) to hidden_size for LNN
        self.encoder_projection = nn.Linear(gru_hidden * 2, hidden_size)

        # LiquidODECell
        self.liquid = LiquidODECell(hidden_size, hidden_size, dt=dt)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            (batch_size, output_size)
        """
        batch_size, seq_len, _ = x.size()

        # BiGRU encode: (batch, seq_len, gru_hidden * 2)
        gru_out, _ = self.gru(x)

        # Project to hidden_size: (batch, seq_len, hidden_size)
        x = self.encoder_projection(gru_out)

        # Step through LNN ODE cell
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(seq_len):
            h = self.liquid(x[:, t, :], h)

        return self.fc(h)


class GRUAdvancedLiquidNetworkModel(nn.Module):
    """
    Bidirectional GRU Encoder + AdvancedLiquidNetworkModel for NILM.

    Architecture:
        Input -> BiGRU -> Linear projection -> stacked AdvancedLiquidTimeLayer
              (with inter-layer LayerNorm) -> FC -> Output

    Mirrors the AdvancedLiquidNetworkModel (v1) pattern: inter-layer LayerNorm only.

    Args:
        input_size:    feature dim per timestep (1 for scalar power)
        hidden_size:   hidden size for the LNN layers and projection
        output_size:   number of outputs (1 for single appliance)
        gru_hidden:    hidden size of the GRU (each direction)
        num_gru_layers: number of stacked GRU layers
        dropout:       dropout between GRU layers (only if num_gru_layers > 1)
        dt:            LNN Euler integration step
        num_layers:    number of stacked AdvancedLiquidTimeLayer layers
    """
    def __init__(self, input_size, hidden_size, output_size,
                 gru_hidden=64, num_gru_layers=2, dropout=0.2, dt=0.1, num_layers=2):
        super(GRUAdvancedLiquidNetworkModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=gru_hidden,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_gru_layers > 1 else 0.0
        )

        self.encoder_projection = nn.Linear(gru_hidden * 2, hidden_size)

        self.liquid_layers = nn.ModuleList([
            AdvancedLiquidTimeLayer(hidden_size, hidden_size, dt) for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            (batch_size, output_size)
        """
        batch_size, seq_len, _ = x.size()

        gru_out, _ = self.gru(x)
        x = self.encoder_projection(gru_out)

        hidden_states = [None] * self.num_layers
        for t in range(seq_len):
            x_t = x[:, t, :]
            for i in range(self.num_layers):
                inp = x_t if i == 0 else self.layer_norms[i - 1](hidden_states[i - 1])
                hidden_states[i] = self.liquid_layers[i](inp, hidden_states[i])

        return self.fc(self.layer_norms[-1](hidden_states[-1]))


class TransformerEncoderLiquidNetworkModel(nn.Module):
    """
    Transformer Encoder + Liquid Neural Network for NILM.

    Architecture:
        Input -> Transformer Encoder (self-attention) -> Liquid Layer (ODE) -> FC -> Output

    The Transformer encoder captures long-range dependencies and global patterns,
    which are then refined by the liquid layer for temporal dynamics.
    """
    def __init__(self, input_size, hidden_size, output_size, dt=0.1,
                 num_encoder_layers=2, num_heads=4, dropout=0.1):
        super(TransformerEncoderLiquidNetworkModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dt = dt

        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Liquid layer
        self.liquid = LiquidODECell(hidden_size, hidden_size, dt=dt)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size, seq_len, _ = x.size()

        # Project input
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_size)

        # Apply Transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, seq_len, hidden_size)

        # Process through liquid layer
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        for t in range(seq_len):
            h = self.liquid(x[:, t, :], h)

        # Final prediction
        output = self.fc(h)  # (batch_size, output_size)

        return output


class BidirectionalEncoderLiquidNetworkModel(nn.Module):
    """
    Bidirectional Encoder + Liquid Neural Network for NILM.

    Architecture:
        Input -> Bidirectional Liquid Layers (forward + backward) -> Concatenate -> FC -> Output

    The bidirectional encoder processes the sequence in both directions,
    capturing context from past and future for better predictions.
    """
    def __init__(self, input_size, hidden_size, output_size, dt=0.1, dropout=0.1):
        super(BidirectionalEncoderLiquidNetworkModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dt = dt

        # Input projection
        self.input_layer = nn.Linear(input_size, hidden_size)

        # Forward and backward liquid layers
        self.liquid_forward = LiquidODECell(hidden_size, hidden_size, dt=dt)
        self.liquid_backward = LiquidODECell(hidden_size, hidden_size, dt=dt)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Output layer (takes concatenated forward + backward)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size, seq_len, _ = x.size()

        # Project input
        x = self.input_layer(x)  # (batch_size, seq_len, hidden_size)

        # Forward pass
        h_forward = torch.zeros(batch_size, self.hidden_size).to(x.device)
        for t in range(seq_len):
            h_forward = self.liquid_forward(x[:, t, :], h_forward)

        # Backward pass
        h_backward = torch.zeros(batch_size, self.hidden_size).to(x.device)
        for t in range(seq_len - 1, -1, -1):
            h_backward = self.liquid_backward(x[:, t, :], h_backward)

        # Concatenate forward and backward representations
        h_combined = torch.cat([h_forward, h_backward], dim=1)  # (batch_size, hidden_size * 2)
        h_combined = self.dropout(h_combined)

        # Final prediction
        output = self.fc(h_combined)  # (batch_size, output_size)

        return output


class HybridTransformerLNNModel(nn.Module):
    """
    Hybrid Transformer + LNN model for NILM.

    Inspired by:
      Gabriel et al. (2025) "Hybrid transformer model with liquid neural networks
      and learnable encodings for buildings' energy forecasting", Energy and AI.
      DOI: 10.1016/j.egyai.2025.100489

    Architecture:
      1. Learnable encoding  — 2-layer MLP with LayerNorm (non-linear input embedding)
      2. CNN encoder         — local feature extraction over the time axis
      3. Transformer encoder — global self-attention over the CNN features
      4. LNN reservoir       — LiquidODECell processes the contextualised sequence
      5. FC output           — maps final LNN hidden state to prediction
    """

    def __init__(self, input_size, hidden_size, output_size, dt=0.1,
                 num_conv_layers=2, num_encoder_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        # 1. Learnable encodings
        self.learnable_encoding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # 2. CNN encoder (operates on time axis)
        cnn_layers = []
        for _ in range(num_conv_layers):
            cnn_layers.append(nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
        self.cnn_encoder = nn.ModuleList(cnn_layers)

        # 3. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers)

        # 4. LNN reservoir
        self.liquid = LiquidODECell(hidden_size, hidden_size, dt=dt)

        # 5. Output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 1. Learnable encodings (applied pointwise across time)
        x = self.learnable_encoding(x)          # (B, seq_len, H)

        # 2. CNN encoder
        x = x.transpose(1, 2)                   # (B, H, seq_len)
        for layer in self.cnn_encoder:
            x = layer(x)
        x = x.transpose(1, 2)                   # (B, seq_len, H)

        # 3. Transformer encoder
        x = self.transformer_encoder(x)          # (B, seq_len, H)

        # 4. LNN reservoir
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(seq_len):
            h = self.liquid(x[:, t, :], h)

        # 5. Output
        return self.fc(h)


# ── Hybrid Transformer-LNN v2 (paper-faithful) helper classes ─────────────────

class LiquidReservoir(nn.Module):
    """
    Spectral-radius controlled reservoir layer (Gabriel et al., 2025).
    Processes Q or K in attention with controlled non-linearity:
        output = tanh(x @ W_reservoir)
    W is initialised so its spectral radius equals `spectral_radius`.
    """
    def __init__(self, size, spectral_radius=0.9):
        super().__init__()
        W = torch.randn(size, size)
        max_eig = torch.linalg.eigvals(W).abs().max()
        W = W / max_eig * spectral_radius
        self.W = nn.Parameter(W)

    def forward(self, x):
        return torch.tanh(x @ self.W)


class AdaptivePositionalEncoding(nn.Module):
    """
    Adaptive positional encoding (Gabriel et al., 2025):
        PE_adaptive = alpha * PE_sinusoidal + beta
    where alpha and beta are learnable scalars.
    """
    def __init__(self, hidden_size, max_len=512):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta  = nn.Parameter(torch.zeros(1))

        pe = torch.zeros(max_len, hidden_size)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, hidden_size, 2).float() *
            -(math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.alpha * self.pe[:seq_len] + self.beta


class LiquidTransformerLayer(nn.Module):
    """
    Transformer encoder layer with two paper-faithful innovations
    (Gabriel et al., 2025):
      - LNN reservoir applied to Q and K before scaled dot-product attention
      - GLU feedforward network (instead of standard ReLU/GELU FFN)
    """
    def __init__(self, hidden_size, num_heads, dropout=0.1, spectral_radius=0.9):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.scale     = self.head_dim ** -0.5

        # Attention projections
        self.q_proj   = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj   = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj   = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # LNN reservoirs for Q and K
        self.liquid_q = LiquidReservoir(hidden_size, spectral_radius)
        self.liquid_k = LiquidReservoir(hidden_size, spectral_radius)

        # GLU feedforward: H → 4H → GLU → 2H → H
        self.ff_gate = nn.Linear(hidden_size, hidden_size * 4)
        self.ff_out  = nn.Linear(hidden_size * 2, hidden_size)

        self.norm1   = nn.LayerNorm(hidden_size)
        self.norm2   = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, H = x.size()

        # ── LNN-enhanced multi-head self-attention ──
        residual = x
        x = self.norm1(x)

        Q = self.liquid_q(self.q_proj(x))   # (B, L, H) — tanh(QW_res)
        K = self.liquid_k(self.k_proj(x))   # (B, L, H) — tanh(KW_res)
        V = self.v_proj(x)                  # (B, L, H)

        def split_heads(t):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        scores  = (Q @ K.transpose(-2, -1)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attn    = (weights @ V).transpose(1, 2).reshape(B, L, H)
        x = residual + self.dropout(self.out_proj(attn))

        # ── GLU feedforward ──
        residual = x
        x = self.norm2(x)
        gate = self.ff_gate(x)                   # (B, L, 4H)
        a, b = gate.chunk(2, dim=-1)             # each (B, L, 2H)
        x = residual + self.dropout(self.ff_out(a * torch.sigmoid(b)))

        return x


class HybridTransformerLNNv2Model(nn.Module):
    """
    Hybrid Transformer + LNN v2 — paper-faithful implementation.

    Gabriel et al. (2025) "Hybrid transformer model with liquid neural networks
    and learnable encodings for buildings' energy forecasting", Energy and AI.
    DOI: 10.1016/j.egyai.2025.100489

    Key differences from HybridTransformerLNNModel (v1):
      - LNN reservoir is embedded INSIDE attention (modifies Q and K),
        not applied sequentially after the Transformer
      - Adaptive positional encoding: PE = α·PE_sinusoidal + β (learnable α, β)
      - CNN encoder uses ReLU + MaxPool1d (per paper) instead of GELU
      - GLU feedforward in each Transformer layer instead of standard FFN
      - Final prediction uses last-token output (no LiquidODECell at end)

    Architecture:
      1. Input embedding    — 2-layer MLP with LayerNorm
      2. CNN encoder        — Conv1d + BN + ReLU + MaxPool1d
      3. Adaptive PE        — sinusoidal PE scaled by learnable α, β
      4. LiquidTransformer  — self-attention with LNN reservoirs on Q,K + GLU FFN
      5. FC output          — last-token representation → prediction
    """

    def __init__(self, input_size, hidden_size, output_size, dt=0.1,
                 num_conv_layers=2, num_encoder_layers=2, num_heads=4,
                 dropout=0.1, spectral_radius=0.9):
        super().__init__()
        self.hidden_size = hidden_size

        # 1. Input embedding (dense learnable encodings)
        self.input_embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # 2. CNN encoder with MaxPool (preserves sequence length)
        cnn_layers = []
        for _ in range(num_conv_layers):
            cnn_layers.append(nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                nn.Dropout(dropout),
            ))
        self.cnn_encoder = nn.ModuleList(cnn_layers)

        # 3. Adaptive positional encoding
        self.pos_encoding = AdaptivePositionalEncoding(hidden_size)

        # 4. LNN-integrated Transformer layers
        self.transformer_layers = nn.ModuleList([
            LiquidTransformerLayer(hidden_size, num_heads, dropout, spectral_radius)
            for _ in range(num_encoder_layers)
        ])

        # 5. Output (last token)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 1. Input embedding
        x = self.input_embedding(x)       # (B, seq_len, H)

        # 2. CNN encoder
        x = x.transpose(1, 2)            # (B, H, seq_len)
        for layer in self.cnn_encoder:
            x = layer(x)
        x = x.transpose(1, 2)            # (B, seq_len, H)

        # 3. Adaptive positional encoding
        x = self.pos_encoding(x)

        # 4. LNN-integrated Transformer
        for layer in self.transformer_layers:
            x = layer(x)

        # 5. Last token → prediction
        return self.fc(self.norm(x[:, -1, :]))


# ── Hybrid Transformer-LNN v5 ─────────────────────────────────────────────────

class HybridTransformerLNNv5Model(nn.Module):
    """
    Hybrid Transformer-LNN v5 for NILM.

    Builds on HybridTransformerLNNModel (v1) with two additions inspired by
    CNNEncoderLiquidNetworkModel and BidirectionalEncoderLiquidNetworkModel:

      1. Graduated CNN channels  — hidden//4 → hidden//2 → hidden with dropout,
                                   instead of flat hidden → hidden throughout.
      2. Bidirectional LNN       — forward + backward LiquidODECell after the
                                   Transformer encoder; outputs are concatenated.
         FC input = hidden * 2.

    Architecture:
      1. Learnable encoding   — 2-layer MLP with LayerNorm
      2. Graduated CNN        — channels grow: H//4 → H//2 → H (+ proj to H)
      3. Transformer encoder  — global self-attention
      4. Bidirectional LNN    — forward pass + backward pass over the sequence
      5. Dropout + FC         — concat(h_fwd, h_bwd) → dropout → output
    """

    def __init__(self, input_size, hidden_size, output_size, dt=0.1,
                 num_conv_layers=3, num_encoder_layers=2, num_heads=4,
                 dropout=0.1, pool='mean'):
        super().__init__()
        self.hidden_size = hidden_size
        self.pool = pool  # 'mean' or 'max'

        # 1. Learnable encodings (same as v1)
        self.learnable_encoding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # 2. Graduated CNN encoder: channels grow H//4 → H//2 → H
        channel_schedule = []
        for i in range(num_conv_layers):
            out_ch = hidden_size // (2 ** (num_conv_layers - 1 - i))
            out_ch = max(out_ch, 1)
            channel_schedule.append(out_ch)
        # channel_schedule for hidden=256, layers=3: [64, 128, 256]

        cnn_layers = []
        in_ch = hidden_size  # after learnable_encoding projects to hidden_size
        for out_ch in channel_schedule:
            cnn_layers.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
            in_ch = out_ch
        self.cnn_encoder = nn.ModuleList(cnn_layers)

        # Project final conv output back to hidden_size (in case last != hidden)
        self.cnn_proj = nn.Linear(in_ch, hidden_size) if in_ch != hidden_size \
            else nn.Identity()

        # 3. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers)

        # 4. Bidirectional LNN: separate cells for forward and backward
        self.liquid_fwd = LiquidODECell(hidden_size, hidden_size, dt=dt)
        self.liquid_bwd = LiquidODECell(hidden_size, hidden_size, dt=dt)

        # 5. Output: concat(h_fwd, h_bwd) → dropout → FC
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 1. Learnable encodings
        x = self.learnable_encoding(x)          # (B, seq_len, H)

        # 2. Graduated CNN encoder
        x = x.transpose(1, 2)                   # (B, H, seq_len)
        for layer in self.cnn_encoder:
            x = layer(x)
        x = x.transpose(1, 2)                   # (B, seq_len, last_ch)
        x = self.cnn_proj(x)                    # (B, seq_len, H)

        # 3. Transformer encoder
        x = self.transformer_encoder(x)          # (B, seq_len, H)

        # 4. Bidirectional LNN — mean-pool over all hidden states
        h_fwd = torch.zeros(batch_size, self.hidden_size, device=x.device)
        h_fwd_states = []
        for t in range(seq_len):
            h_fwd = self.liquid_fwd(x[:, t, :], h_fwd)
            h_fwd_states.append(h_fwd)
        h_fwd_stack = torch.stack(h_fwd_states, dim=1)              # (B, seq, H)
        h_fwd_pool = (h_fwd_stack.max(dim=1).values if self.pool == 'max'
                      else h_fwd_stack.mean(dim=1))                 # (B, H)

        h_bwd = torch.zeros(batch_size, self.hidden_size, device=x.device)
        h_bwd_states = []
        for t in range(seq_len - 1, -1, -1):
            h_bwd = self.liquid_bwd(x[:, t, :], h_bwd)
            h_bwd_states.append(h_bwd)
        h_bwd_stack = torch.stack(h_bwd_states, dim=1)              # (B, seq, H)
        h_bwd_pool = (h_bwd_stack.max(dim=1).values if self.pool == 'max'
                      else h_bwd_stack.mean(dim=1))                 # (B, H)

        # 5. Concat → dropout → FC
        h = torch.cat([h_fwd_pool, h_bwd_pool], dim=1)  # (B, H*2)
        return self.fc(self.dropout(h))


# ── LNN-First Transformer (physics frontend → global context) ─────────────────

class LNNFirstTransformerModel(nn.Module):
    """
    LNN-First Hybrid for NILM.

    The LNN acts as a physics-aware frontend: it unrolls a LiquidODECell
    over every raw input timestep, learning per-channel time constants (tau)
    that capture transient dynamics before any attention is applied.
    The resulting sequence of hidden states is then processed by a standard
    Transformer Encoder for long-range context modelling.

    Architecture:
      Input proj  : Linear(input_size → hidden)
      LNN frontend: LiquidODECell unrolled over seq_len  → (B, T, hidden)
      LNN dropout : Dropout(dropout)
      Transformer : num_encoder_layers × TransformerEncoderLayer(hidden, nheads)
      Pool        : mean (default) or max over T
      FC          : Linear(hidden → output_size)
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 256,
                 output_size: int = 1, dt: float = 0.1,
                 num_encoder_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.1, pool: str = 'mean'):
        super().__init__()
        self.hidden_size = hidden_size
        self.pool = pool

        # Project raw signal to hidden dim before LNN
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        # LNN physics frontend
        self.lnn_cell = LiquidODECell(hidden_size, hidden_size, dt=dt)
        self.lnn_drop = nn.Dropout(dropout)

        # Transformer encoder for global context
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(enc_layer,
                                                  num_layers=num_encoder_layers)

        self.out_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, input_size)
        B, T, _ = x.size()

        # Project to hidden dim
        x = self.input_proj(x)                          # (B, T, H)

        # LNN frontend — unroll cell over time
        h = torch.zeros(B, self.hidden_size, device=x.device)
        states = []
        for t in range(T):
            h = self.lnn_cell(x[:, t, :], h)
            states.append(h)
        lnn_out = self.lnn_drop(
            torch.stack(states, dim=1)                  # (B, T, H)
        )

        # Transformer encoder
        ctx = self.transformer(lnn_out)                 # (B, T, H)

        # Pool over sequence
        if self.pool == 'max':
            pooled = ctx.max(dim=1).values
        else:
            pooled = ctx.mean(dim=1)                    # (B, H)

        return self.fc(self.out_drop(pooled))           # (B, output_size)


# ── LNN-First Transformer with Sliding Window (Local) Attention ───────────────

class LNNLocalTransformerModel(nn.Module):
    """
    LNN-First Hybrid for NILM with banded (local) self-attention.

    Identical to LNNFirstTransformerModel but each Transformer position t
    can only attend to the window [t−attn_window, t+attn_window].  The
    LNN recurrent state still carries global history; the Transformer is
    restricted to local context so transient ON/OFF edges cannot bleed
    across distant timesteps.

    Architecture:
      Input proj  : Linear(input_size → hidden)
      LNN frontend: LiquidODECell unrolled over seq_len  → (B, T, hidden)
      LNN dropout : Dropout(dropout)
      Transformer : num_encoder_layers × TransformerEncoderLayer  (local mask)
      Pool        : mean (default) or max over T
      FC          : Linear(hidden → output_size)
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 256,
                 output_size: int = 1, dt: float = 0.1,
                 num_encoder_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.1, pool: str = 'mean',
                 seq_len: int = 100, attn_window: int = 20):
        super().__init__()
        self.hidden_size = hidden_size
        self.pool = pool

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        self.lnn_cell = LiquidODECell(hidden_size, hidden_size, dt=dt)
        self.lnn_drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(enc_layer,
                                                  num_layers=num_encoder_layers)

        self.out_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

        # Pre-compute banded mask: positions outside [t-w, t+w] are masked
        if attn_window > 0:
            mask = torch.full((seq_len, seq_len), float('-inf'))
            for i in range(seq_len):
                s = max(0, i - attn_window)
                e = min(seq_len, i + attn_window + 1)
                mask[i, s:e] = 0.0
            self.register_buffer('attn_mask', mask)
        else:
            self.attn_mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()

        x = self.input_proj(x)                          # (B, T, H)

        h = torch.zeros(B, self.hidden_size, device=x.device)
        states = []
        for t in range(T):
            h = self.lnn_cell(x[:, t, :], h)
            states.append(h)
        lnn_out = self.lnn_drop(
            torch.stack(states, dim=1)                  # (B, T, H)
        )

        ctx = self.transformer(lnn_out, mask=self.attn_mask)   # (B, T, H)

        if self.pool == 'max':
            pooled = ctx.max(dim=1).values
        else:
            pooled = ctx.mean(dim=1)                    # (B, H)

        return self.fc(self.out_drop(pooled))           # (B, output_size)


# ── Selective SSM Cell (S6-inspired) ──────────────────────────────────────────

class SelectiveSSMCell(nn.Module):
    """
    Selective State Space cell (S6-inspired), sequential/recurrent version.
    Drop-in replacement for LiquidODECell.

    Captures the core "selectivity" of Mamba (Gu & Dao, 2023) in pure PyTorch
    without requiring CUDA kernels — suitable for seq_len ≤ ~200.

    For each timestep t:
        dt    = Softplus(W_Δ · x_t)         input-dependent step size (≥ 0)
        B     = sigmoid(W_B · x_t)           write gate — what to store
        C     = sigmoid(W_C · x_t)           read  gate — what to emit
        Ā     = exp(dt ⊙ A),  A = −exp(A_log)  discretised decay ∈ (0, 1)
        h_new = Ā ⊙ h  +  (1 − Ā) ⊙ B ⊙ W_in · x_t
        y     = C ⊙ h_new

    Interpretation:
      • Large dt (input spike)  → Ā ≈ 0  → new input dominates state
      • Small dt (quiet period) → Ā ≈ 1  → old state is preserved
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Fixed diagonal A, log-parameterised (always negative → stable decay)
        # Init: A_i = -(i+1), so faster neurons have larger decay rates
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, hidden_size + 1, dtype=torch.float32))
        )

        # Project input to hidden space (shared basis for state update)
        self.in_proj = nn.Linear(input_size, hidden_size, bias=False)

        # Selective parameters — all functions of the current input x_t
        self.dt_proj = nn.Linear(input_size, hidden_size)           # step size
        self.B_proj  = nn.Linear(input_size, hidden_size, bias=False)  # write
        self.C_proj  = nn.Linear(input_size, hidden_size, bias=False)  # read

    def forward(self, x, h):
        """
        Args:
            x : (batch, input_size)  — current timestep input
            h : (batch, hidden_size) — previous hidden state
        Returns:
            y : (batch, hidden_size) — new state (also used as output)
        """
        x_proj = self.in_proj(x)                            # (B, H)
        dt     = F.softplus(self.dt_proj(x))                # (B, H), ≥ 0
        B      = torch.sigmoid(self.B_proj(x))              # (B, H), write gate
        C      = torch.sigmoid(self.C_proj(x))              # (B, H), read  gate

        # Discretise A: Ā = exp(Δ · A),  A = −exp(A_log) < 0
        A     = -torch.exp(self.A_log).unsqueeze(0)         # (1, H)
        A_bar = torch.exp(dt * A)                           # (B, H) ∈ (0, 1)

        # State update (ZOH-inspired):
        #   h_t = Ā ⊙ h_{t-1} + (1 − Ā) ⊙ B ⊙ x_proj
        h_new = A_bar * h + (1.0 - A_bar) * B * x_proj     # (B, H)

        # Selective read from state
        y = C * h_new                                       # (B, H)
        return y


# ── Hybrid Transformer-SSM (v6) ───────────────────────────────────────────────

class HybridTransformerSSMModel(nn.Module):
    """
    Hybrid Transformer + Selective SSM model for NILM (v6).

    Identical architecture to HybridTransformerLNNv5Model but replaces the
    LiquidODECell (Euler-integrated ODE) with SelectiveSSMCell (S6-inspired):

      • B, C, Δ all input-dependent → model selectively updates / ignores input
      • Closed-form discretisation  → no Euler error, more stable gradients
      • Bidirectional (fwd + bwd)   → full sequence context before prediction
      • Max-pool or mean-pool over all hidden states (max for bursty appliances)

    Architecture:
      Learnable Encoding → Graduated CNN (H//4 → H//2 → H)
      → Transformer Encoder → BiDir SelectiveSSM → Dropout → FC(H*2 → 1)
    """

    def __init__(self, input_size, hidden_size, output_size,
                 num_conv_layers=3, num_encoder_layers=2, num_heads=4,
                 dropout=0.1, pool='mean'):
        super().__init__()
        self.hidden_size = hidden_size
        self.pool = pool  # 'mean' or 'max'

        # 1. Learnable encoding
        self.learnable_encoding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # 2. Graduated CNN: channels grow H//4 → H//2 → H
        channel_schedule = [
            max(hidden_size // (2 ** (num_conv_layers - 1 - i)), 1)
            for i in range(num_conv_layers)
        ]
        cnn_layers, in_ch = [], hidden_size
        for out_ch in channel_schedule:
            cnn_layers.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
            in_ch = out_ch
        self.cnn_encoder = nn.ModuleList(cnn_layers)
        self.cnn_proj = (nn.Linear(in_ch, hidden_size)
                         if in_ch != hidden_size else nn.Identity())

        # 3. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 4, dropout=dropout,
            batch_first=True, activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers)

        # 4. Bidirectional Selective SSM
        self.ssm_fwd = SelectiveSSMCell(hidden_size, hidden_size)
        self.ssm_bwd = SelectiveSSMCell(hidden_size, hidden_size)

        # 5. Output: concat(fwd, bwd) → dropout → FC
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 1. Learnable encoding
        x = self.learnable_encoding(x)            # (B, seq, H)

        # 2. Graduated CNN
        x = x.transpose(1, 2)                     # (B, H, seq)
        for layer in self.cnn_encoder:
            x = layer(x)
        x = x.transpose(1, 2)                     # (B, seq, last_ch)
        x = self.cnn_proj(x)                      # (B, seq, H)

        # 3. Transformer encoder
        x = self.transformer_encoder(x)            # (B, seq, H)

        # 4. Bidirectional Selective SSM — pool over all hidden states
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        fwd_states = []
        for t in range(seq_len):
            h = self.ssm_fwd(x[:, t, :], h)
            fwd_states.append(h)
        fwd_stack = torch.stack(fwd_states, dim=1)           # (B, seq, H)
        fwd_pool  = (fwd_stack.max(dim=1).values if self.pool == 'max'
                     else fwd_stack.mean(dim=1))             # (B, H)

        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        bwd_states = []
        for t in range(seq_len - 1, -1, -1):
            h = self.ssm_bwd(x[:, t, :], h)
            bwd_states.append(h)
        bwd_stack = torch.stack(bwd_states, dim=1)           # (B, seq, H)
        bwd_pool  = (bwd_stack.max(dim=1).values if self.pool == 'max'
                     else bwd_stack.mean(dim=1))             # (B, H)

        # 5. Concat → dropout → FC
        h_cat = torch.cat([fwd_pool, bwd_pool], dim=1)       # (B, H*2)
        return self.fc(self.dropout(h_cat))


# ---------------------------------------------------------------------------
# GNN + LNN
# ---------------------------------------------------------------------------

class GCNLayer(nn.Module):
    """
    Single Graph Convolutional Network layer.

    Computes:  out = ReLU( A_hat * X * W )
    where A_hat is the pre-normalised adjacency matrix.

    Args:
        in_features:  input feature dimension per node
        out_features: output feature dimension per node
    """
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        Args:
            x:   (batch, num_nodes, in_features)
            adj: (num_nodes, num_nodes)  pre-normalised, on same device as x

        Returns:
            (batch, num_nodes, out_features)
        """
        adj_exp = adj.unsqueeze(0).expand(x.size(0), -1, -1)   # (B, N, N)
        agg = torch.bmm(adj_exp, x)                             # (B, N, in_features)
        return F.relu(self.linear(agg))                         # (B, N, out_features)


class GNNLiquidNetworkModel(nn.Module):
    """
    Graph Neural Network + Liquid Neural Network for multi-appliance NILM.

    Architecture:
        1. TCN encoder  : aggregate window → per-timestep features (B, T, tcn_out)
        2. Node init    : broadcast aggregate feature + learnable node embeddings
                          → (B, N, tcn_out)  where N = num_appliances
        3. GCN layers   : relational message passing across appliance nodes
                          → (B, N, hidden)
        4. LNN cell     : Liquid ODE step per node, using GCN output as input
        5. Output FC    : hidden state of each node → appliance power
                          → (B, N)  (all appliances predicted simultaneously)

    The adjacency matrix encodes co-activation correlations between appliances
    and is computed from training data (passed as adj_matrix at construction).

    Args:
        input_size:    feature dim of each input timestep (1 for scalar power)
        hidden_size:   hidden dimension for GCN and LNN
        num_nodes:     number of appliance nodes (default 4)
        num_channels:  TCN channel list, e.g. [32, 64, 128]
        kernel_size:   TCN kernel size
        dropout:       TCN dropout
        dt:            LNN Euler integration step
        num_gcn_layers: number of stacked GCN layers
        adj_matrix:    (num_nodes, num_nodes) numpy array — normalised adjacency;
                       if None, defaults to a uniform fully-connected graph
    """
    def __init__(self, input_size, hidden_size, num_nodes=4,
                 num_channels=None, kernel_size=3, dropout=0.2,
                 dt=0.1, num_gcn_layers=2, adj_matrix=None):
        super(GNNLiquidNetworkModel, self).__init__()

        if num_channels is None:
            num_channels = [32, 64, 128]

        self.num_nodes   = num_nodes
        self.hidden_size = hidden_size
        self.dt          = dt
        tcn_out          = num_channels[-1]

        # --- TCN encoder (keep sequence dimension) ---
        tcn_layers = []
        for i in range(len(num_channels)):
            dilation  = 2 ** i
            in_ch     = input_size if i == 0 else num_channels[i - 1]
            out_ch    = num_channels[i]
            tcn_layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*tcn_layers)

        # --- Learnable node embeddings (one per appliance) ---
        self.node_embed = nn.Parameter(torch.randn(num_nodes, tcn_out) * 0.01)

        # --- GCN layers ---
        gcn_in = tcn_out
        self.gcn_layers = nn.ModuleList()
        self.gcn_norms  = nn.ModuleList()
        for i in range(num_gcn_layers):
            self.gcn_layers.append(GCNLayer(gcn_in, hidden_size))
            self.gcn_norms.append(nn.LayerNorm(hidden_size))
            gcn_in = hidden_size

        # --- Shared LiquidODECell across nodes ---
        self.lnn_cell  = LiquidODECell(hidden_size, hidden_size, dt)
        self.lnn_norm  = nn.LayerNorm(hidden_size)

        # --- Output: one scalar per node ---
        self.fc = nn.Linear(hidden_size, 1)

        # --- Adjacency matrix (fixed, registered as buffer) ---
        if adj_matrix is not None:
            adj_tensor = torch.FloatTensor(adj_matrix)
        else:
            # Uniform fully-connected (including self-loops), row-normalised
            adj_tensor = torch.ones(num_nodes, num_nodes) / num_nodes
        self.register_buffer('adj', adj_tensor)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 1)  — normalised aggregate power window

        Returns:
            out: (batch, num_nodes) — predicted power for each appliance
        """
        batch_size, seq_len, _ = x.size()

        # 1. TCN encode — keep sequence
        x_tcn = x.permute(0, 2, 1)          # (B, 1, T)
        x_tcn = self.tcn(x_tcn)             # (B, tcn_out, T)
        x_tcn = x_tcn.permute(0, 2, 1)      # (B, T, tcn_out)

        # 2. Initialise per-node hidden states
        hidden = [torch.zeros(batch_size, self.hidden_size, device=x.device)
                  for _ in range(self.num_nodes)]

        # 3. Step through time
        for t in range(seq_len):
            x_t = x_tcn[:, t, :]            # (B, tcn_out)

            # Build node features: broadcast aggregate + node embedding
            node_feats = (x_t.unsqueeze(1)                     # (B, 1, tcn_out)
                          + self.node_embed.unsqueeze(0))       # (B, N, tcn_out)

            # GCN message passing
            for gcn, norm in zip(self.gcn_layers, self.gcn_norms):
                node_feats = norm(gcn(node_feats, self.adj))    # (B, N, hidden)

            # LNN update per node (shared cell weights)
            for n in range(self.num_nodes):
                h_prev = self.lnn_norm(hidden[n])               # normalise hidden
                hidden[n] = self.lnn_cell(node_feats[:, n, :], h_prev)

        # 4. Stack final hidden states and project to output
        hidden_stack = torch.stack(hidden, dim=1)               # (B, N, hidden)
        out = self.fc(hidden_stack).squeeze(-1)                 # (B, N)

        return out


class GRUGNNLiquidNetworkModel(nn.Module):
    """
    Gated fusion of GRU-LNN (temporal) and GNN-LNN (relational) for multi-appliance NILM.

    Architecture:
        GRU branch  : BiGRU → Linear projection → LiquidODECell (step-by-step)
                      → h_t ∈ (B, hidden)  [global temporal embedding]
                      → node_gru_proj: Linear(hidden → hidden*N)
                      → h_t[n] ∈ (B, hidden) per node  [per-node temporal view]

        GNN branch  : TCN encoder → per-node init → GCN layers → LiquidODECell per node
                      → r_t[n] ∈ (B, hidden) [per-appliance relational embedding]

        Gate (shared weights, operates per node):
            g_t[n] = σ( W_g · [h_t[n] ; r_t[n]] + b_g )
            W_g ∈ ℝ^{hidden × 2·hidden},  g_t[n] ∈ (0,1)^hidden

        Convex mix:
            z_t[n] = g_t[n] ⊙ h_t[n]  +  (1 − g_t[n]) ⊙ r_t[n]

        Decoder:
            out[n] = FC( z_t[n] )  →  appliance power scalar

    The per-node projection node_gru_proj gives the gate a distinct temporal
    representation per appliance, allowing it to learn which branch to trust
    differently for each load type (e.g. g→1 for fast spikes, g→0 for
    appliances with strong co-activation patterns).

    The gate is interpretable: after training, its mean value per appliance
    indicates whether the model relied more on temporal (g→1) or relational (g→0)
    context for each load type.

    Args:
        input_size:     feature dim per timestep (1 for scalar power)
        hidden_size:    hidden dim shared by both branches and the gate
        num_nodes:      number of appliances (default 4)
        gru_hidden:     hidden size per GRU direction (BiGRU output = gru_hidden*2)
        num_gru_layers: stacked GRU layers
        num_channels:   TCN channel list for GNN branch, e.g. [32, 64, 128]
        kernel_size:    TCN kernel size
        dropout:        dropout (GRU inter-layer + TCN)
        dt:             LNN Euler integration step
        num_gcn_layers: stacked GCN layers
        adj_matrix:     (num_nodes, num_nodes) numpy array — normalised adjacency;
                        defaults to uniform fully-connected if None
    """
    def __init__(self, input_size, hidden_size, num_nodes=4,
                 gru_hidden=64, num_gru_layers=2,
                 num_channels=None, kernel_size=3, dropout=0.2,
                 dt=0.1, num_gcn_layers=2, adj_matrix=None):
        super(GRUGNNLiquidNetworkModel, self).__init__()

        if num_channels is None:
            num_channels = [32, 64, 128]

        self.num_nodes   = num_nodes
        self.hidden_size = hidden_size
        self.dt          = dt
        tcn_out          = num_channels[-1]

        # ── GRU branch ──────────────────────────────────────────────────────
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=gru_hidden,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_gru_layers > 1 else 0.0
        )
        self.gru_proj      = nn.Linear(gru_hidden * 2, hidden_size)
        self.gru_lnn       = LiquidODECell(hidden_size, hidden_size, dt=dt)
        # Project shared temporal embedding to N distinct per-node views
        self.node_gru_proj = nn.Linear(hidden_size, hidden_size * num_nodes)

        # ── GNN branch ──────────────────────────────────────────────────────
        tcn_layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch    = input_size if i == 0 else num_channels[i - 1]
            out_ch   = num_channels[i]
            tcn_layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*tcn_layers)

        self.node_embed  = nn.Parameter(torch.randn(num_nodes, tcn_out) * 0.01)

        gcn_in = tcn_out
        self.gcn_layers = nn.ModuleList()
        self.gcn_norms  = nn.ModuleList()
        for i in range(num_gcn_layers):
            self.gcn_layers.append(GCNLayer(gcn_in, hidden_size))
            self.gcn_norms.append(nn.LayerNorm(hidden_size))
            gcn_in = hidden_size

        self.gnn_lnn      = LiquidODECell(hidden_size, hidden_size, dt=dt)
        self.gnn_lnn_norm = nn.LayerNorm(hidden_size)

        # ── Gate (shared across nodes) ───────────────────────────────────────
        # Input: [h_t ; r_t[n]]  shape (B, 2*hidden)  →  g_t[n]  shape (B, hidden)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)

        # ── Decoder ─────────────────────────────────────────────────────────
        self.fc = nn.Linear(hidden_size, 1)

        # ── Adjacency (fixed buffer) ─────────────────────────────────────────
        if adj_matrix is not None:
            adj_tensor = torch.FloatTensor(adj_matrix)
        else:
            adj_tensor = torch.ones(num_nodes, num_nodes) / num_nodes
        self.register_buffer('adj', adj_tensor)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 1) — normalised aggregate power window

        Returns:
            out:  (batch, num_nodes) — predicted power for each appliance
            gates: (batch, num_nodes, hidden_size) — gate values for interpretability
        """
        batch_size, seq_len, _ = x.size()

        # ── GRU branch: encode full sequence ────────────────────────────────
        gru_out, _ = self.gru(x)                        # (B, T, gru_hidden*2)
        gru_proj   = self.gru_proj(gru_out)             # (B, T, hidden)

        h_gru = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(seq_len):
            h_gru = self.gru_lnn(gru_proj[:, t, :], h_gru)
        # h_gru: (B, hidden) — global temporal embedding
        # Project to per-node temporal views so the gate can differentiate
        h_nodes = self.node_gru_proj(h_gru).view(
            batch_size, self.num_nodes, self.hidden_size)  # (B, N, hidden)

        # ── GNN branch: TCN encode then step through time ───────────────────
        x_tcn = x.permute(0, 2, 1)                     # (B, 1, T)
        x_tcn = self.tcn(x_tcn)                        # (B, tcn_out, T)
        x_tcn = x_tcn.permute(0, 2, 1)                 # (B, T, tcn_out)

        hidden_gnn = [torch.zeros(batch_size, self.hidden_size, device=x.device)
                      for _ in range(self.num_nodes)]

        for t in range(seq_len):
            x_t = x_tcn[:, t, :]                       # (B, tcn_out)

            node_feats = (x_t.unsqueeze(1)             # (B, 1, tcn_out)
                          + self.node_embed.unsqueeze(0))  # (B, N, tcn_out)

            for gcn, norm in zip(self.gcn_layers, self.gcn_norms):
                node_feats = norm(gcn(node_feats, self.adj))   # (B, N, hidden)

            for n in range(self.num_nodes):
                h_prev = self.gnn_lnn_norm(hidden_gnn[n])
                hidden_gnn[n] = self.gnn_lnn(node_feats[:, n, :], h_prev)
        # hidden_gnn[n]: (B, hidden) — relational embedding per appliance

        # ── Gated fusion ─────────────────────────────────────────────────────
        gate_vals = []
        fused     = []
        for n in range(self.num_nodes):
            h_n  = h_nodes[:, n, :]                         # (B, hidden) per-node temporal
            r_n  = hidden_gnn[n]                            # (B, hidden) per-node relational
            gate = torch.sigmoid(
                self.gate(torch.cat([h_n, r_n], dim=-1))    # (B, 2*hidden) → (B, hidden)
            )
            z_n  = gate * h_n + (1.0 - gate) * r_n         # convex mix
            gate_vals.append(gate)
            fused.append(z_n)

        fused_stack = torch.stack(fused, dim=1)             # (B, N, hidden)
        gate_stack  = torch.stack(gate_vals, dim=1)         # (B, N, hidden)

        out = self.fc(fused_stack).squeeze(-1)              # (B, N)

        return out, gate_stack
