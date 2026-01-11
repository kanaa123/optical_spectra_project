import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------
# Transformer model 
# -------------------------------------------------
class SpectralTransformer(nn.Module):
    """
    Transformer-based model for optical filter classification
    using 1D absorbance spectra.
    """

    def __init__(
        self,
        input_length=401,
        num_classes=4,
        d_model=64,
        nhead=4,
        num_layers=4,
        dim_feedforward=128,
        dropout=0.1,
    ):
        super().__init__()

        # ----------------------------------
        # Input projection
        # ----------------------------------
        # Projects scalar absorbance → embedding
        self.input_proj = nn.Linear(1, d_model)

        # ----------------------------------
        # Positional encoding (learnable)
        # ----------------------------------
        self.pos_embedding = nn.Parameter(
            torch.randn(1, input_length, d_model)
        )

        # ----------------------------------
        # Transformer encoder
        # ----------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # ----------------------------------
        # Classification head
        # ----------------------------------
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape: (batch_size, 401)

        Returns
        -------
        torch.Tensor
            Shape: (batch_size, num_classes)
        """

        # Add channel dimension: (B, 401, 1)
        x = x.unsqueeze(-1)

        # Project to embedding space
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_embedding

        # Transformer encoder
        x = self.encoder(x)

        # Global average pooling over wavelength dimension
        x = x.mean(dim=1)

        # Classification
        logits = self.classifier(x)

        return logits

# -------------------------------------------------
# Quick test 
# -------------------------------------------------
if __name__ == "__main__":
    model = SpectralTransformer()
    dummy_input = torch.randn(8, 401)  # batch of 8 spectra
    output = model(dummy_input)

    print("Output shape:", output.shape)
    # Expected: (8, 4)

# -------------------------------------------------
# Baseline model
# -------------------------------------------------

class SpectralCNN(nn.Module):
    """
    Baseline 1D CNN for optical filter classification
    using absorption spectra.
    """

    def __init__(self, input_length=401, num_classes=4):
        super().__init__()

        # ----------------------------------
        # Convolutional feature extractor
        # ----------------------------------
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=7,
            padding=3
        )

        self.conv2 = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            padding=2
        )

        self.conv3 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool1d(kernel_size=2)

        # ----------------------------------
        # Compute flattened size automatically
        # ----------------------------------
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            dummy = self._forward_features(dummy)
            self.flatten_dim = dummy.shape[1]

        # ----------------------------------
        # Classification head
        # ----------------------------------
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def _forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x.view(x.size(0), -1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape: (batch_size, 401)

        Returns
        -------
        torch.Tensor
            Shape: (batch_size, num_classes)
        """

        # Add channel dimension: (B, 1, 401)
        x = x.unsqueeze(1)

        # Feature extraction
        x = self._forward_features(x)

        # Classification
        logits = self.fc(x)

        return logits


# -------------------------------------------------
# Quick test
# -------------------------------------------------
if __name__ == "__main__":
    model = SpectralCNN()
    dummy_input = torch.randn(8, 401)
    output = model(dummy_input)

    print("Output shape:", output.shape)
    # Expected: torch.Size([8, 4])
