import torch
import torch.nn as nn
from transformers import WhisperModel


class SmartTurnModel(nn.Module):
    def __init__(
        self,
        base_model_name: str = "openai/whisper-tiny",
        freeze_encoder: bool = True,
        unfreeze_last_n: int = 0,
    ):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(base_model_name)
        self.encoder = self.whisper.encoder
        hidden_size = self.whisper.config.d_model

        # Configure encoder for 8-second audio (800 mel frames -> 400 source positions)
        max_positions = 400
        self.encoder.config.max_source_positions = max_positions
        if (
            hasattr(self.encoder, "embed_positions")
            and self.encoder.embed_positions.weight.shape[0] > max_positions
        ):
            self.encoder.embed_positions = nn.Embedding.from_pretrained(
                self.encoder.embed_positions.weight[:max_positions].clone()
            )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

            if unfreeze_last_n > 0:
                num_layers = len(self.encoder.layers)
                for i in range(max(0, num_layers - unfreeze_last_n), num_layers):
                    for param in self.encoder.layers[i].parameters():
                        param.requires_grad = True

        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.Tanh(), nn.Linear(256, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self, input_features: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | None]:
        encoder_outputs = self.encoder(input_features)
        hidden_states = encoder_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # Attention pooling
        attn_weights = self.attention_pool(hidden_states)  # (batch_size, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        pooled_output = torch.sum(hidden_states * attn_weights, dim=1)  # (batch_size, hidden_size)

        logits = self.classifier(pooled_output).squeeze(-1)  # (batch_size,)

        loss = None
        if labels is not None:
            pos_count = (labels == 1).sum()
            neg_count = (labels == 0).sum()
            pos_weight = (
                neg_count / pos_count.clamp(min=1)
                if pos_count > 0
                else torch.tensor(1.0, device=labels.device)
            ).clamp(min=0.1, max=10.0)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = criterion(logits, labels.float())

        return {"loss": loss, "logits": logits, "probabilities": torch.sigmoid(logits)}


class LightweightTurnModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: mel spectrogram (batch_size, 80, seq_len)
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(80, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.gru = nn.GRU(
            input_size=64, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True
        )

        hidden_size = 128  # 64 * 2 (bidirectional)

        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.Tanh(), nn.Linear(256, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )

    def forward(
        self, input_features: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | None]:
        # input_features shape: (batch_size, 80, seq_len)
        x = self.conv_blocks(input_features)  # (batch_size, 64, seq_len')
        x = x.transpose(1, 2)  # (batch_size, seq_len', 64)

        gru_out, _ = self.gru(x)  # (batch_size, seq_len', 128)

        attn_weights = self.attention_pool(gru_out)
        attn_weights = torch.softmax(attn_weights, dim=1)

        pooled_output = torch.sum(gru_out * attn_weights, dim=1)

        logits = self.classifier(pooled_output).squeeze(-1)

        loss = None
        if labels is not None:
            pos_count = (labels == 1).sum()
            neg_count = (labels == 0).sum()
            pos_weight = (
                neg_count / pos_count.clamp(min=1)
                if pos_count > 0
                else torch.tensor(1.0, device=labels.device)
            ).clamp(min=0.1, max=10.0)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = criterion(logits, labels.float())

        return {"loss": loss, "logits": logits, "probabilities": torch.sigmoid(logits)}
