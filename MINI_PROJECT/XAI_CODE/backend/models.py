# models.py
import torch.nn as nn

class ProgressionLSTM(nn.Module):
    def __init__(self, input_size=384, hidden_size=128, num_layers=2, output_size=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        last_hidden = hidden[-1]
        return self.fc(last_hidden)