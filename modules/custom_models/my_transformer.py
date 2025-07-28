import torch
import torch.nn as nn

class MyTransformerNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        # 位置编码（最大序列长度100）
        self.pos_emb = nn.Parameter(torch.randn(1, 100, input_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=4, dim_feedforward=hidden_size, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        # x: (batch, seq, feat)
        seq_len = x.size(1)
        x = x + self.pos_emb[:, :seq_len, :]
        out = self.transformer(x)
        out = out[:, -1, :]  # 取最后一个时间步
        out = self.fc(out)
        return out 