import torch
import torch.nn as nn

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, n_segments, max_len, embed_dim, dropout):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.segment_embed = nn.Embedding(n_segments, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.pos_inp = torch.tensor([i for i in range(max_len)],)

    def forward(self, seq, seg):
        current_max_len = seq.size(1)  # Get current sequence length
        pos_inp = torch.arange(0, current_max_len, device=seq.device).unsqueeze(0)  # Dynamically create position tensor based on input size
        embed_val = self.token_embed(seq) + self.segment_embed(seg) + self.pos_embed(pos_inp)
        embed_val = self.drop(embed_val)
        return embed_val

class BERT(nn.Module):
    def __init__(self, vocab_size, n_segments, max_len, embed_dim, n_layers, attn_heads, dropout):
        super().__init__()
        self.embedding = BERTEmbedding(vocab_size, n_segments, max_len, embed_dim, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(embed_dim, attn_heads, embed_dim*4)
        self.encoder_block = nn.TransformerEncoder(self.encoder_layer, n_layers)

    def forward(self, seq, seg):
        out = self.embedding(seq, seg)
        out = self.encoder_block(out)
        return out
