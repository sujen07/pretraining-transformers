import torch
import math


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.o_proj = torch.nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor = None, mask: torch.Tensor = None):
        batch_size, seq_len, d_model = x.shape
        head_dim = d_model // self.num_heads

        q = self.q_proj(x)
        if encoder_output is not None:
            k = self.k_proj(encoder_output)
            v = self.v_proj(encoder_output)
        else:
            k = self.k_proj(x)
            v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)

        # is_causal already applies the autoregressive mask; attn_mask can't be set with it.
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.o_proj(attn_output)
    

class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, 4 * d_model),
            torch.nn.GELU(),
            torch.nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        attn_output = self.self_attn(self.norm1(x), mask=mask)
        x = x + self.dropout(attn_output)
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)
        return x

class Transformer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, num_vocab: int, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_vocab = num_vocab
        self.max_seq_len = max_seq_len
        self.decoder = torch.nn.ModuleList([DecoderLayer(d_model, num_heads) for _ in range(num_layers)])
        self.embedding = torch.nn.Embedding(num_vocab, d_model)
        self.pos_embedding = torch.nn.Embedding(max_seq_len, d_model)
        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.linear = torch.nn.Linear(d_model, num_vocab)
        self.linear.weight = self.embedding.weight

    def forward(self, x: torch.Tensor):
        mask = self._create_mask(x)
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)
        for layer in self.decoder:
            x = layer(x, mask=mask)
        
        x = self.layer_norm(x)
        logits = self.linear(x)
        return logits

    def _create_mask(self, x: torch.Tensor):
        batch_size, seq_len = x.shape
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        return mask