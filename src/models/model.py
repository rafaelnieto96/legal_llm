import torch
import torch.nn as nn
import math
import xformers.ops as xops

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class XformersMHA(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = dropout

    def forward(self, x, key_padding_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Proyecciones Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim)

        # Atención optimizada de xformers
        attention_output = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=None,
            p=self.dropout if self.training else 0.0,
            key_padding_mask=key_padding_mask
        )
        
        # Reshape y proyección final
        output = attention_output.contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(output)

class LegalLLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.emb_dropout = nn.Dropout(dropout)
        
        # Crear capas del transformer con Xformers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': XformersMHA(d_model, nhead, dropout),
                'norm1': nn.LayerNorm(d_model),
                'feedforward': nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(4 * d_model, d_model)
                ),
                'norm2': nn.LayerNorm(d_model)
            }) for _ in range(num_layers)
        ])
        
        self.d_model = d_model
        self.linear = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, src_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = self.emb_dropout(src)
        
        x = src
        for layer in self.layers:
            # Self-attention
            residual = x
            x = layer['norm1'](x)
            x = layer['attention'](x, key_padding_mask=src_padding_mask)
            x = residual + x
            
            # Feedforward
            residual = x
            x = layer['norm2'](x)
            x = layer['feedforward'](x)
            x = residual + x
        
        output = self.linear(x)
        return output

    def generate(self, start_tokens, max_length, temperature=1.0):
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            curr_tokens = start_tokens.to(device)
            
            for _ in range(max_length - start_tokens.size(1)):
                logits = self(curr_tokens)
                next_token_logits = logits[:, -1, :] / temperature
                
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                curr_tokens = torch.cat([curr_tokens, next_token], dim=1)
            
            return curr_tokens