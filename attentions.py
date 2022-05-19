import torch
import torch.nn as nn
import torch.nn.functional as F

# Scaled Dot Product Attention using Absolute Positional Encoding
class ScaledDotProductAttention(nn.Module):
    def __init__(self, emb_dim):
        super(ScaledDotProductAttention, self).__init__()
        
        # scaling factor 1 / sqrt(dimension of queries and keys)
        self.scaling_factor = torch.sqrt(torch.tensor(emb_dim))
        
        
    def forward(self, query, key, value, mask = None):
        # Scaled score of the Matrix multiplication of query and key (e)
        attn_score = torch.bmm(query, key.transpose(1, 2)) / self.scaling_factor

        # Masking (Optional)
        # shape of mask: (batch size, input length of query, input length of key)
        if mask is not None:
            attn_score.masked_fill_(mask, -1e18)

        # Softmax of the scaled score (alpha)
        attn_score = F.softmax(attn_score, -1)
        
        # Matrix multiplication of the scaled score and value (z)
        output = torch.bmm(attn_score, value)
        
        return output, attn_score


class AbsolutePositionalEncoder(nn.Module):
    def __init__(self, emb_dim, max_position=512):
        super(AbsolutePositionalEncoder, self).__init__()
        self.position = torch.arange(max_position).unsqueeze(1)

        self.positional_encoding = torch.zeros(1, max_position, emb_dim)

        _2i = torch.arange(0, emb_dim, step=2).float()

        # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        self.positional_encoding[0, :, 0::2] = torch.sin(self.position / (10000 ** (_2i / emb_dim)))

        # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        self.positional_encoding[0, :, 1::2] = torch.cos(self.position / (10000 ** (_2i / emb_dim)))

    def forward(self, x):
        # batch_size, input_len, embedding_dim
        batch_size, seq_len, _ = x.size()

        return self.positional_encoding[:batch_size, :seq_len, :]
      
      
      
class MultiHeadAttention(nn.Module):

    def __init__(self, emb_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        self.head_dim = int(emb_dim / num_heads)
        self.num_heads = num_heads
        self.scaled_dot_attn = ScaledDotProductAttention(self.head_dim)
        
        # initialize one feed-forward layer (head dimension x number of heads) of each q, k and v
        # instead of initializing number of heads of feed-forward layers (head dimension / number of heads)
        self.query_proj = nn.Linear(emb_dim, self.head_dim * num_heads)
        self.key_proj = nn.Linear(emb_dim, self.head_dim * num_heads)
        self.value_proj = nn.Linear(emb_dim, self.head_dim * num_heads)
        self.out_proj = nn.Linear(emb_dim, self.head_dim * num_heads)
        
        nn.init.xavier_uniform_(self.query_proj)
        nn.init.xavier_uniform_(self.key_proj)
        nn.init.xavier_uniform_(self.value_proj)
        nn.init.xavier_uniform_(self.out_proj)
        
        
    def reshape_from_feed_forward(self, batch_size, _tensor):
        return _tensor.view(batch_size, -1, self.num_heads, self.head_dim)
    
    
    def reshape_to_ScaledDotProductAttention(self, batch_size, _tensor):
        # before shape: (batch size, input length, number of heads, head dimension)
        # after shape: (batch size, number of heads, input length, head dimension)
        _tensor = _tensor.permute(0, 2, 1, 3)
        
        # reshape to feed the tensor to ScaledDotProductAttention
        return _tensor.contiguous().view(batch_size * self.num_heads, -1, self.head_dim)
    
    
    def reshape_to_concat(self, batch_size, _tensor):
        # before shape: (batch size, number of heads, input length, head dimension)
        # after shape: (batch size, input length, number of heads, head dimension)
        _tensor = _tensor.permute(0, 2, 1, 3)
        return _tensor.contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

    
    def forward(self, query, key, value, mask = None):
        # shape of input of q, k and v: (batch size, input length, embedding dimension)
        batch_size = query.size()[0]
        
        # feed-forward network
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)
        
        # reshape the result of the feed-forward network
        # shape after the feed-forward network of q, k and v: (batch, input length, number of heads, head dimension)
        query = self.reshape_from_feed_forward(batch_size, query)
        key = self.reshape_from_feed_forward(batch_size, key)
        value = self.reshape_from_feed_forward(batch_size, value)    
        
        # reshape the result of the feed-forward network to feed it to ScaledDotProductAttention
        # shape: (number of heads * batch, input length, head dimension)
        query = self.reshape_to_ScaledDotProductAttention(batch_size, query)
        key = self.reshape_to_ScaledDotProductAttention(batch_size, key)
        value = self.reshape_to_ScaledDotProductAttention(batch_size, value)
        
        
        # shape of mask: (batch size, number of heads, input length of query, input length of key)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        output, attn_score = self.scaled_dot_attn(query, key, value, mask)

        # reshape the result of the ScaledDotProductAttention
        # shape: (number of heads, batch size, input length, head dimension)
        output = output.view(self.num_heads, batch_size, -1, self.head_dim)
        
        # reshape to concat
        # shape: (number of heads, batch size, input length, head dimension)
        output = self.reshape_to_concat(batch_size, output)
        
        # final feed-forward network
        output = self.out_proj(output)

        return output, attn_score
