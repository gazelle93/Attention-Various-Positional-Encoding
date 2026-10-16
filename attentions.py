import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encoders import AbsolutePositionalEncoder, RelativePositionalEncoder, T5RelativePositionalEncoder


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
            attn_score = attn_score.masked_fill(mask, torch.finfo(attn_score.dtype).min)

        # Softmax of the scaled score (alpha)
        attn_score = F.softmax(attn_score, -1)

        # Matrix multiplication of the scaled score and value (z)
        output = torch.bmm(attn_score, value)

        return output, attn_score


# Scaled Dot Product Attention using Relative Positional Encoding
class RelativeScaledDotProductAttention(nn.Module):
    def __init__(self, emb_dim):
        super(RelativeScaledDotProductAttention, self).__init__()

        # scaling factor 1 / sqrt(dimension of queries and keys)
        self.scaling_factor = torch.sqrt(torch.tensor(emb_dim))


    def forward(self, query, key, value, a_key, a_value, mask = None):

        # Scaled score of the Matrix multiplication of query and key (e)
        qk_attn = torch.bmm(query, key.transpose(1, 2))
        relative_qk_attn = torch.bmm(query.permute(1, 0, 2).contiguous(), a_key.transpose(1, 2)).transpose(0, 1)
        attn_score = (qk_attn + relative_qk_attn) / self.scaling_factor

        # Masking (Optional)
        # shape of mask: (batch size, input length of query, input length of key)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask, torch.finfo(attn_score.dtype).min)

        # Softmax of the scaled score (alpha)
        attn_score = F.softmax(attn_score, -1)

        # Matrix multiplication of the scaled score and value (z)
        qkv_attn = torch.bmm(attn_score, value)
        relative_qkv_attn = torch.bmm(attn_score.permute(1, 0, 2).contiguous(), a_value).transpose(0, 1)

        output = qkv_attn + relative_qkv_attn

        return output, attn_score



# Scaled Dot Product Attention using T5 Relative Positional Encoding
class T5ScaledDotProductAttention(nn.Module):
    def __init__(self, emb_dim):
        super(T5ScaledDotProductAttention, self).__init__()

        # scaling factor 1 / sqrt(dimension of queries and keys)
        self.scaling_factor = torch.sqrt(torch.tensor(emb_dim))


    def forward(self, query, key, value, relative_bias, mask = None):
        # Scaled score of the Matrix multiplication of query and key (e)
        # relative_bias arrives as (seq_len_q, seq_len_k, num_heads) and is
        # shared across the batch; expand it to match (batch * heads, Lq, Lk)
        bias = relative_bias.permute(2, 0, 1)
        bias = bias.repeat(query.size(0) // bias.size(0), 1, 1)
        attn_score = torch.bmm(query, key.transpose(1, 2)) / self.scaling_factor + bias

        # Masking (Optional)
        # shape of mask: (batch size, input length of query, input length of key)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask, torch.finfo(attn_score.dtype).min)

        # Softmax of the scaled score (alpha)
        attn_score = F.softmax(attn_score, -1)

        output = torch.bmm(attn_score, value)

        return output, attn_score




# Multi-Head Attention using Relation Positional Encoding
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, positional_encoding="abs", dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()

        self.head_dim = int(emb_dim / num_heads)
        self.num_heads = num_heads
        self.positional_encoding = positional_encoding
        self.dropout = nn.Dropout(p=dropout_rate)

        # initialize one feed-forward layer (head dimension x number of heads) of each q, k and v
        # instead of initializing number of heads of feed-forward layers (head dimension / number of heads)
        self.query_proj = nn.Linear(emb_dim, self.head_dim * num_heads)
        self.key_proj = nn.Linear(emb_dim, self.head_dim * num_heads)
        self.value_proj = nn.Linear(emb_dim, self.head_dim * num_heads)
        self.out_proj = nn.Linear(emb_dim, self.head_dim * num_heads)

        if positional_encoding == "abs":
            self.scaled_dot_attn = ScaledDotProductAttention(self.head_dim)

        elif positional_encoding == "rel":
            self.relative_scaled_dot_attn = RelativeScaledDotProductAttention(self.head_dim)
            self.relative_position_k = RelativePositionalEncoder(self.head_dim)
            self.relative_position_v = RelativePositionalEncoder(self.head_dim)

        elif positional_encoding == "t5":
            self.t5_scaled_dot_attn = T5ScaledDotProductAttention(self.head_dim)


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


    def forward(self, query, key, value, mask = None, relative_bias=None, is_dropout=True):
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


        # the attention tensors are (batch * heads, Lq, Lk), so the mask has to
        # be flattened the same way rather than left as (batch, heads, Lq, Lk)
        if mask is not None:
            if mask.dim() == 3:          # (batch, Lq, Lk), shared across heads
                mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            mask = mask.reshape(batch_size * self.num_heads, mask.size(-2), mask.size(-1))

        if self.positional_encoding == "abs":
            output, attn_score = self.scaled_dot_attn(query, key, value, mask)

        elif self.positional_encoding == "rel":
            seq_len_query = query.size()[1]
            seq_len_key = key.size()[1]
            seq_len_value = value.size()[1]
            a_key = self.relative_position_k(seq_len_query, seq_len_key)
            a_value = self.relative_position_v(seq_len_query, seq_len_value)
            output, attn_score = self.relative_scaled_dot_attn(query, key, value, a_key, a_value, mask)

        elif self.positional_encoding == "t5":
            seq_len_query = query.size()[1]
            seq_len_key = key.size()[1]
            output, attn_score = self.t5_scaled_dot_attn(query, key, value, relative_bias, mask)


        # reshape the result of the ScaledDotProductAttention
        # reshape_to_ScaledDotProductAttention packed this batch-major, so the
        # flat index is (batch * num_heads + head). Unpacking it as
        # (num_heads, batch, ...) mixes heads across batch rows and silently
        # returns the wrong answer for any batch_size > 1.
        # shape: (batch size, number of heads, input length, head dimension)
        output = output.view(batch_size, self.num_heads, -1, self.head_dim)

        # reshape to concat
        # shape: (number of heads, batch size, input length, head dimension)
        output = self.reshape_to_concat(batch_size, output)

        # final feed-forward network
        output = self.out_proj(output)
        
        if is_dropout:
            output = self.dropout(output)

        return output, attn_score

    
    
    
def get_candidate_heads(emb_dim, num_heads):
    """Nearest head count that divides emb_dim.

    Returns num_heads untouched when it already divides emb_dim. Otherwise it
    picks the closest divisor and says so, because silently changing a model's
    head count is expensive to debug.
    """
    num_heads = int(num_heads)
    if emb_dim % num_heads == 0:
        return num_heads

    divisors = [i for i in range(1, emb_dim + 1) if emb_dim % i == 0]
    # ties go to the larger divisor, so you never get less parallelism than asked
    nearest = min(divisors, key=lambda d: (abs(d - num_heads), -d))
    warnings.warn(
        f"num_heads={num_heads} does not divide emb_dim={emb_dim}; "
        f"using num_heads={nearest} instead.", stacklevel=2)
    return nearest
    
    
    
def get_attn_output(input_embedding, selected_attn, selected_pe, _num_heads):
    emb_dim = input_embedding.size()[-1]

    # Absolute encoding is added at the input, and only when it is the scheme
    # being used. Shaw et al. and Raffel et al. both replace it rather than
    # stack on top of it, so adding it unconditionally would make every "rel"
    # or "t5" run a mixture and the comparison meaningless.
    if selected_pe == "abs":
        positional_encoder = AbsolutePositionalEncoder(emb_dim)
        input_embedding = input_embedding + positional_encoder(input_embedding)

    query = key = value = input_embedding
    seq_len_query = query.size()[1]
    seq_len_key = key.size()[1]
    seq_len_value = value.size()[1]

    if selected_attn not in ("scaleddotproduct", "multihead"):
        raise ValueError(
            f"selected_attn must be 'scaleddotproduct' or 'multihead', "
            f"got {selected_attn!r}")
    if selected_pe not in ("abs", "rel", "t5"):
        raise ValueError(
            f"selected_pe must be 'abs', 'rel' or 't5', got {selected_pe!r}")

    if selected_attn == "multihead":
        num_heads = get_candidate_heads(emb_dim, _num_heads)
        # selected_pe has to reach the module. Leaving it off meant every
        # multi-head run used the default "abs" branch, so "rel" and "t5"
        # produced bit-identical output and the comparison was a no-op.
        model = MultiHeadAttention(emb_dim, num_heads, selected_pe)

        relative_bias = None
        if selected_pe == "t5":
            # built with num_heads, not the requested count: the two differ
            # whenever the request does not divide emb_dim
            relative_bias = T5RelativePositionalEncoder(num_heads)(
                seq_len_query, seq_len_key)

        return model(query, key, value, relative_bias=relative_bias)

    # single-head scaled dot product
    if selected_pe == "abs":
        model = ScaledDotProductAttention(emb_dim)
        return model(query, key, value)

    if selected_pe == "rel":
        a_key = RelativePositionalEncoder(emb_dim)(seq_len_query, seq_len_key)
        a_value = RelativePositionalEncoder(emb_dim)(seq_len_query, seq_len_value)
        model = RelativeScaledDotProductAttention(emb_dim)
        return model(query, key, value, a_key, a_value)

    relative_bias = T5RelativePositionalEncoder(1)(seq_len_query, seq_len_key)
    model = T5ScaledDotProductAttention(emb_dim)
    return model(query, key, value, relative_bias)
