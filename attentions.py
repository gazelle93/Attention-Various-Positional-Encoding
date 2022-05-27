import torch
import torch.nn as nn
import torch.nn.functional as F


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

# https://github.com/tensorflow/tensor2tensor
class RelativePositionalEncoder(nn.Module):
    def __init__(self, emb_dim, max_position=512):
        super(RelativePositionalEncoder, self).__init__()
        self.max_position = max_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_position * 2 + 1, emb_dim))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, seq_len_q, seq_len_k):
        range_vec_q = torch.arange(seq_len_q)
        range_vec_k = torch.arange(seq_len_k)
        relative_matrix = range_vec_k[None, :] - range_vec_q[:, None]
        clipped_relative_matrix = torch.clamp(relative_matrix, -self.max_position, self.max_position)
        relative_position_matrix = clipped_relative_matrix + self.max_position
        embeddings = self.embeddings_table[relative_position_matrix]

        return embeddings


class T5RelativePositionalEncoder(nn.Module):
    def __init__(self, num_heads, max_position=512):
        super(T5RelativePositionalEncoder, self).__init__()
        self.max_position = max_position
        self.embeddings_table = nn.Embedding(max_position*max_position, num_heads)

    def forward(self, seq_len_q, seq_len_k):
        range_vec_q = torch.arange(seq_len_q)
        range_vec_k = torch.arange(seq_len_k)
        relative_position = range_vec_k[None, :] - range_vec_q[:, None]
        relative_position_clipped = torch.clamp(relative_position, -self.max_position, self.max_position)
        final_mat = relative_position_clipped + self.max_position
        embeddings = self.embeddings_table(final_mat)

        return embeddings



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
            attn_score.masked_fill_(mask, -1e18)

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
        attn_score = torch.bmm(query, key.transpose(1, 2)) / self.scaling_factor + relative_bias.permute(2,0,1)

        # Masking (Optional)
        # shape of mask: (batch size, input length of query, input length of key)
        if mask is not None:
            attn_score.masked_fill_(mask, -1e18)

        # Softmax of the scaled score (alpha)
        attn_score = F.softmax(attn_score, -1)

        output = torch.bmm(attn_score, value)

        return output, attn_score




# Multi-Head Attention using Relation Positional Encoding
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, positional_encoding="abs"):
        super(MultiHeadAttention, self).__init__()

        self.head_dim = int(emb_dim / num_heads)
        self.num_heads = num_heads
        self.positional_encoding = positional_encoding

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
            self.relative_position_bias = T5RelativePositionalEncoder(num_heads)


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
            relative_bias = self.relative_position_bias(seq_len_query, seq_len_key)
            output, attn_score = self.t5_scaled_dot_attn(query, key, value, relative_bias, mask)


        # reshape the result of the ScaledDotProductAttention
        # shape: (number of heads, batch size, input length, head dimension)
        output = output.view(self.num_heads, batch_size, -1, self.head_dim)

        # reshape to concat
        # shape: (number of heads, batch size, input length, head dimension)
        output = self.reshape_to_concat(batch_size, output)

        # final feed-forward network
        output = self.out_proj(output)

        return output, attn_score

    
    
    
def get_attn_output(input_embedding, selected_attn, selected_pe, _num_heads):
    emb_dim = input_embedding.size()[-1]
    
    # input embedding + positional encoding
    positional_encoder = AbsolutePositionalEncoder(emb_dim)
    input_embedding = input_embedding + positional_encoder(input_embedding)
    query = key = value = input_embedding

    seq_len_query = query.size()[1]
    seq_len_key = key.size()[1]

    # Absolute Positional Encoding
    if selected_pe == "abs":
        if selected_attn == "scaleddotproduct":
            model = ScaledDotProductAttention(emb_dim)
            output, attn_score = model(query, key, value)

            return output, attn_score

        elif selected_attn == "multihead":
            if emb_dim % _num_heads != 0:
                divisor_list = []
                for i in range(1, emb_dim):
                    if emb_dim % i == 0:
                        divisor_list.append(i)
                num_heads = divisor_list[len(divisor_list)//2]
            else:
                num_heads = _num_heads

            model = MultiHeadAttention(emb_dim, num_heads)
            output, attn_score = model(query, key, value)

            return output, attn_score


    # Relative Positional Encoding
    elif selected_pe == "rel":
        if selected_attn == "scaleddotproduct":
            relative_position_k = RelativePositionalEncoder(emb_dim)
            relative_position_v = RelativePositionalEncoder(emb_dim)

            seq_len_query = query.size()[1]
            seq_len_key = key.size()[1]
            seq_len_value = value.size()[1]

            a_key = relative_position_k(seq_len_query, seq_len_key)
            a_value = relative_position_v(seq_len_query, seq_len_value)

            model = RelativeScaledDotProductAttention(emb_dim)

            output, attn_score = model(query, key, value, a_key, a_value)

            return output, attn_score


        elif selected_attn == "multihead":
            if emb_dim % _num_heads != 0:
                divisor_list = []
                for i in range(1, emb_dim):
                    if emb_dim % i == 0:
                        divisor_list.append(i)
                num_heads = divisor_list[len(divisor_list)//2]
            else:
                num_heads = _num_heads

            model = MultiHeadAttention(emb_dim, num_heads)
            output, attn_score = model(query, key, value)

            return output, attn_score


    # T5 Relative Positional Encoding
    elif selected_pe == "t5":
        if selected_attn == "scaleddotproduct":
            relative_position_bias = T5RelativePositionalEncoder(1)

            seq_len_query = query.size()[1]
            seq_len_key = key.size()[1]

            relative_bias = relative_position_bias(seq_len_query, seq_len_key)

            model = T5ScaledDotProductAttention(emb_dim)

            output, attn_score = model(query, key, value, relative_bias)

            return output, attn_score


        elif selected_attn == "multihead":
            if emb_dim % _num_heads != 0:
                divisor_list = []
                for i in range(1, emb_dim):
                    if emb_dim % i == 0:
                        divisor_list.append(i)
                num_heads = divisor_list[len(divisor_list)//2]
            else:
                num_heads = _num_heads

            model = MultiHeadAttention(emb_dim, num_heads)
            output, attn_score = model(query, key, value)

            return output, attn_score
