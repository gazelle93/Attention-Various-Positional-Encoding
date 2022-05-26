import torch
import argparse
from text_processing import get_nlp_pipeline, word_tokenization
import my_onehot
from attentions import AbsolutePositionalEncoder, RelativePositionalEncoder, T5RelativePositionalEncoder
from attentions import ScaledDotProductAttention, RelativeScaledDotProductAttention, T5ScaledDotProductAttention
from attentions import MultiHeadAttention

def main(args):
    text_list = ["We are about to study the idea of a computational process.", 
             "Computational processes are abstract beings that inhabit computers.",
            "As they evolve, processes manipulate other abstract things called data.",
            "The evolution of a process is directed by a pattern of rules called a program.",
            "People create programs to direct processes.",
            "In effect, we conjure the spirits of the computer with our spells."]
    
    cur_text = "People create a computational process."

    
    # One-hot Encoding
    embeddings = my_onehot.get_onehot_encoding(text_list, cur_text, args.nlp_pipeline, args.unk_ignore)


    batch_size = 1
    input_length, emb_dim = embeddings.size()

    input_embedding = embeddings.reshape(batch_size, input_length, emb_dim)

    # input embedding + positional encoding
    positional_encoder = AbsolutePositionalEncoder(emb_dim)
    input_embedding = input_embedding + positional_encoder(input_embedding)
    query = key = value = input_embedding

    seq_len_query = query.size()[1]
    seq_len_key = key.size()[1]

    # Absolute Positional Encoding
    if args.positional_encoding == "abs":
        if args.attention == "scaleddotproduct":
            model = ScaledDotProductAttention(emb_dim)
            output, attn_score = model(query, key, value)

            print("Scaled-Dot-Product Attention Result")
            print(output)

        elif args.attention == "multihead":
            if emb_dim % args.num_heads != 0:
                divisor_list = []
                for i in range(1, emb_dim):
                    if emb_dim % i == 0:
                        divisor_list.append(i)
                num_heads = divisor_list[len(divisor_list)//2]
            else:
                num_heads = args.num_heads

            model = MultiHeadAttention(emb_dim, num_heads)
            output, attn_score = model(query, key, value)

            print("Multi-Head Attention Result (Number of heads: {})".format(num_heads))
            print(output)


    # Relative Positional Encoding
    elif args.positional_encoding == "rel":
        if args.attention == "scaleddotproduct":
            relative_position_k = RelativePositionalEncoder(emb_dim)
            relative_position_v = RelativePositionalEncoder(emb_dim)

            seq_len_query = query.size()[1]
            seq_len_key = key.size()[1]
            seq_len_value = value.size()[1]

            a_key = relative_position_k(seq_len_query, seq_len_key)
            a_value = relative_position_v(seq_len_query, seq_len_value)

            model = RelativeScaledDotProductAttention(emb_dim)

            output, attn_score = model(query, key, value, a_key, a_value)

            print("Relative Scaled-Dot-Product Attention Result")
            print(output)


        elif args.attention == "multihead":
            if emb_dim % args.num_heads != 0:
                divisor_list = []
                for i in range(1, emb_dim):
                    if emb_dim % i == 0:
                        divisor_list.append(i)
                num_heads = divisor_list[len(divisor_list)//2]
            else:
                num_heads = args.num_heads

            model = MultiHeadAttention(emb_dim, num_heads)
            output, attn_score = model(query, key, value)

            print("Relative Multi-Head Attention Result (Number of heads: {})".format(num_heads))
            print(output)


    # T5 Relative Positional Encoding
    elif args.positional_encoding == "t5":
        if args.attention == "scaleddotproduct":
            relative_position_bias = T5RelativePositionalEncoder(1)

            seq_len_query = query.size()[1]
            seq_len_key = key.size()[1]

            relative_bias = relative_position_bias(seq_len_query, seq_len_key)
            print(relative_bias.size())
            print("-----")

            model = T5ScaledDotProductAttention(emb_dim)

            output, attn_score = model(query, key, value, relative_bias)

            print("Relative Scaled-Dot-Product Attention Result")
            print(output)


        elif args.attention == "multihead":
            if emb_dim % args.num_heads != 0:
                divisor_list = []
                for i in range(1, emb_dim):
                    if emb_dim % i == 0:
                        divisor_list.append(i)
                num_heads = divisor_list[len(divisor_list)//2]
            else:
                num_heads = args.num_heads

            model = MultiHeadAttention(emb_dim, num_heads)
            output, attn_score = model(query, key, value)

            print("Relative Multi-Head Attention Result (Number of heads: {})".format(num_heads))
            print(output)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--nlp_pipeline", default="spacy", type=str, help="NLP preprocessing pipeline.")
    parser.add_argument("--unk_ignore", default=True, help="Ignore unknown tokens.")
    parser.add_argument("--num_heads", default=8, help="The number of heads for multi-head attention.")
    parser.add_argument("--attention", default="scaleddotproduct", type=str, help="Type of attention layer. (scaleddotproduct, multihead)")
    parser.add_argument("--positional_encoding", default="abs", type=str, help="Type of positional encoding. (abs, rel, t5)")
    args = parser.parse_args()

    main(args)
