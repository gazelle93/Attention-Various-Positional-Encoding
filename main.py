import torch
import argparse
from text_processing import get_nlp_pipeline, word_tokenization
import my_onehot
from attentions import ScaledDotProductAttention, MultiHeadAttention

def main(args):
    text_list = ["We are about to study the idea of a computational process.", 
             "Computational processes are abstract beings that inhabit computers.",
            "As they evolve, processes manipulate other abstract things called data.",
            "The evolution of a process is directed by a pattern of rules called a program.",
            "People create programs to direct processes.",
            "In effect, we conjure the spirits of the computer with our spells."]
    
    cur_text = "People create a computational process."

    
    # One-hot Encoding
    sklearn_onehotencoder = my_onehot.build_onehot_encoding_model(args.unk_ignore)
    token2idx_dict, _ = my_onehot.init_token2idx(text_list, args.nlp_pipeline)
    sklearn_onehotencoder.fit([[t] for t in token2idx_dict])
    tks = my_onehot.get_tokens(cur_text, args.nlp_pipeline)

    embeddings = my_onehot.onehot_encoding(sklearn_onehotencoder, tks)
    print("Raw Sklearn One-hot Encoding Result")
    print(embeddings)
    
    batch_size = 1
    input_length, emb_dim = embeddings.size()
    query = key = value = embeddings.reshape(batch_size, input_length, emb_dim)

    print(query.type())
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
      
    else:
        print("Not seleceted.")
    
    

          
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--nlp_pipeline", default="spacy", type=str, help="NLP preprocessing pipeline.")
    parser.add_argument("--unk_ignore", default=True, help="Ignore unknown tokens.")
    parser.add_argument("--num_heads", default=8, help="The number of heads for multi-head attention.")
    parser.add_argument("--attention", default="multihead", type=str, help="Type of attention layer. (scaleddotproduct, multihead)")
    args = parser.parse_args()

    main(args)
