import torch
import argparse
from text_processing import get_nlp_pipeline, word_tokenization
import my_onehot
from attentions import get_attn_output

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
    
    
    output, attn_score = get_attn_output(input_embedding, args.attention, args.positional_encoding, args.num_heads)
    print("Selected Attention: {}".format(args.attention))
    print("Selected Positional Encoding: {}".format(args.positional_encoding))
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
