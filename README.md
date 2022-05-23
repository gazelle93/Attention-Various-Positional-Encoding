# Overview
- After the emergence of Attention, the language models leveraging the attention layer show the best performance in various NLP tasks. Attention allows attending to utilize the most relevant parts of the input sequence by leveraging the attention score which is a weighted result of all of the encoded input vectors simultaneously. Therefore, attention layers are able to increase the learning speed through parallelization without the restrictions appearing in such sequential architectures. This project aims to implement the Scaled-Dot-Product Attention layer and the Multi-Head Attention layer using Absolute Positional Encoding.

$$\alpha_{ij} = \frac{1}{\sqrt{d}}((w_i+p_i)W^{Q,1})(w_j+p_j)W^{K,1})^T$$

# Brief description
- text_processing.py
> Output format
> - output: Tokenized result of a given text. (list)
- my_onehot.py
> Output format
> - output: List of tensor of input tokens. (Tensor)
- attentions.py
> Output format
> - output: List of tensor of attention results. (Tensor)


# Prerequisites
- argparse
- torch
- stanza
- spacy
- nltk
- gensim

# Parameters
- nlp_pipeline(str, defaults to "stanza"): NLP preprocessing pipeline.
- unk_ignore(bool, defaults to True): Ignore unseen word or not.
- num_heads(int, defaults to 8): The number of heads for multi-head attention.
- attention(str, defaults to "multihead"): Type of attention layer. (scaleddotproduct, multihead)

# References
- Attention: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.
- Stanza: Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020). Stanza: A Python natural language processing toolkit for many human languages. arXiv preprint arXiv:2003.07082.
- Spacy: Matthew Honnibal and Ines Montani. 2017. spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing. To appear (2017).
- NLTK: Bird, Steven, Edward Loper and Ewan Klein (2009). Natural Language Processing with Python. O'Reilly Media Inc.
- Gensim: Rehurek, R., & Sojka, P. (2010). Software framework for topic modelling with large corpora. In In Proceedings of the LREC 2010 workshop on new challenges for NLP frameworks.
