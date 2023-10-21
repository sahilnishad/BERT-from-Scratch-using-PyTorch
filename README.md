#  BERT from Scratch with PyTorch for PCA Embedding Visualization

This project provides an implementation of the BERT model, as described in the paper "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", using PyTorch. In addition to replicating the model's foundational architecture, the project also features utilities for visualizing the embeddings, serving as a practical tool for those interested in understanding the inner workings of BERT.

#
### BERT Architecture
![bert architecture](https://github.com/sahilnishad/BERT-from-Scratch-using-PyTorch/blob/main/images/bert%20architecture.PNG)

The image depicting the BERT model architecture showcases its components:

* The embedding section consists of token, segment, and positional embeddings, followed by a dropout layer.
* The encoder_layer includes the primary elements of the Transformer encoder, encapsulating the multi-head attention mechanism and an array of linear transformations.
* The encoder_block stacks these encoder layers multiple times, with 12 repetitions, reflecting the typical structure of the BERT model.


#
### Embedding Visualization
![embedding](https://github.com/sahilnishad/BERT-from-Scratch-using-PyTorch/blob/main/images/embedding.png)

The 2D visualization derived from the BERT model word embeddings is a result of applying Principal Component Analysis (PCA). This technique condenses the high-dimensional embeddings into a two-dimensional space, revealing the spatial associations among the words. This representation offers a perspective on how the model discerns semantic relationships between terms, indicating contextual relationships in processed sentences.

Explaination behind the plot for the example sentence of "The fox travels around the world":
* Words like "The" and "the" (in different cases) are relatively close to each other, indicating that the model recognizes them as semantically similar despite the difference in case.
* The word "fox" stands apart, suggesting its unique semantic meaning compared to other words in the sentence.
* "world." and "travels" are positioned on the farther right. This could indicate a shared semantic space or similar context, as they are both related to movement or space.
* The word "around" is placed distinctively at the top, possibly indicating its role as a preposition, setting the context for movement or direction.
