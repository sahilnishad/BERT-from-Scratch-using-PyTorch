import torch

def build_vocab(sentences):
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for sentence in sentences:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

def sentence_to_token_ids(sentence, vocab):
    return [vocab.get(word, vocab["[UNK]"]) for word in sentence.split()]

def get_embedding_tensor(sentence, vocab, embedding_layer):
    token_ids = sentence_to_token_ids(sentence, vocab)
    seq = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    seg = torch.zeros_like(seq)
    embedding_tensor = embedding_layer(seq, seg)
    return embedding_tensor

def get_input_tensors(sentence, vocab):
    def sentence_to_token_ids(sentence, vocab):
        return [vocab.get(word, vocab["[UNK]"]) for word in sentence.split()]
    token_ids = sentence_to_token_ids(sentence, vocab)
    seq = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    seg = torch.zeros_like(seq)

    return seq, seg
