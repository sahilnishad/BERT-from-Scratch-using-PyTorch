from bert_model import BERT, BERTEmbedding
from utils import build_vocab, get_embedding_tensor
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_embeddings(embeddings, sentence):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings[0].detach().numpy())
    plt.scatter(reduced[:, 0], reduced[:, 1])

    for (x, y), word in zip(reduced, sentence.split()):
        plt.text(x, y, word)

    plt.show()

if __name__ == "__main__":
    # Build vocabulary
    example_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning provides computers the ability to learn from data.",
        "Climate change is a pressing issue for the planet.",
        "Traveling around the world offers diverse cultural experiences.",
        "Books are gateways to vast worlds of knowledge."
    ]
    vocab = build_vocab(example_sentences)

    # Configuration
    VOCAB_SIZE = len(vocab)
    N_SEGMENTS = 2
    MAX_LEN = 512
    EMBED_DIM = 768
    N_LAYERS = 12
    ATTN_HEADS = 12
    DROPOUT = 0.1

    # Get embeddings
    embedding_layer = BERTEmbedding(VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBED_DIM, DROPOUT)
    user_sentence = "The fox travels around the world."
    embedding_tensor = get_embedding_tensor(user_sentence, vocab, embedding_layer)

    # Visualization of embeddings
    plot_embeddings(embedding_tensor, user_sentence)

    #BERT Model Architecture visualize
    model = BERT(VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBED_DIM, N_LAYERS, ATTN_HEADS, DROPOUT)
    print(model)



