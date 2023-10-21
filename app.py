import streamlit as st
from bert_model import BERTEmbedding
from utils import build_vocab, get_embedding_tensor
from sklearn.decomposition import PCA
import plotly.express as px


def plot_2d_embeddings(embeddings, sentence):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings[0].detach().numpy())
    fig = px.scatter(
        x=reduced[:, 0], 
        y=reduced[:, 1], 
        text=sentence.split()
    )
    st.plotly_chart(fig)

def plot_3d_embeddings(embeddings, sentence):
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(embeddings[0].detach().numpy())
    fig = px.scatter_3d(
        x=reduced[:, 0], 
        y=reduced[:, 1], 
        z=reduced[:, 2], 
        text=sentence.split()
    )
    st.plotly_chart(fig)

# Configuration
N_SEGMENTS = 2
MAX_LEN = 512
EMBED_DIM = 768
N_LAYERS = 12
ATTN_HEADS = 12
DROPOUT = 0.1


def main():
    st.title("BERT Embeddings Visualization")

    uploaded_file = st.file_uploader("Upload a text file to build vocabulary", type=['txt'])

    if uploaded_file:
        uploaded_data = uploaded_file.read().decode('utf-8').splitlines()
        st.success("Vocabulary built successfully!")
    else:
        st.warning("Using default vocabulary.")
        with open('data/default_vocab.txt', 'r') as file:
            uploaded_data = file.read().splitlines()

    vocab = build_vocab(uploaded_data)

    VOCAB_SIZE = len(vocab)
    embedding_layer = BERTEmbedding(VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBED_DIM, DROPOUT)
    
    user_sentence = st.text_input("Enter your sentence:", "AI in healthcare predicts patient outcomes and diagnoses.")
    
    viz_option = st.selectbox("Select Visualization Type", ["2D", "3D"])
    
    if st.button('Visualize Embeddings'):
        embedding_tensor = get_embedding_tensor(user_sentence, vocab, embedding_layer)
        
        if viz_option == "2D":
            plot_2d_embeddings(embedding_tensor, user_sentence)
        elif viz_option == "3D":
            plot_3d_embeddings(embedding_tensor, user_sentence)

if __name__ == "__main__":
    main()
