import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import streamlit as st

# Preprocessing setup
clean_spcl = re.compile(r'[/(){}\[\]\|@,;]')
clean_symbol = re.compile(r'[^0-9a-z #+_]')

# Create Sastrawi instances
factory = StopWordRemoverFactory()
stopword_list = factory.get_stop_words()
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = clean_spcl.sub(' ', text)
    text = clean_symbol.sub('', text)
    text = ' '.join(word for word in text.split() if word not in stopword_list)  # Remove stopwords
    text = stemmer.stem(text)  # Stemming
    return text

@st.cache_data
def load_and_preprocess_data(file_path):
    # Load dataset
    amazon_df = pd.read_excel(file_path)

    # Ensure required columns exist
    if 'product_name' not in amazon_df.columns:
        st.error("Kolom 'product_name' tidak ditemukan di dataset.")
        st.stop()

    # Add preprocessed column if not exists
    if 'judul_prosessing' not in amazon_df.columns:
        amazon_df['judul_prosessing'] = amazon_df['product_name'].apply(clean_text)

    amazon_df.reset_index(inplace=True, drop=True)
    return amazon_df

@st.cache_data
def compute_tfidf_matrix(data):
    # Initialize TF-IDF Vectorizer
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=1)
    tfidf_matrix = tf.fit_transform(data)
    return tf, tfidf_matrix

def recommendations(query, tf, tfidf_matrix, amazon_df, top=10):
    query_cleaned = clean_text(query)
    query_vec = tf.transform([query_cleaned])  # Transform query into TF-IDF vector
    query_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()  # Similarity with all products

    # Get top indices with highest similarity
    top_indices = query_sim.argsort()[-top:][::-1]  # Sort from highest to lowest similarity

    recommended_products = amazon_df.iloc[top_indices]
    results = recommended_products[['judul_prosessing', 'product_name']].reset_index(drop=True)

    # If no matches are found
    if results.empty:
        return [f"Tidak ada produk amazon yang cocok dengan kata kunci '{query}'"]

    # Return results as a list of strings
    return results.apply(
        lambda row: f"{row['product_name']} (Preprocessed: {row['judul_prosessing']})", axis=1
    ).tolist()

# Streamlit app
st.title("Sistem Rekomendasi Produk Teknologi Amazon")

# Load and preprocess data
file_path = "amazon.xlsx"
amazon_df = load_and_preprocess_data(file_path)

# Compute TF-IDF matrix
tf, tfidf_matrix = compute_tfidf_matrix(amazon_df['judul_prosessing'])

query_input = st.text_input("Masukkan kata atau kalimat pencarian:")
num_recommendations = st.slider("Jumlah rekomendasi amazon", min_value=1, max_value=30, value=5)

if st.button("Cari Rekomendasi"):
    if query_input:
        with st.spinner("Mencari rekomendasi..."):
            hasil_rekomendasi = recommendations(query_input, tf, tfidf_matrix, amazon_df, top=num_recommendations)
            st.write("Rekomendasi amazon untuk Anda:")
            if isinstance(hasil_rekomendasi, list):  # If results are found
                for idx, rekomendasi in enumerate(hasil_rekomendasi, start=1):
                    st.write(f"{idx}. {rekomendasi}")

# Display dataset for reference
st.write("Dataset amazon:")
st.dataframe(amazon_df[['judul_prosessing', 'product_name']])
