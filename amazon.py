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
factory = StopWordRemoverFactory()
stopword_list = factory.get_stop_words()
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = clean_spcl.sub(' ', text)
    text = clean_symbol.sub('', text)
    text = ' '.join(word for word in text.split() if word not in stopword_list)  # Hapus stopwords
    text = stemmer.stem(text)  # Stemming
    return text

# Load dataset
amazon_df = pd.read_csv('amazon.csv')

# Preprocess text data
amazon_df['product_name'] = amazon_df['product_name'].apply(clean_text)
amazon_df.reset_index(inplace=True, drop=True)

# Initialize TF-IDF Vectorizer
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=1)
tfidf_matrix = tf.fit_transform(amazon_df['product_name'])
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommendations(query, top=10):
    query_cleaned = clean_text(query)
    query_vec = tf.transform([query_cleaned])  # Transform query into TF-IDF vector
    query_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()  # Similarity with all journals

    # Ambil indeks dengan skor similarity tertinggi
    top_indices = query_sim.argsort()[-top:][::-1]  # Sort dari skor tertinggi ke terendah

    recommended_journals = amazon_df.iloc[top_indices]
    results = recommended_journals[['product_name', 'judul_prosessing']].reset_index(drop=True)

    # Jika tidak ada yang cocok
    if results.empty:
        return [f"Tidak ada produk amazon yang cocok dengan kata kunci '{query}'"]

    return results

# Streamlit app
st.title("Sistem Rekomendasi Produk Amazon ")

query_input = st.text_input("Masukkan kata atau kalimat pencarian:")
num_recommendations = st.slider("Jumlah rekomendasi amazon", min_value=1, max_value=30, value=5)

if st.button("Cari Rekomendasi"):
    if query_input:
        with st.spinner("Mencari rekomendasi..."):
            hasil_rekomendasi = recommendations(query_input, top=num_recommendations)
            st.write("Rekomendasi amazon untuk Anda:")
            if isinstance(hasil_rekomendasi, list):  # Jika tidak ada hasil
                st.write(hasil_rekomendasi[0])
            else:  # Jika ada hasil
                for idx, row in hasil_rekomendasi.iterrows():
                    st.write(f"{idx + 1}. {row['judul_prosessing']}")

# Tampilkan dataframe amazon untuk referensi
st.write("Dataset amazon:")
st.dataframe(amazon_df[['judul', 'judul_prosessing']])
