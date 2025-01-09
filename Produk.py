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
    if not isinstance(text, str):
        return ""  # Handle non-string values
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
    required_columns = ['Item Purchased', 'Review Rating', 'Purchase Amount (USD)', 'Color', 'Size', 'Location']
    for col in required_columns:
        if col not in amazon_df.columns:
            st.error(f"Kolom '{col}' tidak ditemukan di dataset.")
            st.stop()

    # Handle missing values
    amazon_df['Item Purchased'].fillna("Unknown Product", inplace=True)
    amazon_df['Review Rating'].fillna(amazon_df['Review Rating'].mean(), inplace=True)  # Fill numeric columns with mean
    amazon_df['Purchase Amount (USD)'].fillna(amazon_df['Purchase Amount (USD)'].mean(), inplace=True)
    amazon_df['Color'].fillna("Unknown Color", inplace=True)
    amazon_df['Size'].fillna("Unknown Size", inplace=True)
    amazon_df['Location'].fillna("Unknown Location", inplace=True)

    # Remove noise (e.g., duplicate rows)
    amazon_df.drop_duplicates(inplace=True)

    # Add preprocessed column
    amazon_df['item processing'] = amazon_df['Item Purchased'].apply(clean_text)

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
    results = recommended_products[['Item Purchased', 'Review Rating', 'Purchase Amount (USD)', 'Color', 'Size', 'Location']]

    # If no matches are found
    if results.empty:
        return [f"Tidak ada produk amazon yang cocok dengan kata kunci '{query}'"]

    # Convert to list of dictionaries
    return results.to_dict('records')

# Streamlit app
st.title("Shopping Trends Recommendation System")

# Load and preprocess data
file_path = "shopping_trends.xlsx"
amazon_df = load_and_preprocess_data(file_path)

# Compute TF-IDF matrix
tf, tfidf_matrix = compute_tfidf_matrix(amazon_df['item processing'])

query_input = st.text_input("Enter a search word or phrase :")
num_recommendations = st.slider("Recommendation For You", min_value=1, max_value=30, value=5)

if st.button("Cari Rekomendasi"):
    if query_input:
        with st.spinner("Mencari rekomendasi..."):
            hasil_rekomendasi = recommendations(query_input, tf, tfidf_matrix, amazon_df, top=num_recommendations)
            st.write("Rekomendasi amazon untuk Anda:")

            if isinstance(hasil_rekomendasi, list): 
                for idx, item in enumerate(hasil_rekomendasi, start=1):
                    st.write(f"**{idx}. {item['Item Purchased']}**")
                    st.write(f"- **Review Rating:** {item['Review Rating']}")
                    st.write(f"- **Purchase Amount (USD):** {item['Purchase Amount (USD)']}")
                    st.write(f"- **Color:** {item['Color']}")
                    st.write(f"- **Size:** {item['Size']}")
                    st.write(f"- **Location:** {item['Location']}")
                    st.write("---")  

# Display dataset for reference
st.write("Product Trends All :")
st.dataframe(amazon_df[['item processing', 'Item Purchased', 'Review Rating', 'Purchase Amount (USD)', 'Color', 'Size', 'Location']])
