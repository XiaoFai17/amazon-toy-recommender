import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# ---------------------------------
# 1. Preprocessing Data
# ---------------------------------
def load_and_clean_data(file_path):
    """Membaca dan membersihkan dataset mainan Amazon."""
    # Check if file exists before reading
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File data tidak ditemukan di: {file_path}")
        
    df = pd.read_csv(file_path)

    # manufacturer & kategori
    df['manufacturer'] = df['manufacturer'].fillna('Unknown')
    df['amazon_category_and_sub_category'] = df['amazon_category_and_sub_category'].fillna('Unknown')

    # price
    # Use a more robust regex to handle various currency symbols and ensure proper cleaning
    df['price'] = df['price'].astype(str).str.replace(r'[^\d\.]', '', regex=True).str.strip()
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['price'] = df['price'].fillna(df['price'].median())

    # number_of_reviews
    df['number_of_reviews'] = pd.to_numeric(df['number_of_reviews'], errors='coerce')
    df['number_of_reviews'] = df['number_of_reviews'].fillna(df['number_of_reviews'].median())

    # average_review_rating
    df['average_review_rating'] = df['average_review_rating'].apply(
        lambda x: float(re.search(r'(\d+\.\d+)', str(x)).group(1))
        if pd.notnull(x) and re.search(r'(\d+\.\d+)', str(x))
        else np.nan
    )
    df['average_review_rating'] = df['average_review_rating'].fillna(df['average_review_rating'].median())

    # teks kolom
    df['description'] = df['description'].fillna('')
    df['product_information'] = df['product_information'].fillna('')
    df['product_description'] = df['product_description'].fillna('')

    return df


# ---------------------------------
# 2. Persiapan Fitur
# ---------------------------------
def clean_text(text):
    """Membersihkan teks: huruf kecil, hapus simbol, dan spasi berlebih."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def prepare_features(df):
    """Mempersiapkan fitur numerik dan teks untuk model rekomendasi."""
    # TF-IDF kategori
    df['category_text'] = df['amazon_category_and_sub_category'].str.replace('>', ' ', regex=False)
    df['category_text'] = df['category_text'].apply(clean_text)
    vectorizer_category = TfidfVectorizer(max_features=300)
    category_tfidf_matrix = vectorizer_category.fit_transform(df['category_text'])

    # TF-IDF teks gabungan
    df['combined_text'] = (
        df['description'].fillna('') + ' ' +
        df['product_information'].fillna('') + ' ' +
        df['product_description'].fillna('')
    ).apply(clean_text)
    vectorizer_text = TfidfVectorizer(max_features=1000)
    text_tfidf_matrix = vectorizer_text.fit_transform(df['combined_text'])

    # manufacturer encoding
    manufacturer_freq = df['manufacturer'].value_counts()
    df['manufacturer_encoded'] = df['manufacturer'].map(manufacturer_freq)
    manufacturer_encoded_reshaped = df['manufacturer_encoded'].values.reshape(-1, 1)

    # fitur numerik
    numeric_cols = ['price', 'number_of_reviews', 'average_review_rating']
    scaler = StandardScaler()
    scaled_numeric = scaler.fit_transform(df[numeric_cols])
    scaled_numeric_df = pd.DataFrame(
        scaled_numeric,
        columns=[col + '_scaled' for col in numeric_cols]
    )
    df = pd.concat([df.reset_index(drop=True), scaled_numeric_df], axis=1)

    numeric_scaled_matrix = scaled_numeric_df.values

    # gabungkan semua fitur
    combined_matrix = hstack([
        category_tfidf_matrix,
        text_tfidf_matrix,
        manufacturer_encoded_reshaped,
        numeric_scaled_matrix
    ])

    # matriks kemiripan
    similarity_matrix = cosine_similarity(combined_matrix)

    return df, combined_matrix, similarity_matrix, vectorizer_category, vectorizer_text, scaler


# ---------------------------------
# 3. Pencarian Rekomendasi
# ---------------------------------
# Updated function to take product index instead of product name
def recommend_products(df, product_index, similarity_matrix, n=5):
    """Memberikan rekomendasi produk mirip berdasarkan kemiripan konten."""
    
    # Check if the index is valid
    if product_index not in df.index:
        return None

    # Get the similarity scores for the product at the given index
    sim_scores = list(enumerate(similarity_matrix[product_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores for the top N products (skipping the first one which is the product itself)
    sim_scores = sim_scores[1 : n + 1]

    indices = [i[0] for i in sim_scores]
    
    # Return the recommended products' details
    return df.iloc[indices][["product_name", "manufacturer", "price", "average_review_rating", "amazon_category_and_sub_category"]]


# ---------------------------------
# 4. Pipeline Lengkap
# ---------------------------------
def load_and_prepare_model(file_path):
    """Pipeline lengkap untuk load data, cleaning, dan menyiapkan fitur."""
    df = load_and_clean_data(file_path)
    df, combined_matrix, similarity_matrix, vectorizer_category, vectorizer_text, scaler = prepare_features(df)
    return df, combined_matrix, similarity_matrix, vectorizer_category, vectorizer_text, scaler
