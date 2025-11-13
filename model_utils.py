import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# ---------------------------------
# 1. Preprocessing Data
# ---------------------------------
def load_and_clean_data(file_path):
    """Membaca dan membersihkan dataset mainan Amazon."""
    df = pd.read_csv(file_path)

    # manufacturer & kategori
    df['manufacturer'].fillna('Unknown', inplace=True)
    df['amazon_category_and_sub_category'].fillna('Unknown', inplace=True)

    # price
    df['price'] = df['price'].astype(str).str.replace('£', '').str.strip()
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['price'].fillna(df['price'].median(), inplace=True)

    # number_of_reviews
    df['number_of_reviews'] = pd.to_numeric(df['number_of_reviews'], errors='coerce')
    df['number_of_reviews'].fillna(df['number_of_reviews'].median(), inplace=True)

    # average_review_rating
    df['average_review_rating'] = df['average_review_rating'].apply(
        lambda x: float(re.search(r'(\d+\.\d+)', str(x)).group(1))
        if pd.notnull(x) and re.search(r'(\d+\.\d+)', str(x))
        else np.nan
    )
    df['average_review_rating'].fillna(df['average_review_rating'].median(), inplace=True)

    # teks kolom
    df['description'].fillna('', inplace=True)
    df['product_information'].fillna('', inplace=True)
    df['product_description'].fillna('', inplace=True)

    return df


# ---------------------------------
# 2. Persiapan Fitur
# ---------------------------------
def clean_text(text):
    """Membersihkan teks: huruf kecil, hapus simbol, dan spasi berlebih."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def prepare_features(df):
    """Mempersiapkan fitur numerik dan teks untuk model rekomendasi."""
    # TF-IDF kategori
    df['category_text'] = df['amazon_category_and_sub_category'].str.replace('>', ' ', regex=False)
    df['category_text'] = df['category_text'].apply(clean_text)
    vectorizer_category = TfidfVectorizer(max_features=1000)
    category_tfidf_matrix = vectorizer_category.fit_transform(df['category_text'])

    # TF-IDF teks gabungan
    df['combined_text'] = (
        df['description'].fillna('') + ' ' +
        df['product_information'].fillna('') + ' ' +
        df['product_description'].fillna('')
    ).apply(clean_text)
    vectorizer_text = TfidfVectorizer(max_features=5000)
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
    df = pd.concat([df, scaled_numeric_df], axis=1)

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
def search_product(df, keyword):
    """Mencari produk berdasarkan nama (case-insensitive)."""
    return df[df["product_name"].str.contains(keyword, case=False, na=False)]


def recommend_products(df, product_name, similarity_matrix, n=5):
    """Memberikan rekomendasi produk mirip berdasarkan kemiripan konten."""
    matches = df[df["product_name"] == product_name]

    if matches.empty:
        return None  # biar Streamlit bisa tampilkan warning sendiri

    idx = matches.index[0]

    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : n + 1]  # lewati dirinya sendiri

    indices = [i[0] for i in sim_scores]
    return df.iloc[indices][["product_name", "manufacturer", "price", "average_review_rating"]]


# ---------------------------------
# 4. Pipeline Lengkap
# ---------------------------------
def load_and_prepare_model(file_path):
    """Pipeline lengkap untuk load data, cleaning, dan menyiapkan fitur."""
    df = load_and_clean_data(file_path)
    df, combined_matrix, similarity_matrix, vectorizer_category, vectorizer_text, scaler = prepare_features(df)
    return df, combined_matrix, similarity_matrix, vectorizer_category, vectorizer_text, scaler
