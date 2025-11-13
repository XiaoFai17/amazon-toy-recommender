import streamlit as st
import pandas as pd
import base64
import os
from model_utils import load_and_prepare_model, recommend_products

# === SETUP STREAMLIT ===
st.set_page_config(page_title="Amazon Toy Recommender", layout="wide")

# === CSS UNTUK MARQUEE LOGO SAJA ===
st.markdown("""
    <style>
        .marquee-container {
            overflow: hidden;
            background-color: #f8f8f8;
            padding: 20px 0;
        }

        .marquee-track {
            display: flex;
            width: fit-content;
            animation: marquee 30s linear infinite;
        }

        .marquee-track img {
            height: 100px;
            margin: 0 20px;
            background: white;
            padding: 10px;
            border-radius: 15px;
            box-shadow: 0px 0px 8px rgba(0, 0, 0, 0.1);
        }

        @keyframes marquee {
            0% { transform: translateX(0); }
            100% { transform: translateX(-50%); }
        }
       
        footer {
            text-align: center;
            padding: 20px;
            color: #888;
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

# === HEADER AMAZON ===
st.markdown("""
    <div style='text-align: center'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg' width='250'/>
        <h1>Amazon Toy Recommender 🏱</h1>
        <p>Sistem rekomendasi mainan berbasis konten dari data Amazon</p>
    </div>
""", unsafe_allow_html=True)

# === SECTION: LOGO BRAND ===
st.markdown("#### 🔍 Jelajahi berdasarkan Merek Populer")

manufacturer_logos = [
    ("LEGO", "lego.jpg"),
    ("Disney", "disney.png"),
    ("Oxford Diecast", "oxford.webp"),
    ("Playmobil", "playmobil.png"),
    ("Star Wars", "starwars.jpg"),
    ("Mattel", "mattel.png"),
    ("Hasbro", "hasbro.png"),
    ("The Puppet Company", "thepuppetcompany.png"),
    ("My Tiny World", "mytinyworld.png"),
    ("Hot Wheels", "hotwheels.webp"),
]

# duplikat untuk animasi marquee
logos_full = manufacturer_logos + manufacturer_logos

scroll_html = "<div class='marquee-container'><div class='marquee-track'>"
for brand, filename in logos_full:
    filepath = os.path.join("logo", filename)
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            data_url = base64.b64encode(f.read()).decode("utf-8")
            ext = filename.split(".")[-1]
            scroll_html += f"<img src='data:image/{ext};base64,{data_url}' title='{brand}'/>"
    else:
        scroll_html += f"<div style='text-align:center'>{brand}<br/>(gambar tidak ditemukan)</div>"
scroll_html += "</div></div>"

st.markdown(scroll_html, unsafe_allow_html=True)
st.markdown("---")

# === LOAD DATA DAN SIAPKAN FITUR ===
# Use st.cache_data for heavy computations like model loading and feature preparation.
@st.cache_resource(show_spinner=False)
def load_data_pipeline():
    """Load dataset dan siapkan fitur untuk sistem rekomendasi."""
    # Ensure the data file path is correct relative to the Streamlit app
    file_path = os.path.join(os.path.dirname(__file__), "data", "amazon_co-ecommerce_sample.csv")
    return load_and_prepare_model(file_path)

# Check if data is already loaded in session state
if 'df' not in st.session_state:
    with st.spinner("🔄 Memuat data dan menyiapkan fitur..."):
        # The load_data_pipeline function returns 6 objects
        (
            st.session_state.df,
            st.session_state.combined_matrix,
            st.session_state.similarity_matrix,
            st.session_state.vectorizer_category,
            st.session_state.vectorizer_text,
            st.session_state.scaler
        ) = load_data_pipeline()

# Assign variables from session state for cleaner code
df = st.session_state.df
similarity_matrix = st.session_state.similarity_matrix

# === DISPLAY CARD ===
def display_product_card(product):
    """Menampilkan 1 kartu produk di Streamlit."""
    st.markdown("----")
    st.subheader(product.get("product_name", "—"))
    st.markdown(f"**Manufacturer**: {product.get('manufacturer', 'Unknown')}")
    # Format price to two decimal places
    price = product.get('price', '—')
    if isinstance(price, (int, float)):
        price = f"{price:.2f}"
    st.markdown(f"**Price**: £{price}")
    st.markdown(f"**Average Rating**: ⭐ {product.get('average_review_rating', '—')}")
    st.markdown(f"**Category**: `{product.get('amazon_category_and_sub_category', '—')}`")
    st.markdown("----")

# === SEARCH INTERFACE ===
keyword = st.text_input("🔍 Cari mainan berdasarkan nama atau merek (contoh: hotwheels, lego)")

if keyword:
    # Use .str.lower() for case-insensitive search on columns that might contain NaN
    mask = df["product_name"].str.lower().str.contains(keyword.lower(), na=False) | \
           df["manufacturer"].str.lower().str.contains(keyword.lower(), na=False)
    filtered_df = df[mask]

    if not filtered_df.empty:
        # Use a key for the selectbox to prevent Streamlit warning
        selected_product_name = st.selectbox("🎯 Pilih Produk:", filtered_df["product_name"].unique(), key="product_select")

        if selected_product_name:
            selected_product = df[df["product_name"] == selected_product_name].iloc[0]

            st.markdown("## 📌 Produk yang Dipilih")
            display_product_card(selected_product)

            if st.button("🔁 Tampilkan Rekomendasi"):
                st.markdown("## 🏱 Produk Rekomendasi")
                
                # Get the index of the selected product
                selected_idx = df[df["product_name"] == selected_product_name].index[0]
                
                # Call the updated recommend_products function
                recommendations = recommend_products(df, selected_idx, similarity_matrix, n=10)

                if recommendations is not None and not recommendations.empty:
                    for _, row in recommendations.iterrows():
                        display_product_card(row)
                else:
                    st.warning("⚠️ Tidak ada produk mirip yang ditemukan.")
    else:
        st.warning("⚠️ Tidak ditemukan produk dengan kata kunci tersebut.")
else:
    st.info("ℹ️ Masukkan kata kunci di atas untuk mencari produk.")

# === FOOTER ===
st.markdown("""
    <footer>
        3323600017 <b>Faishal IR</b>
    </footer>
""", unsafe_allow_html=True)
