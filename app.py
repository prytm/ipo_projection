import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
import skfuzzy as fuzz

# Load Data
st.set_page_config(page_title="Fuzzy Risk Projection", layout="wide")
st.title("📊 Fuzzy Mahalanobis Risk Projection")

# Load data
@st.cache_data
def load_data():
    return pd.read_excel('Data_Skripsi_Selected.xlsx')

df = load_data()

# Sidebar inputs
st.sidebar.header("Input Stock Information")

input_data = {
    'code': st.sidebar.text_input("Kode Saham", value='ACRO.JK'),
    'ipo_price': st.sidebar.number_input("IPO Price (Z-Score)", value=-0.412359),
    'market_cap': st.sidebar.number_input("Market Cap (Z-Score)", value=-0.251000),
    'roe': st.sidebar.number_input("ROE (Z-Score)", value=0.017450),
    'net_income': st.sidebar.number_input("Net Income (Z-Score)", value=-0.093913),
    'der': st.sidebar.number_input("DER (Z-Score)", value=-0.087869),
    'free_float': st.sidebar.number_input("Free Float (Z-Score)", value=-0.035093)
}

subsektor_options = [
    'Apparel & Luxury Goods', 'Properties & Real Estate',
    'Oil, Gas, & Coal', 'Automobiles & Components', 'Basic Materials',
    'Food & Beverage', 'Software & IT Service', 'Utilities',
    'Retailing', 'Heavy Constructions & Civil', 'Consumer Services',
    'Healthcare Equipment & Providers', 'Leisure Goods',
    'Industrial Services', 'Nondurable Household Products',
    'Food & Staples Retailing ', 'Technology Hardware',
    'Telecommunication', 'Media & Entertainment',
    'Logistics & Deliveries', 'Multi Sector Holdings',
    'Industrial Goods', 'Household Goods', 'Banks',
    'Phramaceuticals & Healthcare', 'Alternative Energy',
    'Financing Service'
]

input_data['Sub Sektor'] = st.sidebar.selectbox("Sub Sektor:", subsektor_options, index=subsektor_options.index('Apparel & Luxury Goods'))

# Function definitions
def calculate_mahalanobis_fuzzy(input_vector, df_sub, features):
    try:
        cov_matrix = df_sub[features].cov()
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        inv_cov_matrix = np.diag(1 / df_sub[features].var())

    df_sub = df_sub.copy()
    df_sub['mahalanobis_distance'] = df_sub[features].apply(
        lambda row: mahalanobis(input_vector, row.values, inv_cov_matrix), axis=1
    )

    dist_min = df_sub['mahalanobis_distance'].min()
    dist_max = df_sub['mahalanobis_distance'].max()
    dist_range = dist_max - dist_min
    dist_x = np.arange(0, dist_range + 1, 0.1)
    dist_close = fuzz.trimf(dist_x, [0, 0, dist_range / 2])

    df_sub['fuzzy_similarity'] = df_sub['mahalanobis_distance'].apply(
        lambda d: fuzz.interp_membership(dist_x, dist_close, d - dist_min)
    )

    return df_sub

def get_avg_top3_similarity(input_data, df):
    results = {}
    subsektor = input_data['Sub Sektor']

    metric_features = {
        'std': ['ipo_price', 'der'],
        'dsd': ['ipo_price', 'free_float'],
        'sharpe_ratio': ['der', 'net_income'],
        'sortino_ratio': ['roe', 'der'],
        'liquidity_ratio': ['market_cap', 'der', 'free_float']
    }

    for metric, features in metric_features.items():
        input_vector = np.array([input_data[feat] for feat in features])
        df_sub = df[df['Sub Sektor'] == subsektor].copy()
        if df_sub.empty:
            df_sub = df.copy()

        df_sub = calculate_mahalanobis_fuzzy(input_vector, df_sub, features)
        top3 = df_sub.sort_values(by='fuzzy_similarity', ascending=False).head(3)
        avg_value = top3[metric].mean()
        results[metric] = avg_value

    return results

# Calculate and Display
if st.button("🔍 Run Projection"):
    st.subheader("📈 Rata-rata Proyeksi Risiko dari 3 Saham Termirip")
    avg_top3 = get_avg_top3_similarity(input_data, df)

    for metric, value in avg_top3.items():
        color = "#90ee90" if value >= 0 else "#ffcccb"  # green if positive, red if negative
        st.markdown(f"""
        <div style='background-color: {color}; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>
            <strong>{metric.upper()}</strong>: {value:.4f}
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("Masukkan data terlebih dahulu dan klik tombol 'Run Projection'")
