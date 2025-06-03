import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
import skfuzzy as fuzz

# Load data
@st.cache_data
def load_data():
    return pd.read_excel('Data_Skripsi_Selected.xlsx')

df = load_data()

# UI layout
st.set_page_config(layout="wide")
st.markdown("""
    <style>
        .main {
            background-color: #1e1e1e;
            color: white;
        }
        .block-red {
            background-color: #ff5c5c;
            border-radius: 30px;
            padding: 20px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: white;
        }
        .block-green {
            background-color: #b6ff6c;
            border-radius: 30px;
            padding: 20px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Proyeksi Risiko Saham IPO Berdasarkan Fuzzy Mahalanobis")

col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### Deskripsi singkat...")
    st.markdown("**Input disini :**")
    ipo_price = st.number_input("IPO Price")
    market_cap = st.number_input("Market Cap")
    roe = st.number_input("Return on Equity")
    net_income = st.number_input("Net Income")
    der = st.number_input("Debt to Equity Ratio")
    free_float = st.number_input("Free Float")

    subsektor_list = [
        'Apparel & Luxury Goods', 'Properties & Real Estate', 'Oil, Gas, & Coal',
        'Automobiles & Components', 'Basic Materials', 'Food & Beverage', 'Software & IT Service',
        'Utilities', 'Retailing', 'Heavy Constructions & Civil', 'Consumer Services',
        'Healthcare Equipment & Providers', 'Leisure Goods', 'Industrial Services',
        'Nondurable Household Products', 'Food & Staples Retailing ', 'Technology Hardware',
        'Telecommunication', 'Media & Entertainment', 'Logistics & Deliveries',
        'Multi Sector Holdings', 'Industrial Goods', 'Household Goods', 'Banks',
        'Phramaceuticals & Healthcare', 'Alternative Energy', 'Financing Service'
    ]
    subsektor = st.selectbox("Pilih Sub Sektor", subsektor_list)


with col2:
    st.markdown("### Input Data")
    st.dataframe(pd.DataFrame([{
        'Harga IPO': ipo_price,
        'Market Cap': market_cap,
        'Return on Equity': roe,
        'Net Income': net_income,
        'Debt to Equity Ratio': der,
        'Free Float': free_float
    }]))

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

        metric_features = {
            'std': ['ipo_price', 'der'],
            'dsd': ['ipo_price', 'free_float'],
            'sharpe_ratio': ['der', 'net_income'],
            'sortino_ratio': ['roe', 'der'],
            'liquidity_ratio': ['market_cap', 'der', 'free_float']
        }

        for metric, features in metric_features.items():
            input_vector = np.array([input_data[feat] for feat in features])
            df_sub = df[df['Sub Sektor'] == input_data['Sub Sektor']].copy()
            if df_sub.empty:
                df_sub = df.copy()

            df_sub = calculate_mahalanobis_fuzzy(input_vector, df_sub, features)
            top3 = df_sub.sort_values(by='fuzzy_similarity', ascending=False).head(3)
            avg_value = top3[metric].mean()
            results[metric] = avg_value

        return results

    if st.button("Hitung Proyeksi Risiko"):
        input_data = {
            'ipo_price': ipo_price,
            'market_cap': market_cap,
            'roe': roe,
            'net_income': net_income,
            'der': der,
            'free_float': free_float,
            'Sub Sektor': subsektor
        }

        hasil = get_avg_top3_similarity(input_data, df)

        st.markdown(f"""
            <div class="block-red">NILAI STD : {hasil['std']:.4f}</div>
            <br>
            <div class="block-red">NILAI DSD : {hasil['dsd']:.4f}</div>
            <br><br>
            <div class="block-green">Sharpe Ratio : {hasil['sharpe_ratio']:.4f}</div>
            <br>
            <div class="block-green">Sortino Ratio : {hasil['sortino_ratio']:.4f}</div>
            <br>
            <div class="block-green">Liquidity Ratio : {hasil['liquidity_ratio']:.4f}</div>
        """, unsafe_allow_html=True)
