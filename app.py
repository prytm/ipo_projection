import streamlit as st
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(
    page_title="Risk-Projection Model",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================== Sidebar =====================
st.sidebar.title("📊 Risk-Projection Model")
st.sidebar.markdown("""
**Created by:**

<span style='color: lightgreen;'>🟩 Priya Tammam</span>
""", unsafe_allow_html=True)

stock_code = st.sidebar.text_input("Stock Target Code (ex: KAQI.JK):", "KAQI.JK")
total_assets = st.sidebar.number_input("Total Assets Target (in IDR):", value=74129409370)
total_equity = st.sidebar.number_input("Total Equities Target (in IDR):", value=59158929209)
market_cap = st.sidebar.number_input("Market Cap Target (in IDR):", value=531000000)
net_profit = st.sidebar.number_input("net Profit Current Year Period (in IDR):", value=73414622023)

subsektor_list = ['Apparel & Luxury Goods', 'Properties & Real Estate',
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
       'Financing Service']

subsector = st.sidebar.selectbox("Sub Sektor Target:", subsektor_list, index=1)

# ===================== Main Section =====================
st.title("Mahalanobis Risk-Projection Model")

# Display Input Summary
input_df = pd.DataFrame({
    "Stock": [stock_code],
    "Total Assets": [total_assets],
    "Market Cap": [market_cap],
    "Total Equities": [total_equity],
    "Net Income": [net_profit]
})

st.table(input_df)

# Dummy Value at Risk (to be replaced with actual Mahalanobis or Fuzzy calc)
st.markdown("""
<div style='background-color:#ffcccc; padding: 20px; border-radius: 10px; text-align: center;'>
    <h5>📉 Absolute Value at Risk</h5>
    <h1 style='font-size: 2.5rem;'>0.32</h1>
</div>
""", unsafe_allow_html=True)

# ===================== Weekly Returns Plot Section =====================
st.subheader("Weekly Returns Plot Projection")
st.markdown("""
Explore how the daily returns projection with 95% confidence interval Moving Average fluctuate 3 months after IPO
""")

# Dummy chart
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(42)
data = np.random.normal(0.2, 0.15, size=12)
dates = pd.date_range("2023-01-01", periods=12, freq="W")
df_returns = pd.DataFrame({"date": dates, "return": data})
df_returns.set_index("date", inplace=True)

fig, ax = plt.subplots(figsize=(10, 4))
df_returns["return"].plot(ax=ax, label="Daily Return", color="cyan")
df_returns["return"].rolling(3).mean().plot(ax=ax, label="SMA (10)", color="orange")
std = df_returns["return"].std()
mean = df_returns["return"].mean()
upper = mean + 1.96 * std
lower = mean - 1.96 * std
ax.axhline(upper, linestyle="--", color="green", label="Upper Band")
ax.axhline(lower, linestyle="--", color="red", label="Lower Band")
ax.set_title(f"{stock_code} Weekly Returns")
ax.legend()
st.pyplot(fig)
