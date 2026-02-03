import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- 1. KONFIGURASI HALAMAN (Wajib di baris pertama) ---
st.set_page_config(
    page_title="Revenue Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- 2. FUNGSI UTILITIES (MODULAR) ---
@st.cache_data
def load_data():
    """Memuat dan membersihkan data dari CSV."""
    try:
        df_time = pd.read_csv('Dim_Time.csv')
        df_sales = pd.read_csv('Fact_Sales.csv')
        
        # Merge & Preprocessing
        df = df_sales.merge(df_time, on='Date_Key', how='left')
        df['Full_Date'] = pd.to_datetime(df['Full_Date'])
        df = df.sort_values('Full_Date')
        
        # Agregasi Bulanan
        monthly_data = df.set_index('Full_Date').resample('M')['Total Revenue'].sum()
        return monthly_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def build_model(data, n_test, seasonal_periods):
    """Membangun model Holt-Winters dan melakukan prediksi."""
    train = data.iloc[:-n_test]
    test = data.iloc[-n_test:]
    
    model = ExponentialSmoothing(
        train, 
        trend='add', 
        seasonal='add', 
        seasonal_periods=seasonal_periods,
        initialization_method='estimated'
    ).fit()
    
    forecast = model.forecast(len(test))
    return train, test, forecast, model

# --- 3. UI SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Konfigurasi Model")
    st.info("Atur parameter di bawah ini untuk melihat perubahan hasil prediksi secara real-time.")
    
    n_test = st.slider("Horizon Prediksi (Bulan)", 1, 12, 9)
    seasonal_periods = st.number_input("Periode Musiman", min_value=2, value=8)
    
    st.markdown("---")
    st.caption("Dikembangkan oleh: [Nama Kamu]")

# --- 4. MAIN INTERFACE ---
st.title("📈 European Sales Revenue Forecasting")
st.markdown("""
Dashboard ini dirancang untuk memprediksi pendapatan penjualan di masa depan menggunakan metode 
**Holt-Winters Exponential Smoothing**. Metode ini dipilih karena kemampuannya menangkap **Tren** dan **Musiman** pada data.
""")

# Load Data
data = load_data()

if data is not None:
    # Build Model
    train, test, forecast, model = build_model(data, n_test, seasonal_periods)
    
    # Hitung Error
    mae = mean_absolute_error(test, forecast)
    mape = np.mean(np.abs((test - forecast) / test.replace(0, np.nan))) * 100
    
    # --- KPI METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue (Aktual)", f"€ {test.sum():,.0f}")
    col2.metric("Total Revenue (Prediksi)", f"€ {forecast.sum():,.0f}")
    col3.metric("MAE (Rata-rata Error)", f"€ {mae:,.0f}", delta_color="inverse")
    col4.metric("MAPE (Akurasi Error)", f"{mape:.2f}%", delta_color="inverse")

    # --- TABS LAYOUT ---
    tab1, tab2, tab3 = st.tabs(["📊 Grafik Analisis", "🗃️ Data Detail", "ℹ️ Penjelasan Model"])

    with tab1:
        st.subheader("Perbandingan Data Aktual vs Prediksi")
        
        fig = go.Figure()
        # Data Latih
        fig.add_trace(go.Scatter(x=train.index, y=train.values, mode='lines', name='Data Latih (History)', line=dict(color='gray')))
        # Data Test (Aktual)
        fig.add_trace(go.Scatter(x=test.index, y=test.values, mode='lines+markers', name='Data Aktual', line=dict(color='green')))
        # Data Prediksi
        fig.add_trace(go.Scatter(x=test.index, y=forecast.values, mode='lines+markers', name='Prediksi Model', line=dict(color='red', dash='dash')))
        
        fig.update_layout(height=500, template='plotly_white', xaxis_title="Tanggal", yaxis_title="Revenue (€)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Tabel Hasil Prediksi")
        
        eval_df = pd.DataFrame({
            'Tanggal': test.index,
            'Aktual': test.values,
            'Prediksi': forecast.values,
            'Selisih': test.values - forecast.values
        })
        
        st.dataframe(eval_df.style.format("{:,.2f}"), use_container_width=True)
        
        # Download Button
        csv = eval_df.to_csv().encode('utf-8')
        st.download_button("📥 Download CSV", csv, "forecast_result.csv", "text/csv")

    with tab3:
        st.markdown("""
        ### Mengapa Holt-Winters?
        Data penjualan ini menunjukkan pola yang berulang (musiman) dan tren yang berubah seiring waktu. 
        Holt-Winters Triple Exponential Smoothing sangat cocok karena memiliki tiga komponen:
        1.  **Level:** Nilai dasar rata-rata.
        2.  **Trend:** Arah pergerakan data (naik/turun).
        3.  **Seasonality:** Pola musiman yang berulang.
        """)