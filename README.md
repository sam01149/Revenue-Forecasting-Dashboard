European Sales Revenue Forecasting

Project Overview
Proyek ini bertujuan untuk membantu perusahaan memprediksi pendapatan penjualan (Revenue) bulanan di masa depan. Prediksi yang akurat sangat penting untuk perencanaan anggaran, manajemen inventaris, dan strategi pemasaran.

Aplikasi ini menggunakan algoritma **Holt-Winters Exponential Smoothing** karena karakteristik data penjualan yang memiliki pola **Tren** dan **Musiman** yang kuat.

Live Demo:[Klik disini untuk melihat Dashboard](LINK_STREAMLIT_KAMU_NANTI_DISINI)

## Features
- **Preprocessing Otomatis:** Menggabungkan data dimensi waktu dan fakta penjualan.
- **Model Tuning:** Sidebar interaktif untuk mengubah parameter `seasonal_periods` dan horizon prediksi.
- **Visualisasi Interaktif:** Grafik Plotly yang dinamis untuk membandingkan data aktual vs prediksi.
- **Evaluasi Metrik:** Perhitungan otomatis MAE (Mean Absolute Error) dan MAPE (Mean Absolute Percentage Error).

## Dataset
Data yang digunakan terdiri dari dua tabel utama:
1.  `Fact_Sales.csv`: Data transaksi penjualan harian.
2.  `Dim_Time.csv`: Dimensi waktu untuk detail tanggal.

## Tech Stack
- **Python**: Bahasa pemrograman utama.
- **Pandas & NumPy**: Manipulasi dan agregasi data.
- **Statsmodels**: Pembuatan model Time Series (Holt-Winters).
- **Streamlit**: Framework untuk membuat dashboard interaktif.
- **Plotly**: Visualisasi data.

## How to Run Locally
Jika Anda ingin menjalankan proyek ini di komputer lokal Anda:

1. **Clone repository ini**
   ```bash
   git clone [https://github.com/username-kamu/revenue-forecasting.git](https://github.com/username-kamu/revenue-forecasting.git)
