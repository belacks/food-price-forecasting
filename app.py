import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import re
# import gdown # Tetap dikomentari, fokus pada local models folder
import time
from datetime import datetime, date, timedelta # Import datetime

# --- Konfigurasi ---
MODELS_BASE_FOLDER = "multioutput-models" # Folder model/scaler
DATA_FILE_PATH = 'data_pangan_jabodetabek_wide_imputed.parquet'
MODEL_FILENAME_PATTERN = "model_multioutput30d_{safe_name}.keras"
SCALER_FILENAME_PATTERN = "scaler_multioutput30d_{safe_name}.gz"
LOOK_BACK = 60
MAX_HORIZON = 30 # Batas prediksi maksimal

# --- Helper Functions ---
def safe_filename(name):
    name = re.sub(r'[\\/ ]', '_', name)
    name = name[:80]
    return name

@st.cache_resource(ttl=3600)
def load_keras_model(target_column_name):
    safe_name = safe_filename(target_column_name)
    model_filename = MODEL_FILENAME_PATTERN.format(safe_name=safe_name)
    model_path = os.path.join(MODELS_BASE_FOLDER, model_filename)
    if os.path.exists(model_path):
        try:
            return load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model {model_filename}: {e}")
            return None
    else:
        st.error(f"Model file '{model_filename}' not found in '{MODELS_BASE_FOLDER}'.")
        return None

@st.cache_resource(ttl=3600)
def load_joblib_scaler(target_column_name):
    safe_name = safe_filename(target_column_name)
    scaler_filename = SCALER_FILENAME_PATTERN.format(safe_name=safe_name)
    scaler_path = os.path.join(MODELS_BASE_FOLDER, scaler_filename)
    if os.path.exists(scaler_path):
        try:
            return joblib.load(scaler_path)
        except Exception as e:
            st.error(f"Error loading scaler {scaler_filename}: {e}")
            return None
    else:
        st.error(f"Scaler file '{scaler_filename}' not found in '{MODELS_BASE_FOLDER}'.")
        return None

@st.cache_data(ttl=3600)
def load_data(filepath):
    print(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        st.error(f"Data file not found: {filepath}")
        return None, None, None, None
    try:
        df = pd.read_parquet(filepath)
        df.index = pd.to_datetime(df.index)
        locations = sorted(list(set(col.split('_')[0] for col in df.columns)))
        ref_loc = locations[0] if locations else ""
        commodities = sorted(list(set("_".join(col.split('_')[1:]) for col in df.columns if col.startswith(ref_loc + '_'))))
        last_data_date = df.index[-1] # Tanggal data terakhir
        print("Data loaded successfully.")
        return df, locations, commodities, last_data_date
    except Exception as e:
        st.error(f"Error loading/parsing data: {e}")
        return None, None, None, None

def predict_recursive(model, initial_sequence_scaled, n_steps_out, scaler):
    current_sequence = initial_sequence_scaled.copy().reshape(1, LOOK_BACK, 1)
    predictions_scaled = []
    for _ in range(n_steps_out):
        next_pred_scaled = model.predict(current_sequence, verbose=0)[0, 0]
        predictions_scaled.append(next_pred_scaled)
        next_pred_scaled_reshaped = np.array([[next_pred_scaled]])
        current_sequence = np.vstack((current_sequence[0, 1:, :], next_pred_scaled_reshaped))
        current_sequence = current_sequence.reshape(1, LOOK_BACK, 1)
    predictions_actual = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))
    return predictions_actual.flatten()

# --- Streamlit App Layout ---
st.set_page_config(page_title="Harga Pangan Jabodetabek", page_icon="📈", layout="wide")

# Load Data Pertama Kali
df_wide, locations, commodities, last_data_date = load_data(DATA_FILE_PATH)

# Header Aplikasi
col_header1, col_header2 = st.columns([1, 5])
with col_header1:
     # Anda bisa ganti dengan logo jika punya
     st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Flag_of_Indonesia.svg/800px-Flag_of_Indonesia.svg.png", width=80)
with col_header2:
    st.title("📈 Prediksi Harga Pangan Strategis - Jabodetabek")
    st.caption(f"Analisis & Prediksi Harian | Data terakhir diperbarui: {last_data_date.strftime('%d %B %Y') if last_data_date else 'N/A'}")

# Peringatan Penting tentang Update Data & Status Proyek
st.info(f"""
    ℹ️ **Informasi Penting:**
    *   Data historis yang digunakan dalam aplikasi ini terakhir diperbarui pada **{last_data_date.strftime('%d %B %Y') if last_data_date else 'Tanggal Tidak Diketahui'}**.
    *   Prediksi dihasilkan berdasarkan data hingga tanggal tersebut.
    *   Aplikasi ini adalah **Proyek Portofolio (Work In Progress v0.2)**. Update data dan model direncanakan secara periodik (misalnya bulanan).
    *   Prediksi bersifat **eksperimental** dan akurasi menurun untuk jangka waktu yang lebih panjang.
""")
st.divider()

# Cek jika data gagal dimuat
if df_wide is None or not locations or not commodities or last_data_date is None:
    st.error("❌ Gagal memuat data historis. Aplikasi tidak dapat melanjutkan.")
    st.stop() # Hentikan eksekusi skrip jika data tidak ada

# --- Sidebar untuk Input ---
st.sidebar.header("📌 Pilih Prediksi")
selected_location = st.sidebar.selectbox("1. Pilih Lokasi:", locations, key="loc_select")
selected_commodity = st.sidebar.selectbox("2. Pilih Komoditas:", commodities, key="com_select")

st.sidebar.header("🗓️ Pilih Tanggal Prediksi")
# Tanggal mulai prediksi adalah H+1 dari data terakhir
min_pred_date = last_data_date.date() + timedelta(days=1)
# Tanggal maksimal prediksi adalah H+30 dari data terakhir
max_pred_date = last_data_date.date() + timedelta(days=MAX_HORIZON)

selected_pred_date = st.sidebar.date_input(
    "Prediksi hingga tanggal:",
    value=min_pred_date, # Default ke hari berikutnya
    min_value=min_pred_date,
    max_value=max_pred_date, # Batasi pemilihan maksimal 30 hari
    key="pred_date_select",
    help=f"Pilih tanggal akhir prediksi (maksimal {MAX_HORIZON} hari setelah {last_data_date.strftime('%d %b %Y')})"
)

# Hitung jumlah hari prediksi
days_to_predict = (selected_pred_date - last_data_date.date()).days
st.sidebar.caption(f"Ini adalah prediksi untuk {days_to_predict} hari ke depan.")

# Tombol Prediksi
st.sidebar.divider()
predict_button = st.sidebar.button(f"🔮 Prediksi {days_to_predict} Hari", type="primary", use_container_width=True)

# Info Tambahan di Sidebar
st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ Tentang Model & Data"):
    st.markdown(f"""
    *   **Model:** LSTM Univariate (terpisah per komoditas-lokasi)
    *   **Input:** Histori {LOOK_BACK} hari terakhir
    *   **Data Source:** PIHPS Nasional (diolah)
    *   **Data Cutoff:** {last_data_date.strftime('%d %b %Y')}
    *   **Update:** Direncanakan bulanan
    """)

# --- Area Konten Utama ---
target_column_name = f"{selected_location}_{selected_commodity}"

if target_column_name not in df_wide.columns:
    st.error(f"Data untuk '{selected_commodity}' di '{selected_location}' tidak tersedia.")
else:
    # Ambil data historis untuk target terpilih
    ts_data = df_wide[[target_column_name]].copy()
    ts_data.rename(columns={target_column_name: 'Harga Aktual'}, inplace=True)
    last_actual_value = ts_data.iloc[-1, 0]

    # --- Tab untuk Tampilan ---
    tab_hist, tab_pred = st.tabs(["📊 Analisis Historis", "🚀 Prediksi Harga"])

    with tab_hist:
        st.subheader(f"Analisis Harga Historis: {selected_commodity}")
        st.caption(f"Lokasi: {selected_location}")

        # Pilih periode waktu untuk ditampilkan (Fitur dari kode Anda)
        time_periods = {
            "30 Hari Terakhir": 30, "90 Hari Terakhir": 90,
            "6 Bulan Terakhir": 180, "1 Tahun Terakhir": 365,
            "Semua Waktu": len(ts_data)
        }
        selected_period_key = st.selectbox("Tampilkan Periode:", options=list(time_periods.keys()), index=1) # Default 90 hari
        days_to_show = time_periods[selected_period_key]
        plot_data_hist = ts_data.iloc[-min(days_to_show, len(ts_data)):]

        # Plot Historis
        fig_hist = px.line(
            plot_data_hist, x=plot_data_hist.index, y='Harga Aktual',
            labels={'index': 'Tanggal', 'Harga Aktual': 'Harga (Rp)'},
            template="plotly_white"
        )
        fig_hist.update_traces(line=dict(color='royalblue', width=2))
        fig_hist.update_layout(hovermode='x unified', height=450, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

        # Statisik Dasar
        st.markdown("---")
        st.markdown("##### Ringkasan Statistik")
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
             st.metric("Harga Terakhir", f"Rp {last_actual_value:,.0f}")
             st.metric("Harga Minimum", f"Rp {ts_data['Harga Aktual'].min():,.0f}")
        with col_stat2:
             st.metric("Harga Rata-rata", f"Rp {ts_data['Harga Aktual'].mean():,.0f}")
             st.metric("Harga Maksimum", f"Rp {ts_data['Harga Aktual'].max():,.0f}")


    with tab_pred:
        st.subheader(f"Prediksi Harga: {selected_commodity}")
        st.caption(f"Lokasi: {selected_location} | Prediksi untuk {days_to_predict} hari ke depan (hingga {selected_pred_date.strftime('%d %B %Y')})")

        if not predict_button:
            st.info("Tekan tombol '🔮 Prediksi...' di sidebar untuk melihat hasil forecast.")
        else:
            # Memuat model dan scaler saat tombol ditekan
            with st.spinner(f"Memuat model & scaler untuk {target_column_name}..."):
                model = load_keras_model(target_column_name)
                scaler = load_joblib_scaler(target_column_name)

            if model is None or scaler is None:
                st.error("Gagal memuat file model/scaler. Prediksi dibatalkan.")
            else:
                with st.spinner(f"Menjalankan prediksi untuk {days_to_predict} hari..."):
                    # Ambil data terakhir untuk input awal
                    last_sequence_actual = ts_data.iloc[-LOOK_BACK:]
                    if len(last_sequence_actual) < LOOK_BACK:
                         st.warning(f"Data historis tidak cukup ({len(last_sequence_actual)} hari) untuk look_back={LOOK_BACK}.")
                    else:
                        # Scaling input awal
                        initial_sequence_scaled = scaler.transform(last_sequence_actual.values)

                        # Lakukan prediksi rekursif
                        predictions_actual = predict_recursive(model, initial_sequence_scaled, days_to_predict, scaler)

                        # Buat tanggal prediksi
                        prediction_dates = pd.date_range(start=last_actual_date + timedelta(days=1), periods=days_to_predict, freq='D')

                        # Buat DataFrame hasil prediksi
                        pred_df = pd.DataFrame({'Harga Prediksi': predictions_actual}, index=prediction_dates)

                        # Tampilkan metrik ringkasan
                        st.markdown("##### Ringkasan Prediksi")
                        col_pred1, col_pred2, col_pred3 = st.columns(3)
                        with col_pred1:
                            st.metric(
                                label=f"Prediksi Besok ({prediction_dates[0].strftime('%d %b')})",
                                value=f"Rp {predictions_actual[0]:,.0f}",
                                delta=f"{predictions_actual[0] - last_actual_value:,.0f} vs Hari Ini"
                            )
                        with col_pred2:
                             if days_to_predict > 1:
                                 st.metric(
                                    label=f"Prediksi {days_to_predict} Hari ({prediction_dates[-1].strftime('%d %b')})",
                                    value=f"Rp {predictions_actual[-1]:,.0f}",
                                    delta=f"{predictions_actual[-1] - last_actual_value:,.0f} vs Hari Ini"
                                )
                        with col_pred3:
                            # Rata-rata prediksi
                             avg_pred = predictions_actual.mean()
                             st.metric(label=f"Rata-rata Prediksi {days_to_predict} Hari", value=f"Rp {avg_pred:,.0f}")

                        # Plot hasil prediksi vs historis
                        st.markdown("##### Grafik Prediksi")
                        fig_pred = go.Figure()
                        # Plot data historis (misal 30 hari terakhir)
                        hist_plot_data_pred = ts_data.iloc[-30:]
                        fig_pred.add_trace(go.Scatter(x=hist_plot_data_pred.index, y=hist_plot_data_pred['Harga Aktual'],
                                                    mode='lines', name='Harga Aktual', line=dict(color='royalblue')))
                        # Plot data prediksi
                        fig_pred.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Harga Prediksi'],
                                                    mode='lines+markers', name='Harga Prediksi', line=dict(color='firebrick', dash='dash')))

                        fig_pred.update_layout(
                            title=f'Historis & Prediksi Harga {days_to_predict} Hari',
                            xaxis_title='Tanggal', yaxis_title='Harga (Rp)',
                            hovermode='x unified', height=450, margin=dict(l=10, r=10, t=40, b=10),
                            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                        )
                        st.plotly_chart(fig_pred, use_container_width=True)

                        # Detail prediksi harian
                        with st.expander("Lihat Detail Prediksi Harian"):
                            pred_df_display = pred_df.copy()
                            pred_df_display.index = pred_df_display.index.strftime('%A, %d %b %Y')
                            pred_df_display['Harga Prediksi'] = pred_df_display['Harga Prediksi'].apply(lambda x: f"Rp {x:,.0f}")
                            st.dataframe(pred_df_display, use_container_width=True)

                        # Peringatan Akurasi
                        if days_to_predict > 7:
                            st.warning("⚠️ Akurasi prediksi cenderung menurun signifikan untuk horizon waktu yang lebih panjang (> 7 hari). Gunakan sebagai indikasi tren.")
                        st.caption(f"Prediksi dihasilkan menggunakan model LSTM univariate (look_back={LOOK_BACK}) secara rekursif.")

# Footer Aplikasi
st.divider()
st.caption(f"Sumber Data: PIHPS Nasional (Bank Indonesia) | Data terakhir: {last_data_date.strftime('%d %b %Y') if last_data_date else 'N/A'} | Model: LSTM Univariate | V0.2 WIP")
