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
# import gdown # Commented out - using local folder approach for now
import time

# --- Configuration ---
# Folder containing the saved models and scalers (relative to app.py)
MODELS_BASE_FOLDER = "models" # Make sure this folder exists in your repo/deployment

# Path to the main historical data file
DATA_FILE_PATH = 'data_pangan_jabodetabek_wide_imputed.parquet'

# Model/Scaler file naming pattern (MUST match how you saved them)
MODEL_FILENAME_PATTERN = "model_univar_{safe_name}.keras"
SCALER_FILENAME_PATTERN = "scaler_univar_{safe_name}.gz"

# LSTM Model Configuration (MUST match the trained models)
LOOK_BACK = 60 # The sequence length used during training

# Maximum forecast horizon allowed (for multi-step)
MAX_HORIZON = 30

# --- Helper Functions ---

def safe_filename(name):
    """Cleans location/commodity name for use in filenames."""
    name = re.sub(r'[\\/ ]', '_', name) # Replace space, /, \ with _
    name = name[:80] # Limit length
    return name

# Cache resource for expensive model/scaler loading
@st.cache_resource(ttl=3600) # Cache for 1 hour
def load_keras_model(target_column_name):
    """Loads a specific Keras model from the models subfolder."""
    safe_name = safe_filename(target_column_name)
    model_filename = MODEL_FILENAME_PATTERN.format(safe_name=safe_name)
    model_path = os.path.join(MODELS_BASE_FOLDER, model_filename)

    if os.path.exists(model_path):
        try:
            print(f"Loading model: {model_path}")
            model = load_model(model_path)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            st.error(f"Error loading Keras model from {model_path}: {e}")
            return None
    else:
        st.error(f"Model file '{model_filename}' not found in '{MODELS_BASE_FOLDER}'.")
        return None

@st.cache_resource(ttl=3600)
def load_joblib_scaler(target_column_name):
    """Loads a specific joblib scaler from the models subfolder."""
    safe_name = safe_filename(target_column_name)
    scaler_filename = SCALER_FILENAME_PATTERN.format(safe_name=safe_name)
    scaler_path = os.path.join(MODELS_BASE_FOLDER, scaler_filename)

    if os.path.exists(scaler_path):
        try:
            print(f"Loading scaler: {scaler_path}")
            scaler = joblib.load(scaler_path)
            print("Scaler loaded successfully.")
            return scaler
        except Exception as e:
            st.error(f"Error loading Scaler from {scaler_path}: {e}")
            return None
    else:
        st.error(f"Scaler file '{scaler_filename}' not found in '{MODELS_BASE_FOLDER}'.")
        return None

# Cache data loading
@st.cache_data(ttl=3600) # Cache data for 1 hour
def load_data(filepath):
    """Loads the main wide, imputed dataframe from Parquet."""
    print(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        st.error(f"Data file not found at {filepath}. Please ensure it's included.")
        return None, None, None # Return None for df, locations, commodities
    try:
        df = pd.read_parquet(filepath)
        df.index = pd.to_datetime(df.index) # Ensure index is datetime
        print("Data loaded successfully.")
        # Parse locations and commodities robustly
        locations = sorted(list(set(col.split('_')[0] for col in df.columns)))
        # Use a reference location to get commodity list (assuming consistent)
        ref_loc = locations[0] if locations else ""
        commodities = sorted(list(set(
            "_".join(col.split('_')[1:]) for col in df.columns if col.startswith(ref_loc + '_')
            )))
        return df, locations, commodities
    except Exception as e:
        st.error(f"Error loading or parsing data from {filepath}: {e}")
        return None, None, None

def predict_recursive(model, initial_sequence_scaled, n_steps_out, scaler):
    """
    Performs recursive multi-step forecasting.
    Args:
        model: Trained Keras LSTM model.
        initial_sequence_scaled (np.array): The last 'look_back' actual data points, scaled. Shape (look_back, 1)
        n_steps_out (int): Number of future steps to predict.
        scaler: Fitted scaler object for inverse transforming.
    Returns:
        np.array: Array of predicted values in actual scale. Shape (n_steps_out,)
    """
    current_sequence = initial_sequence_scaled.copy().reshape(1, LOOK_BACK, 1) # Reshape for model input
    predictions_scaled = []

    for _ in range(n_steps_out):
        # Predict next step
        next_pred_scaled = model.predict(current_sequence, verbose=0)[0, 0] # Get single scalar prediction
        predictions_scaled.append(next_pred_scaled)

        # Update the sequence: remove oldest, append newest prediction
        # Reshape prediction to (1, 1) before vstack
        next_pred_scaled_reshaped = np.array([[next_pred_scaled]])
        current_sequence = np.vstack((current_sequence[0, 1:, :], next_pred_scaled_reshaped ))
        # Reshape back for next prediction
        current_sequence = current_sequence.reshape(1, LOOK_BACK, 1)


    # Inverse transform all predictions at once
    predictions_actual = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))
    return predictions_actual.flatten() # Return as 1D array

# --- Streamlit App UI and Logic ---

st.set_page_config(page_title="Harga Pangan Jabodetabek", page_icon="📈", layout="wide")

# --- Load Data ---
# This is now the first crucial step
df_wide, locations, commodities = load_data(DATA_FILE_PATH)

# --- App Title and Description ---
st.title("📈 Prediksi Harga Pangan Strategis - Jabodetabek")
st.markdown("Analisis historis dan prediksi harga harian menggunakan model LSTM.")
st.warning("⚠️ **Work In Progress:** Model dan fitur masih dalam pengembangan. Prediksi bersifat eksperimental.")

# Proceed only if data is loaded successfully
if df_wide is not None and locations and commodities:

    # --- Sidebar for User Inputs ---
    st.sidebar.header("📌 Pilih Lokasi & Komoditas")
    selected_location = st.sidebar.selectbox("Lokasi:", locations, key="loc_select")
    selected_commodity = st.sidebar.selectbox("Komoditas:", commodities, key="com_select")

    st.sidebar.header("🗓️ Horizon Prediksi")
    days_to_predict = st.sidebar.slider(
        "Prediksi untuk berapa hari ke depan?",
        min_value=1,
        max_value=MAX_HORIZON,
        value=7, # Default 7 hari
        step=1,
        help=f"Pilih jumlah hari (1-{MAX_HORIZON}) yang ingin diprediksi. Akurasi menurun untuk horizon yang lebih panjang."
    )

    # --- Main Area ---
    target_column_name = f"{selected_location}_{selected_commodity}"

    # Check if the selected combination exists
    if target_column_name not in df_wide.columns:
        st.error(f"Kombinasi data untuk '{selected_commodity}' di '{selected_location}' tidak ditemukan.")
    else:
        st.header(f"{selected_commodity} di {selected_location}")

        # Get the actual time series data
        ts_data = df_wide[[target_column_name]].copy()
        ts_data.rename(columns={target_column_name: 'Harga Aktual'}, inplace=True)
        last_actual_date = ts_data.index[-1]
        last_actual_value = ts_data.iloc[-1, 0]

        # --- Display Historical Data ---
        st.subheader("Tren Harga Historis")
        fig_hist = px.line(ts_data.iloc[-180:], # Tampilkan 6 bulan terakhir
                          x=ts_data.iloc[-180:].index, y='Harga Aktual',
                          labels={'index': 'Tanggal', 'Harga Aktual': 'Harga (Rp)'},
                          template="plotly_white")
        fig_hist.update_traces(line=dict(color='royalblue', width=2))
        fig_hist.update_layout(hovermode='x unified', height=400, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

        # --- Prediction Section ---
        st.subheader(f"Prediksi Harga {days_to_predict} Hari ke Depan")

        # Load model and scaler based on selection
        with st.spinner(f"Memuat model & scaler untuk {target_column_name}..."):
            model = load_keras_model(target_column_name)
            scaler = load_joblib_scaler(target_column_name)

        if model is None or scaler is None:
            st.error("Gagal memuat model atau scaler yang diperlukan.")
        else:
            if st.button(f"🚀 Jalankan Prediksi {days_to_predict} Hari", key="predict_btn", type="primary"):
                with st.spinner("Melakukan prediksi..."):
                    # Prepare last sequence for input
                    last_sequence_actual = ts_data.iloc[-LOOK_BACK:]

                    if len(last_sequence_actual) < LOOK_BACK:
                        st.warning(f"Data historis tidak cukup ({len(last_sequence_actual)} hari) untuk look_back={LOOK_BACK}.")
                    else:
                        # Scale the initial sequence
                        initial_sequence_scaled = scaler.transform(last_sequence_actual.values)

                        # Perform recursive prediction
                        start_pred_time = time.time()
                        predictions_actual = predict_recursive(model, initial_sequence_scaled, days_to_predict, scaler)
                        pred_duration = time.time() - start_pred_time
                        print(f"Prediction took {pred_duration:.2f} seconds.")

                        # Create prediction dates
                        prediction_dates = pd.date_range(start=last_actual_date + pd.Timedelta(days=1), periods=days_to_predict, freq='D')

                        # Create prediction DataFrame
                        pred_df = pd.DataFrame({'Harga Prediksi': predictions_actual}, index=prediction_dates)

                        # --- Display Results ---
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                label=f"Prediksi Besok ({prediction_dates[0].strftime('%d %b')})",
                                value=f"Rp {predictions_actual[0]:,.0f}",
                                delta=f"{predictions_actual[0] - last_actual_value:,.0f} vs {last_actual_date.strftime('%d %b')}"
                            )
                        with col2:
                             if days_to_predict > 1:
                                 st.metric(
                                    label=f"Prediksi {days_to_predict} Hari ({prediction_dates[-1].strftime('%d %b')})",
                                    value=f"Rp {predictions_actual[-1]:,.0f}",
                                    delta=f"{predictions_actual[-1] - last_actual_value:,.0f} vs {last_actual_date.strftime('%d %b')}"
                                )

                        # Plot historical + prediction
                        fig_pred = go.Figure()
                        # Historical (zoom ke 30 hari terakhir + prediksi)
                        hist_plot_data = ts_data.iloc[-(LOOK_BACK//2):] # Tampilkan separuh look_back history
                        fig_pred.add_trace(go.Scatter(x=hist_plot_data.index, y=hist_plot_data['Harga Aktual'],
                                                    mode='lines', name='Harga Aktual', line=dict(color='royalblue')))
                        # Prediction
                        fig_pred.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Harga Prediksi'],
                                                    mode='lines+markers', name='Harga Prediksi', line=dict(color='firebrick', dash='dash')))

                        fig_pred.update_layout(
                            title=f'Prediksi Harga {days_to_predict} Hari ke Depan',
                            xaxis_title='Tanggal', yaxis_title='Harga (Rp)',
                            hovermode='x unified', height=450, margin=dict(l=10, r=10, t=40, b=10),
                            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                        )
                        st.plotly_chart(fig_pred, use_container_width=True)

                        # Tampilkan tabel prediksi (opsional)
                        with st.expander("Lihat Detail Prediksi Harian"):
                             pred_df_display = pred_df.copy()
                             pred_df_display.index = pred_df_display.index.strftime('%A, %d %b %Y')
                             pred_df_display['Harga Prediksi'] = pred_df_display['Harga Prediksi'].apply(lambda x: f"Rp {x:,.0f}")
                             st.dataframe(pred_df_display, use_container_width=True)

                        st.caption(f"Prediksi dihasilkan menggunakan model LSTM univariate dengan look_back={LOOK_BACK} hari.")
                        if days_to_predict > 7:
                             st.warning("⚠️ Akurasi prediksi cenderung menurun signifikan untuk horizon waktu yang lebih panjang (> 7 hari). Gunakan sebagai indikasi tren.")


else:
    st.error("Gagal memuat data utama. Aplikasi tidak bisa berjalan.")
    st.info("Pastikan file `data_pangan_jabodetabek_wide_imputed.parquet` ada di direktori aplikasi.")

# Footer
st.markdown("---")
st.caption("Sumber Data: PIHPS Nasional (Bank Indonesia) | Model: LSTM Univariate | Proyek Portofolio oleh Muhammad Ali Ashari")
