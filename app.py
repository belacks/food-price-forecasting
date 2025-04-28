import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import joblib
import os
import re
import gdown # Library to download from Google Drive
import time

# --- Configuration ---
# !!! IMPORTANT: Replace with YOUR Google Drive Folder ID !!!
# Get this from the shareable link of the folder containing models/scalers
# URL: https://drive.google.com/drive/folders/YOUR_FOLDER_ID?usp=sharing
GDRIVE_FOLDER_ID = "101n5CICPuq6E02hPLFfcTVZo7F5XBdm9" # <--- *** PASTE YOUR ID HERE ***

# Path to the main historical data file (relative to the script)
# Assume it's in the same directory as app.py for deployment simplicity
DATA_FILE_PATH = 'data_pangan_jabodetabek_wide_imputed.parquet'

# Model/Scaler file naming pattern (must match how you saved them)
MODEL_FILENAME_PATTERN = "model_univar_{safe_name}.keras"
SCALER_FILENAME_PATTERN = "scaler_univar_{safe_name}.gz"

# LSTM Model Configuration (must match the trained models)
LOOK_BACK = 60 # The sequence length used during training

# --- Helper Functions ---

def safe_filename(name):
    """Cleans location/commodity name for use in filenames."""
    # Replace invalid characters (space, /, \) with _
    name = re.sub(r'[\\/ ]', '_', name)
    # Limit length if needed (optional)
    name = name[:80]
    return name

# Cache resource for expensive model/scaler loading
@st.cache_resource(ttl=3600) # Cache for 1 hour
def download_and_load_model(target_column_name):
    """Downloads and loads a specific Keras model from Google Drive."""
    safe_name = safe_filename(target_column_name)
    model_filename = MODEL_FILENAME_PATTERN.format(safe_name=safe_name)
    local_model_path = f"./{model_filename}" # Save locally in the container

    # Try loading locally first (if already downloaded/cached by Streamlit)
    if os.path.exists(local_model_path):
        try:
            print(f"Loading cached model: {local_model_path}")
            model = load_model(local_model_path)
            print("Model loaded successfully from local cache.")
            return model
        except Exception as e:
            print(f"Failed to load cached model {local_model_path}: {e}. Re-downloading...")

    # Construct Google Drive download URL using gdown format
    # Note: gdown needs the FOLDER ID for listing/downloading specific files within it.
    # A direct file ID approach might also work if you have those IDs.
    # This simplified approach assumes gdown can handle folder listing or direct file links.
    # A more robust approach might involve listing files via gdown and finding the correct one.
    # For simplicity here, we assume direct download or cached file exists.
    # **This part might need adjustment based on how gdown handles folder contents**
    # A common pattern is to get individual file IDs, but that's harder to manage.
    # Let's try a direct approach first, relying on cache/prior existence.
    # If direct download by name isn't feasible via gdown folder ID,
    # you might need to get individual file IDs.

    # Placeholder for a more robust download logic if needed:
    # file_id = find_file_id_in_gdrive_folder(GDRIVE_FOLDER_ID, model_filename)
    # if file_id:
    #     gdown.download(id=file_id, output=local_model_path, quiet=False)
    # else:
    #     st.error(f"Model file '{model_filename}' not found in GDrive folder.")
    #     return None

    # Simplified: Assume file must exist locally after potential download (relying on cache)
    # If you manually place models or if cache works across restarts
    if not os.path.exists(local_model_path):
         st.error(f"Model file '{local_model_path}' not found locally and download from GDrive folder ID is complex/not implemented directly here. Please ensure models are accessible.")
         # Try a placeholder gdown command - THIS MIGHT FAIL depending on gdown version and folder permissions
         try:
             print(f"Attempting to download model {model_filename} using gdown (may require file ID)...")
             # THIS IS A GUESS - gdown usually needs a file ID or direct link
             # Replace with actual file ID if you have it: gdown.download(id=YOUR_FILE_ID, ...)
             # Or manually ensure files are present in the deployment environment
             st.warning(f"Attempting simplified GDrive download for {model_filename}. This might fail. Consider providing direct file IDs or placing files in the repo if small enough.")
             # Example using folder ID - likely needs specific file ID:
             # gdown.download(id=GDRIVE_FOLDER_ID, output=local_model_path, quiet=False, fuzzy=True) # Fuzzy might help find by name
             raise NotImplementedError("Direct download by name from folder ID is unreliable with gdown. Use file IDs or alternative storage.")

         except Exception as gd_err:
             st.error(f"Failed to download/find model {model_filename} using gdown: {gd_err}")
             return None


    try:
        print(f"Loading model: {local_model_path}")
        model = load_model(local_model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading Keras model from {local_model_path}: {e}")
        return None


@st.cache_resource(ttl=3600)
def download_and_load_scaler(target_column_name):
    """Downloads and loads a specific joblib scaler from Google Drive."""
    safe_name = safe_filename(target_column_name)
    scaler_filename = SCALER_FILENAME_PATTERN.format(safe_name=safe_name)
    local_scaler_path = f"./{scaler_filename}"

    # Try loading locally first
    if os.path.exists(local_scaler_path):
        try:
            print(f"Loading cached scaler: {local_scaler_path}")
            scaler = joblib.load(local_scaler_path)
            print("Scaler loaded successfully from local cache.")
            return scaler
        except Exception as e:
            print(f"Failed to load cached scaler {local_scaler_path}: {e}. Re-downloading...")

    # Simplified download logic (same caveats as model download)
    if not os.path.exists(local_scaler_path):
        st.error(f"Scaler file '{local_scaler_path}' not found locally. Download from GDrive folder ID not implemented directly. Ensure files are accessible.")
        # Placeholder download attempt (likely needs specific file ID)
        try:
            print(f"Attempting to download scaler {scaler_filename} using gdown (may require file ID)...")
            st.warning(f"Attempting simplified GDrive download for {scaler_filename}. This might fail. Use file IDs or alternative storage.")
            raise NotImplementedError("Direct download by name from folder ID is unreliable with gdown. Use file IDs or alternative storage.")
        except Exception as gd_err:
             st.error(f"Failed to download/find scaler {scaler_filename} using gdown: {gd_err}")
             return None

    try:
        print(f"Loading scaler: {local_scaler_path}")
        scaler = joblib.load(local_scaler_path)
        print("Scaler loaded successfully.")
        return scaler
    except Exception as e:
        st.error(f"Error loading Scaler from {local_scaler_path}: {e}")
        return None

# Cache data loading
@st.cache_data(ttl=3600) # Cache data for 1 hour
def load_data(filepath):
    """Loads the main wide, imputed dataframe from Parquet."""
    print(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        st.error(f"Data file not found at {filepath}. Please ensure it's included in the deployment.")
        return None
    try:
        df = pd.read_parquet(filepath)
        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        st.error(f"Error loading data from {filepath}: {e}")
        return None

# --- Main App ---
st.set_page_config(layout="wide")
st.title("📈 Indonesian Food Price Predictor (Jabodetabek)")
st.markdown("""
Welcome! Select a location and commodity below to see the historical price trend
and get a price prediction for the next available day using a dedicated LSTM model.
""")

# Load the main dataframe
df_wide = load_data(DATA_FILE_PATH)

if df_wide is not None:
    # --- User Inputs ---
    st.sidebar.header("Select Options")

    # Get unique locations and commodities from column names
    # Assuming format 'Lokasi_NamaKomoditas'
    try:
        locations = sorted(list(set(col.split('_')[0] for col in df_wide.columns)))
        # Get commodities based on the FIRST location (assuming they are mostly the same)
        # A more robust way would be to parse all columns or have a predefined list
        commodities = sorted(list(set(
            "_".join(col.split('_')[1:]) for col in df_wide.columns if col.startswith(locations[0])
            )))
    except Exception as e:
        st.error(f"Could not parse locations/commodities from DataFrame columns: {e}")
        st.warning("Using placeholder lists. Please check column naming format.")
        locations = ["Jakarta Pusat", "Kota Bogor", "Kota Bekasi", "Kota Depok", "Kota Tangerang"]
        commodities = ["Beras Kualitas Medium I", "Cabai Merah Keriting", "Daging Ayam Ras Segar"] # Example

    selected_location = st.sidebar.selectbox("Select Location:", locations)
    selected_commodity = st.sidebar.selectbox("Select Commodity:", commodities)

    predict_button = st.sidebar.button("📊 Predict Price", type="primary")

    # --- Display Area ---
    st.header(f"Analysis for: {selected_commodity} in {selected_location}")

    target_column_name = f"{selected_location}_{selected_commodity}"

    if target_column_name not in df_wide.columns:
        st.error(f"Data for '{selected_commodity}' in '{selected_location}' is not available in the dataset. Please check column names or data processing.")
    else:
        # --- Historical Data Plot ---
        st.subheader("Historical Price Trend")
        historical_data = df_wide[[target_column_name]].copy()
        historical_data.rename(columns={target_column_name: 'Price'}, inplace=True)

        fig_hist = px.line(historical_data, x=historical_data.index, y='Price',
                           title=f"Historical Price of {selected_commodity} in {selected_location}",
                           labels={'index': 'Date', 'Price': 'Price (Rp)'})
        fig_hist.update_layout(hovermode='x unified')
        st.plotly_chart(fig_hist, use_container_width=True)

        # --- Prediction Logic ---
        if predict_button:
            st.subheader("Price Prediction")
            with st.spinner(f"Loading model and predicting for {selected_commodity} in {selected_location}..."):
                # 1. Load Model and Scaler (Cached)
                model = download_and_load_model(target_column_name)
                scaler = download_and_load_scaler(target_column_name)

                if model is None or scaler is None:
                    st.error("Could not load the required model or scaler. Prediction aborted.")
                else:
                    # 2. Get Last 'look_back' Data Points
                    last_known_data = df_wide[[target_column_name]].iloc[-LOOK_BACK:] # Select column and last N rows

                    if len(last_known_data) < LOOK_BACK:
                        st.warning(f"Not enough historical data ({len(last_known_data)} points) available to make a prediction using look_back={LOOK_BACK}. Need at least {LOOK_BACK}.")
                    else:
                        # 3. Scale the input data
                        input_data_scaled = scaler.transform(last_known_data.values) # .values gets numpy array

                        # 4. Reshape for LSTM: [samples, timesteps, features]
                        input_data_reshaped = input_data_scaled.reshape(1, LOOK_BACK, 1) # 1 sample, LOOK_BACK timesteps, 1 feature

                        # 5. Predict
                        try:
                            predicted_price_scaled = model.predict(input_data_reshaped)

                            # 6. Inverse Transform
                            predicted_price_actual = scaler.inverse_transform(predicted_price_scaled)

                            # Extract the single predicted value
                            prediction_value = predicted_price_actual[0, 0]
                            last_actual_date = last_known_data.index[-1].strftime('%d %b %Y')
                            last_actual_value = last_known_data.iloc[-1, 0]
                            next_day_date = (last_known_data.index[-1] + pd.Timedelta(days=1)).strftime('%d %b %Y')

                            st.metric(
                                label=f"Predicted Price for {next_day_date}",
                                value=f"Rp {prediction_value:,.0f}",
                                delta=f"{prediction_value - last_actual_value:,.0f} (Rp) vs {last_actual_date} (Rp {last_actual_value:,.0f})",
                                delta_color="normal" # or "inverse" or "off"
                                )

                            # (Optional) Add a small chart showing last few actuals + prediction
                            plot_df = historical_data.iloc[-10:].copy() # Last 10 actual points
                            # Create a date for the prediction point
                            pred_date = plot_df.index[-1] + pd.Timedelta(days=1)
                            # Add prediction to df for plotting
                            pred_series = pd.Series([prediction_value], index=[pred_date])
                            pred_df = pd.DataFrame(pred_series, columns=['Price'])

                            fig_pred = go.Figure()
                            fig_pred.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Price'], mode='lines+markers', name='Actual Price'))
                            fig_pred.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Price'], mode='markers', name='Predicted Price', marker=dict(color='red', size=10)))
                            fig_pred.update_layout(title="Recent Actual vs. Next Day Prediction", xaxis_title="Date", yaxis_title="Price (Rp)", height=300)
                            st.plotly_chart(fig_pred, use_container_width=True)


                        except Exception as pred_err:
                            st.error(f"An error occurred during prediction: {pred_err}")

        else:
            st.info("Click the 'Predict Price' button in the sidebar to get the forecast.")

else:
    st.error("Failed to load the main data. Application cannot start.")

st.sidebar.markdown("---")
st.sidebar.markdown("Created as a Portfolio Project")
# Add your name or GitHub link if you like
# st.sidebar.markdown("[Your Name/GitHub](link)")
