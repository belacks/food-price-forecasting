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
    """Loads a specific Keras model from the models directory."""
    safe_name = safe_filename(target_column_name)
    model_filename = MODEL_FILENAME_PATTERN.format(safe_name=safe_name)
    
    # First check in the models directory
    model_path_in_folder = os.path.join("models", model_filename)
    local_model_path = f"./{model_filename}"  # Fallback to root directory
    
    # Check if model exists in the models folder
    if os.path.exists(model_path_in_folder):
        try:
            print(f"Loading model from models directory: {model_path_in_folder}")
            model = load_model(model_path_in_folder)
            print("Model loaded successfully from models directory.")
            return model
        except Exception as e:
            print(f"Failed to load model from models directory {model_path_in_folder}: {e}. Trying root directory...")
    
    # Try loading from root directory as fallback (if already downloaded/cached by Streamlit)
    if os.path.exists(local_model_path):
        try:
            print(f"Loading model from root directory: {local_model_path}")
            model = load_model(local_model_path)
            print("Model loaded successfully from root directory.")
            return model
        except Exception as e:
            print(f"Failed to load model from root directory {local_model_path}: {e}.")
    
    # If we got here, we couldn't find or load the model
    st.error(f"Model file '{model_filename}' not found in the models directory or root directory. Please ensure the model file exists in the 'models' folder of your repository.")
    return None


@st.cache_resource(ttl=3600)
def download_and_load_scaler(target_column_name):
    """Loads a specific joblib scaler from the models directory."""
    safe_name = safe_filename(target_column_name)
    scaler_filename = SCALER_FILENAME_PATTERN.format(safe_name=safe_name)
    
    # First check in the models directory
    scaler_path_in_folder = os.path.join("models", scaler_filename)
    local_scaler_path = f"./{scaler_filename}"  # Fallback to root directory
    
    # Check if scaler exists in the models folder
    if os.path.exists(scaler_path_in_folder):
        try:
            print(f"Loading scaler from models directory: {scaler_path_in_folder}")
            scaler = joblib.load(scaler_path_in_folder)
            print("Scaler loaded successfully from models directory.")
            return scaler
        except Exception as e:
            print(f"Failed to load scaler from models directory {scaler_path_in_folder}: {e}. Trying root directory...")
    
    # Try loading from root directory as fallback (if already downloaded/cached by Streamlit)
    if os.path.exists(local_scaler_path):
        try:
            print(f"Loading scaler from root directory: {local_scaler_path}")
            scaler = joblib.load(local_scaler_path)
            print("Scaler loaded successfully from root directory.")
            return scaler
        except Exception as e:
            print(f"Failed to load scaler from root directory {local_scaler_path}: {e}.")
    
    # If we got here, we couldn't find or load the scaler
    st.error(f"Scaler file '{scaler_filename}' not found in the models directory or root directory. Please ensure the scaler file exists in the 'models' folder of your repository.")
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
# Set page configuration with wider layout and custom title/icon
st.set_page_config(
    page_title="Indonesian Food Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for storing prediction results
if 'has_predicted' not in st.session_state:
    st.session_state.has_predicted = False
if 'prediction_value' not in st.session_state:
    st.session_state.prediction_value = None
if 'last_actual_value' not in st.session_state:
    st.session_state.last_actual_value = None
if 'last_actual_date' not in st.session_state:
    st.session_state.last_actual_date = None
if 'pred_date' not in st.session_state:
    st.session_state.pred_date = None
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'plot_df' not in st.session_state:
    st.session_state.plot_df = None
if 'multi_day_predictions' not in st.session_state:
    st.session_state.multi_day_predictions = None
if 'selected_location' not in st.session_state:
    st.session_state.selected_location = None
if 'selected_commodity' not in st.session_state:
    st.session_state.selected_commodity = None

# Add custom CSS for better visual appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1.5rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1E88E5;
    }
    .info-text {
        color: #555;
        font-size: 0.9rem;
    }
    .caption-text {
        color: #777;
        font-size: 0.8rem;
        font-style: italic;
    }
    .highlight {
        background-color: #e6f7ff;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title Section with Logo and Introduction
st.markdown("<div class='main-header'>📊 Indonesian Food Price Predictor</div>", unsafe_allow_html=True)
st.markdown("### Jabodetabek Region Market Analysis & Forecasting Tool")

with st.expander("ℹ️ About This Application", expanded=False):
    st.markdown("""
    This application uses advanced machine learning (LSTM neural networks) to predict food commodity prices 
    in the Jakarta, Bogor, Depok, Tangerang, and Bekasi (Jabodetabek) region of Indonesia. 
    
    **Features:**
    - Historical price visualization for various food commodities
    - Next-day price predictions using trained LSTM models
    - Location-specific analysis across the Jabodetabek region
    
    **How it works:**
    1. Select your location and commodity of interest
    2. View the historical price trend 
    3. Generate a prediction for the next market day
    
    **Note on predictions:** The forecasts are based on historical patterns and may not account for 
    unexpected market disruptions, policy changes, or extreme weather events.
    """)

# Load the main dataframe with a loading indicator
with st.spinner("📂 Loading historical price data..."):
    df_wide = load_data(DATA_FILE_PATH)

# Create a better sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Flag_of_Indonesia.svg/800px-Flag_of_Indonesia.svg.png", width=250)
st.sidebar.title("🛒 Selection Panel")

# Main content based on data availability
if df_wide is not None:
    # Parse locations and commodities from column names
    try:
        locations = sorted(list(set(col.split('_')[0] for col in df_wide.columns)))
        commodities = sorted(list(set(
            "_".join(col.split('_')[1:]) for col in df_wide.columns if col.startswith(locations[0])
        )))
    except Exception as e:
        st.sidebar.error("⚠️ Column parsing error. Using default values.")
        locations = ["Jakarta Pusat", "Kota Bogor", "Kota Bekasi", "Kota Depok", "Kota Tangerang"]
        commodities = ["Beras Kualitas Medium I", "Cabai Merah Keriting", "Daging Ayam Ras Segar"]

    # Enhanced sidebar selection area
    st.sidebar.markdown("### 📍 Select Region")
    st.sidebar.caption("Choose a specific area in Jabodetabek:")
    selected_location = st.sidebar.selectbox(
        "Location",
        locations,
        index=0,
        help="Select the market location you're interested in"
    )
    
    st.sidebar.markdown("### 🥕 Select Food Item")
    st.sidebar.caption("Choose a food commodity:")
    selected_commodity = st.sidebar.selectbox(
        "Commodity",
        commodities,
        index=0,
        help="Select the food item you want to analyze"
    )
    
    st.sidebar.markdown("### 🗓️ Forecast Horizon")
    st.sidebar.caption("How many days into the future do you want to predict?")
    
    # Calculate reasonable date range (today plus 1-30 days)
    today = pd.Timestamp.now().date()
    min_date = today + pd.Timedelta(days=1)
    max_date = today + pd.Timedelta(days=30)
    
    # Let user select a target date
    forecast_end_date = st.sidebar.date_input(
        "Select forecast end date:",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
        help="The model will predict prices from tomorrow until this date"
    )
    
    # Calculate how many days ahead to predict
    days_ahead = (pd.Timestamp(forecast_end_date) - pd.Timestamp(today)).days
    days_ahead = max(1, min(days_ahead, 30))  # Limit to between 1 and 30 days
    
    if days_ahead > 1:
        st.sidebar.caption(f"Forecasting {days_ahead} days ahead")
    else:
        st.sidebar.caption("Single day forecast")
    
    # Add a divider before the predict button
    st.sidebar.divider()
    
    # Make the predict button more prominent
    forecast_button = st.sidebar.button(
        f"🔮 Generate {'Multi-Day' if days_ahead > 1 else 'Next-Day'} Forecast", 
        type="primary",
        use_container_width=True,
        help=f"Click to predict prices for the next {days_ahead} day(s)"
    )
    
    # Clear prediction state if location or commodity changes
    current_selection = f"{selected_location}_{selected_commodity}"
    if 'last_selection' in st.session_state and st.session_state.last_selection != current_selection:
        # Clear previous predictions when selection changes
        if 'has_predicted' in st.session_state:
            st.session_state.has_predicted = False
        if 'multi_day_predictions' in st.session_state:
            st.session_state.multi_day_predictions = None
    
    # Update last selection
    st.session_state.last_selection = current_selection
    
    # Function to save prediction results to session state
    def save_prediction_to_state(prediction_value, last_actual_value, last_actual_date, pred_date, historical_data, plot_df, selected_location, selected_commodity):
        st.session_state.has_predicted = True
        st.session_state.prediction_value = prediction_value
        st.session_state.last_actual_value = last_actual_value
        st.session_state.last_actual_date = last_actual_date
        st.session_state.pred_date = pred_date
        st.session_state.historical_data = historical_data
        st.session_state.plot_df = plot_df
        st.session_state.selected_location = selected_location
        st.session_state.selected_commodity = selected_commodity
    
    # Information box in sidebar
    with st.sidebar.expander("ℹ️ How accurate are predictions?"):
        st.markdown("""
        The predictions are based on LSTM (Long Short-Term Memory) neural networks trained on 
        historical price data. While generally reliable for short-term forecasts, unexpected 
        events (weather, supply chain disruptions, policy changes) can affect actual prices.
        
        The model uses the last 60 days of price data to predict the next day's price.
        """)
    
    # Target column construction
    target_column_name = f"{selected_location}_{selected_commodity}"
    
    # Main content area
    if target_column_name not in df_wide.columns:
        st.error(f"⚠️ Data for '{selected_commodity}' in '{selected_location}' is not available. Please select a different combination.")
    else:
        # Create tabs for different views
        tab1, tab2 = st.tabs(["📈 Historical Analysis", "🔮 Price Prediction"])
        
        # Historical Data Tab
        with tab1:
            st.markdown(f"<div class='sub-header'>Historical Prices: {selected_commodity}</div>", unsafe_allow_html=True)
            st.caption(f"Location: {selected_location} | Data Updated: {df_wide.index[-1].strftime('%d %B %Y')}")
            
            # Extract historical data
            historical_data = df_wide[[target_column_name]].copy()
            historical_data.rename(columns={target_column_name: 'Price'}, inplace=True)
            
            # Create columns for metrics and insights
            col1, col2, col3 = st.columns(3)
            
            # Calculate some basic statistics
            current_price = historical_data['Price'].iloc[-1]
            avg_price = historical_data['Price'].mean()
            price_change = historical_data['Price'].iloc[-1] - historical_data['Price'].iloc[-30]
            price_pct_change = (price_change / historical_data['Price'].iloc[-30]) * 100
            
            # Display metrics
            with col1:
                st.metric(
                    label="Current Price",
                    value=f"Rp {current_price:,.0f}",
                    delta=None
                )
                
            with col2:
                st.metric(
                    label="30-Day Change",
                    value=f"Rp {abs(price_change):,.0f}",
                    delta=f"{price_pct_change:.1f}%",
                    delta_color="inverse" if price_change < 0 else "normal"
                )
                
            with col3:
                st.metric(
                    label="Historical Average",
                    value=f"Rp {avg_price:,.0f}",
                    delta=f"{((current_price - avg_price) / avg_price) * 100:.1f}% vs avg",
                    delta_color="inverse" if current_price < avg_price else "normal"
                )
            
            # Historical chart with enhanced features
            st.markdown("#### Price History")
            
            # Create date range selector
            time_periods = {
                "Last 30 Days": 30,
                "Last 90 Days": 90,
                "Last 6 Months": 180,
                "Last Year": 365,
                "All Time": len(historical_data)
            }
            
            selected_period = st.select_slider(
                "Select Time Period",
                options=list(time_periods.keys()),
                value="Last 90 Days"
            )
            
            # Filter data based on selected time period
            days_to_show = time_periods[selected_period]
            plot_data = historical_data.iloc[-min(days_to_show, len(historical_data)):]
            
            # Enhanced historical plot
            fig_hist = px.line(
                plot_data, 
                x=plot_data.index, 
                y='Price',
                title=None,
                labels={'index': 'Date', 'Price': 'Price (Rp)'},
                template="plotly_white"
            )
            
            fig_hist.update_traces(line=dict(color='#1E88E5', width=2))
            fig_hist.update_layout(
                hovermode='x unified',
                hoverlabel=dict(bgcolor='white', font_size=12),
                height=450,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(showgrid=True, gridcolor='#eee'),
                yaxis=dict(showgrid=True, gridcolor='#eee')
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Add context about the chart
            with st.expander("📊 Understanding This Chart"):
                st.markdown("""
                This chart shows the historical price trend for the selected commodity in the chosen location. 
                
                **Key points to note:**
                - **Rising trends** may indicate increasing demand, reduced supply, or seasonal factors
                - **Falling trends** often suggest increased supply, lower demand, or post-seasonal normalization
                - **Sharp spikes** typically represent sudden supply disruptions or demand surges
                
                Use the time period selector above to zoom in on specific timeframes.
                """)
                
            # Show data statistics in a clean format
            with st.expander("📑 Statistical Summary"):
                col_stats1, col_stats2 = st.columns(2)
                
                with col_stats1:
                    st.markdown("##### Basic Statistics")
                    stats_df = pd.DataFrame({
                        'Metric': ['Mean (Average)', 'Median', 'Minimum', 'Maximum', 'Standard Deviation'],
                        'Value': [
                            f"Rp {historical_data['Price'].mean():,.0f}",
                            f"Rp {historical_data['Price'].median():,.0f}",
                            f"Rp {historical_data['Price'].min():,.0f}",
                            f"Rp {historical_data['Price'].max():,.0f}",
                            f"Rp {historical_data['Price'].std():,.0f}"
                        ]
                    })
                    st.dataframe(stats_df, hide_index=True, use_container_width=True)
                
                with col_stats2:
                    st.markdown("##### Recent Trend")
                    # Calculate monthly averages for the last few months
                    monthly_avg = historical_data.resample('M').mean()
                    monthly_avg = monthly_avg.tail(3)
                    monthly_avg.index = monthly_avg.index.strftime('%b %Y')
                    monthly_avg.rename(columns={'Price': 'Average Price'}, inplace=True)
                    st.dataframe(monthly_avg, use_container_width=True)
        
        # Prediction Tab
        with tab2:
            st.markdown(f"<div class='sub-header'>Price Prediction: {selected_commodity}</div>", unsafe_allow_html=True)
            st.caption(f"Location: {selected_location} | Prediction Generated: {pd.Timestamp.now().strftime('%d %B %Y, %H:%M')}")
            
            if not predict_button:
                # Show placeholder content when prediction hasn't been run
                st.markdown("### ⏳ Ready for Prediction")
                st.info("Click the '🔮 Generate Price Prediction' button in the sidebar to forecast the next available price.")
                
                # Add an image or visualization to make the empty state more appealing
                placeholder_cols = st.columns([1, 2, 1])
                with placeholder_cols[1]:
                    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/73/Flat_tick_icon.svg/1200px-Flat_tick_icon.svg.png", width=100)
                    st.markdown("<p style='text-align: center'>LSTM model ready for prediction</p>", unsafe_allow_html=True)
            else:
                # Show a progress indicator during prediction
                with st.spinner("🔮 Crystal ball warming up... Running LSTM prediction model"):
                    # 1. Load Model and Scaler (Cached)
                    model = download_and_load_model(target_column_name)
                    scaler = download_and_load_scaler(target_column_name)
                    
                    if model is None or scaler is None:
                        st.error("🚫 Required model or scaler could not be loaded. Please try a different commodity/location combination.")
                    else:
                        # 2. Get Last 'look_back' Data Points
                        last_known_data = df_wide[[target_column_name]].iloc[-LOOK_BACK:]
                        
                        if len(last_known_data) < LOOK_BACK:
                            st.warning(f"⚠️ Not enough historical data available ({len(last_known_data)} points). Need at least {LOOK_BACK} data points for prediction.")
                        else:
                            # 3. Scale the input data
                            input_data_scaled = scaler.transform(last_known_data.values)
                            
                            # 4. Reshape for LSTM: [samples, timesteps, features]
                            input_data_reshaped = input_data_scaled.reshape(1, LOOK_BACK, 1)
                            
                            # 5. Predict
                            try:
                                predicted_price_scaled = model.predict(input_data_reshaped)
                                
                                # 6. Inverse Transform
                                predicted_price_actual = scaler.inverse_transform(predicted_price_scaled)
                                
                                # Extract prediction info
                                prediction_value = predicted_price_actual[0, 0]
                                last_actual_date = last_known_data.index[-1].strftime('%d %b %Y')
                                last_actual_value = last_known_data.iloc[-1, 0]
                                next_day_date = (last_known_data.index[-1] + pd.Timedelta(days=1)).strftime('%d %b %Y')
                                
                                # Success message
                                st.success(f"✅ Prediction complete for {next_day_date}")
                                
                                # Display prediction in a visually appealing box
                                st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
                                
                                prediction_cols = st.columns([2, 1])
                                
                                with prediction_cols[0]:
                                    st.markdown(f"### Predicted Price for {next_day_date}")
                                    
                                    # Calculate the price change
                                    price_diff = prediction_value - last_actual_value
                                    pct_change = (price_diff / last_actual_value) * 100
                                    
                                    # Show the main prediction metric
                                    st.metric(
                                        label="Next Day Forecast",
                                        value=f"Rp {prediction_value:,.0f}",
                                        delta=f"{price_diff:,.0f} ({pct_change:.2f}%)",
                                        delta_color="normal" if price_diff >= 0 else "inverse"
                                    )
                                    
                                    # Add context about the prediction
                                    if abs(pct_change) < 1:
                                        trend_message = "The price is expected to remain relatively stable."
                                    elif price_diff > 0:
                                        trend_message = "The price is expected to increase."
                                    else:
                                        trend_message = "The price is expected to decrease."
                                        
                                    st.markdown(f"**Trend Analysis:** {trend_message}")
                                    
                                with prediction_cols[1]:
                                    st.markdown("#### Previous Price")
                                    st.metric(
                                        label=f"Price on {last_actual_date}",
                                        value=f"Rp {last_actual_value:,.0f}"
                                    )
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                                # Visualization of the prediction
                                st.markdown("#### Prediction Visualization")
                                
                                # Create dataframe for visualization with the last 14 days (more focused view)
                                plot_df = historical_data.iloc[-14:].copy()
                                pred_date = plot_df.index[-1] + pd.Timedelta(days=1)
                                pred_series = pd.Series([prediction_value], index=[pred_date])
                                pred_df = pd.DataFrame(pred_series, columns=['Price'])
                                
                                # Create an enhanced visualization
                                fig_pred = go.Figure()
                                
                                # Now let's add a date selector for multi-day predictions
                                st.markdown("#### Multi-Day Forecast")
                                st.markdown("Select a future date to predict prices until that date:")
                                
                                # Calculate reasonable date range (between 1 day and 30 days in the future)
                                min_date = pred_date
                                max_date = pred_date + pd.Timedelta(days=30)
                                
                                # Let user select a target date
                                target_date = st.date_input(
                                    "Select prediction end date:",
                                    value=min_date.date(),
                                    min_value=min_date.date(),
                                    max_value=max_date.date()
                                )
                                
                                # Convert to pandas timestamp for calculation
                                target_date = pd.Timestamp(target_date)
                                
                                # Calculate how many days ahead to predict
                                days_ahead = (target_date - pred_date).days + 1
                                days_ahead = max(1, min(days_ahead, 30))  # Limit to between 1 and 30 days
                                
                                if days_ahead > 1:
                                    st.info(f"Generating prediction for {days_ahead} days into the future. Note that accuracy decreases with longer forecast horizons.")
                                    
                                    # Show a progress bar for multi-day predictions
                                    multi_day_progress = st.progress(0)
                                    
                                    # Function to perform multi-day prediction
                                    def predict_multiple_days(initial_sequence, days, scaler, model):
                                        """Predict multiple days ahead recursively."""
                                        # Make a copy of the initial sequence to avoid modifying the original
                                        sequence = initial_sequence.copy()
                                        predictions = []
                                        dates = []
                                        
                                        # Get the last date from the original sequence
                                        last_date = plot_df.index[-1]
                                        
                                        for i in range(days):
                                            # Update progress bar
                                            multi_day_progress.progress((i + 1) / days)
                                            
                                            # Prepare the sequence for prediction (scale and reshape)
                                            sequence_scaled = scaler.transform(sequence)
                                            sequence_reshaped = sequence_scaled.reshape(1, LOOK_BACK, 1)
                                            
                                            # Make prediction
                                            pred_scaled = model.predict(sequence_reshaped, verbose=0)
                                            pred_value = scaler.inverse_transform(pred_scaled)[0, 0]
                                            
                                            # Add one day to last date
                                            next_date = last_date + pd.Timedelta(days=i+1)
                                            
                                            # Store the prediction and date
                                            predictions.append(pred_value)
                                            dates.append(next_date)
                                            
                                            # Update sequence for next prediction (remove oldest, add newest)
                                            sequence = np.vstack([sequence[1:], [[pred_value]]])
                                            
                                        return pd.DataFrame({'Price': predictions}, index=dates)
                                    
                                    # Make multi-day prediction
                                    multi_day_predictions = predict_multiple_days(
                                        last_known_data.values, 
                                        days_ahead,
                                        scaler,
                                        model
                                    )
                                    
                                    # For visualization
                                    pred_df = multi_day_predictions
                                
                                # Add confidence intervals as shaded area (wider for longer forecasts)
                                confidence_intervals = []
                                for i, (idx, row) in enumerate(pred_df.iterrows()):
                                    # Increase uncertainty for days further in the future
                                    uncertainty_factor = 0.03 + (i * 0.005)  # Starts at 3%, increases by 0.5% per day
                                    confidence_intervals.append({
                                        'date': idx,
                                        'lower': row['Price'] * (1 - uncertainty_factor),
                                        'upper': row['Price'] * (1 + uncertainty_factor)
                                    })
                                
                                ci_df = pd.DataFrame(confidence_intervals)
                                
                                # Add the historical line
                                fig_pred.add_trace(go.Scatter(
                                    x=plot_df.index, 
                                    y=plot_df['Price'], 
                                    mode='lines+markers', 
                                    name='Historical Price',
                                    line=dict(color='#1976D2', width=2),
                                    marker=dict(size=6)
                                ))
                                
                                # Add the prediction line
                                fig_pred.add_trace(go.Scatter(
                                    x=pred_df.index, 
                                    y=pred_df['Price'], 
                                    mode='lines+markers', 
                                    name='Predicted Price',
                                    line=dict(color='#D32F2F', width=2, dash='dot'),
                                    marker=dict(color='#D32F2F', size=8, symbol='circle')
                                ))
                                
                                # Add shaded confidence interval
                                fig_pred.add_trace(go.Scatter(
                                    x=ci_df['date'].tolist() + ci_df['date'].tolist()[::-1],
                                    y=ci_df['upper'].tolist() + ci_df['lower'].tolist()[::-1],
                                    fill='toself',
                                    fillcolor='rgba(255,0,0,0.1)',
                                    line=dict(color='rgba(255,0,0,0)'),
                                    hoverinfo='skip',
                                    showlegend=True,
                                    name='Prediction Confidence Interval'
                                ))
                                
                                # Update layout for better appearance
                                fig_pred.update_layout(
                                    title="Price Forecast" if days_ahead > 1 else "Next-Day Price Prediction",
                                    xaxis_title="Date",
                                    yaxis_title="Price (Rupiah)",
                                    hovermode="x unified",
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                                    height=500,
                                    template="plotly_white",
                                    margin=dict(l=10, r=10, t=50, b=30)
                                )
                                
                                # Add prediction summary table if multi-day prediction
                                if days_ahead > 1:
                                    st.markdown("#### Detailed Price Forecasts")
                                    
                                    # Create a readable summary table with key dates
                                    summary_data = []
                                    
                                    # Add tomorrow's prediction
                                    summary_data.append({
                                        "Date": pred_df.index[0].strftime("%d %b %Y"),
                                        "Predicted Price": f"Rp {pred_df['Price'].iloc[0]:,.0f}",
                                        "Change from Today": f"{((pred_df['Price'].iloc[0] - last_actual_value) / last_actual_value) * 100:.2f}%"
                                    })
                                    
                                    # Add a week ahead if available
                                    if days_ahead >= 7:
                                        week_idx = min(6, len(pred_df) - 1)  # Index 6 = 7 days (0-indexed)
                                        summary_data.append({
                                            "Date": pred_df.index[week_idx].strftime("%d %b %Y"),
                                            "Predicted Price": f"Rp {pred_df['Price'].iloc[week_idx]:,.0f}",
                                            "Change from Today": f"{((pred_df['Price'].iloc[week_idx] - last_actual_value) / last_actual_value) * 100:.2f}%"
                                        })
                                    
                                    # Add the last prediction
                                    if days_ahead > 1:
                                        summary_data.append({
                                            "Date": pred_df.index[-1].strftime("%d %b %Y"),
                                            "Predicted Price": f"Rp {pred_df['Price'].iloc[-1]:,.0f}",
                                            "Change from Today": f"{((pred_df['Price'].iloc[-1] - last_actual_value) / last_actual_value) * 100:.2f}%"
                                        })
                                    
                                    # Display as a table
                                    st.table(pd.DataFrame(summary_data))
                                    
                                    # Add download button for full prediction data
                                    csv = pred_df.reset_index().rename(columns={'index': 'Date'}).to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Full Forecast Data (CSV)",
                                        data=csv,
                                        file_name=f"forecast_{selected_commodity}_{selected_location}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                                        mime="text/csv",
                                    )
                                    
                                    # Add trend analysis
                                    last_prediction = pred_df['Price'].iloc[-1]
                                    total_change = last_prediction - last_actual_value
                                    percent_change = (total_change / last_actual_value) * 100
                                    
                                    trend_message = ""
                                    if percent_change > 5:
                                        trend_message = f"🔴 **Strong upward trend expected**: Price projected to increase by {percent_change:.1f}% over {days_ahead} days."
                                    elif percent_change > 1:
                                        trend_message = f"🟠 **Moderate upward trend expected**: Price projected to increase by {percent_change:.1f}% over {days_ahead} days."
                                    elif percent_change > -1:
                                        trend_message = f"🟢 **Stable price expected**: Price projected to change by only {percent_change:.1f}% over {days_ahead} days."
                                    elif percent_change > -5:
                                        trend_message = f"🔵 **Moderate downward trend expected**: Price projected to decrease by {abs(percent_change):.1f}% over {days_ahead} days."
                                    else:
                                        trend_message = f"🟣 **Strong downward trend expected**: Price projected to decrease by {abs(percent_change):.1f}% over {days_ahead} days."
                                    
                                    st.markdown("#### Price Trend Analysis")
                                    st.markdown(trend_message)
                                    
                                    st.warning("⚠️ **Forecast Accuracy Notice**: Predictions become less reliable the further into the future they extend. Use long-term forecasts as general trend indicators rather than exact price points.")
                                
                                st.plotly_chart(fig_pred, use_container_width=True)
                                
                                # Add context about the prediction
                                with st.expander("🔍 Understanding These Predictions"):
                                    st.markdown("""
                                    These predictions are generated using a Long Short-Term Memory (LSTM) neural network, 
                                    a type of deep learning model specialized in time series forecasting.
                                    
                                    **How it works:**
                                    - The model analyzes the last 60 days of price data
                                    - It identifies patterns, trends, and seasonal factors
                                    - It projects these patterns forward to estimate future prices
                                    
                                    **For multi-day forecasts:**
                                    - The model predicts one day at a time
                                    - Each prediction is fed back into the model to predict the next day
                                    - This is known as recursive or iterative forecasting
                                    
                                    **Limitations:**
                                    - The model cannot account for unexpected events (weather disasters, policy changes)
                                    - Accuracy generally decreases the further into the future we predict
                                    - Market interventions or unusual volatility may reduce prediction accuracy
                                    - Errors compound over time in multi-day forecasts
                                    
                                    **Visual Elements:**
                                    - The red dotted line shows predicted prices
                                    - The shaded area represents the confidence interval
                                    - Note that the confidence interval widens over time to reflect increasing uncertainty
                                    """)
                                
                            except Exception as pred_err:
                                st.error(f"❌ Prediction Error: {pred_err}")
                                st.markdown("Please try a different commodity or location.")
else:
    # Enhanced error display if data loading fails
    st.error("❌ Critical Error: Unable to load the main dataset")
    
    error_cols = st.columns([1, 2, 1])
    with error_cols[1]:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Exclamation_mark_white_icon.svg/1200px-Exclamation_mark_white_icon.svg.png", width=100)
        st.markdown("<p style='text-align: center; color: red;'>The application cannot start without the required data file.</p>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Possible Solutions:
    1. Ensure the data file `data_pangan_jabodetabek_wide_imputed.parquet` is present in the application directory
    2. Check file permissions and ensure Streamlit has read access
    3. Verify the data file is not corrupted
    4. Restart the Streamlit server
    """)

# Application footer
st.sidebar.divider()
st.sidebar.markdown("### 📝 App Information")
st.sidebar.info(
    """
    **Version:** 1.0.0  
    **Last Updated:** April 2025  
    **Created By:** [Your Name]  
    **GitHub:** [Repository Link](https://github.com/yourusername/food-price-predictor)
    """
)

# Add a footer to the main area
st.divider()
st.caption("""
**Disclaimer:** This application is for informational purposes only. Predictions are based on historical data 
and may not accurately reflect future prices. Market decisions should not be based solely on these forecasts.
""")
