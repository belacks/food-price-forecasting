# Indonesian Food Price Prediction App

## Description

This application provides historical price analysis and future price predictions for various key food commodities across selected markets in the Jabodetabek (Jakarta, Bogor, Depok, Tangerang, Bekasi) region of Indonesia. It utilizes individual univariate LSTM (Long Short-Term Memory) deep learning models trained for each specific commodity-location pair to generate forecasts.

The goal is to offer an informative and user-friendly tool for both general users and experts interested in monitoring and understanding food price dynamics in this vital economic area.

**Data Source:** [Pusat Informasi Harga Pangan Strategis (PIHPS) Nasional - Bank Indonesia](https://hargapangan.id/) (Data processed from 2018 onwards due to availability).

**Disclaimer:** This is a portfolio project demonstrating data science and deployment skills. Predictions are based on historical data and model assumptions; they are not financial advice and involve inherent uncertainties.

## Features

*   **User-friendly Interface:** Select Location (City/Regency) and Commodity from dropdown menus.
*   **Historical Price Visualization:** Displays an interactive chart of the historical price trend for the selected commodity and location.
*   **LSTM-Based Prediction:** Generates a price prediction for the next day (or potentially a few days ahead, depending on implementation) using a dedicated pre-trained LSTM model.
*   **Informative Output:** Clearly presents the predicted price and contextualizes it with historical data.

## Model

The prediction engine uses a collection of **Univariate LSTM models**. Each model is trained *specifically* on the historical price data of *one* commodity at *one* location (e.g., a dedicated model for 'Beras Kualitas Medium I' in 'Jakarta Pusat', another for 'Cabai Merah Keriting' in 'Kota Bogor', etc.).

## Setup & Installation (Local)

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Model & Scaler Files (Local Setup - Optional):** For local running without GDrive download, you would need to place the `univariate_models_saved` folder (containing `.keras` and `.gz` files) within the project directory or update the paths in `app.py`.
5.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## Deployment (Streamlit Cloud with Google Drive)

This app is designed to be deployed on Streamlit Cloud, loading models and scalers directly from a shared Google Drive folder.

1.  **Prepare Google Drive:**
    *   Ensure all your trained model files (`.keras`) and scaler files (`.gz`) are located within a **single folder** in your Google Drive (e.g., a folder named `univariate_models_saved`).
    *   Right-click on this folder in Google Drive.
    *   Select "Share" -> "Share".
    *   Under "General access", change "Restricted" to **"Anyone with the link"**.
    *   Ensure the role is set to **"Viewer"**.
    *   Copy the **Folder ID** from the shareable link URL. The URL looks like `https://drive.google.com/drive/folders/FOLDER_ID?usp=sharing`. You only need the `FOLDER_ID` part.
2.  **Update `app.py`:**
    *   Open the `app.py` script.
    *   Find the `GDRIVE_FOLDER_ID` variable in the configuration section.
    *   **Paste your actual Google Drive Folder ID** here.
3.  **Push to GitHub:** Commit and push your code (including `app.py` with the updated Folder ID and `requirements.txt`) to your GitHub repository. **Do NOT commit the actual model/scaler files to GitHub if they are large or numerous; rely on the GDrive download.**
4.  **Deploy on Streamlit Cloud:**
    *   Log in to your Streamlit Cloud account.
    *   Click "New app" -> "From repo".
    *   Select your GitHub repository, branch, and the main file path (usually `app.py`).
    *   Click "Deploy!".
5.  **First Run & Caching:** The first time the app runs (or after an update), it will need to download the required model and scaler files from Google Drive when a user makes a selection. This might take a moment. Subsequent requests for the *same* model/scaler should be much faster due to Streamlit's caching.

## File Structure
├── univariate_models_saved/ # (On Google Drive, NOT necessarily in Repo)
│ ├── model_univar_Lokasi1_KomoditasA.keras
│ ├── scaler_univar_Lokasi1_KomoditasA.gz
│ ├── model_univar_Lokasi1_KomoditasB.keras
│ ├── scaler_univar_Lokasi1_KomoditasB.gz
│ └── ...
├── data_pangan_jabodetabek_wide_imputed.parquet # Main historical data file
├── app.py # The main Streamlit application script
├── requirements.txt # Python dependencies
└── README.md # This file

## Contributing

This is currently a portfolio project. Contributions are welcome via pull requests after discussing potential changes.
