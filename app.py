import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler # Tidak digunakan langsung di app.py jika scaler dari joblib
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import re
# import gdown # Tetap dikomentari, fokus pada local models folder
import time
from datetime import datetime, date, timedelta # Import datetime
import json # Untuk memuat GeoJSON

# --- Konfigurasi Aplikasi ---
MODELS_BASE_FOLDER = "multioutput-models" # Folder model/scaler (UNTUK PREDIKSI)
DATA_FILE_PATH = 'data_pangan_jabodetabek_wide_imputed.parquet'
MODEL_FILENAME_PATTERN = "model_multiout30d_{safe_name}.keras"
SCALER_FILENAME_PATTERN = "scaler_multiout30d_{safe_name}.gz"
LOOK_BACK = 60  # Sesuaikan dengan look_back model Anda
MAX_HORIZON = 30 # Batas prediksi maksimal

GEOJSON_FILE_PATH = '38 Provinsi Indonesia - Kabupaten.json' # Path ke file GeoJSON Anda

# --- Helper Functions ---
def safe_filename(name):
    name = re.sub(r'[\\/ ]', '_', name)
    name = name[:80] # Batasi panjang untuk nama file
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
        st.error(f"Model file '{model_filename}' not found in '{MODELS_BASE_FOLDER}'. Ensure models are in this folder.")
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
        st.error(f"Data file not found: {filepath}. Please ensure it's in the root directory or provide the correct path.")
        return None, None, None, None
    try:
        df = pd.read_parquet(filepath)
        df.index = pd.to_datetime(df.index)
        
        # Ekstrak lokasi dan komoditas dengan lebih hati-hati
        locations = []
        commodities_set = set()
        if not df.empty:
            # Asumsi format kolom adalah "Lokasi_Komoditas"
            # Coba dapatkan lokasi dari bagian pertama nama kolom
            all_locations_in_cols = sorted(list(set(col.split('_')[0] for col in df.columns if '_' in col)))
            
            # Filter untuk lokasi yang umum (bisa disesuaikan atau dibuat lebih dinamis)
            # Ini penting agar 'locations' hanya berisi nama kota/kabupaten
            # Jika ada kolom seperti "Unnamed: 0_level_0", itu akan error.
            # Kita asumsikan lokasi yang valid tidak mengandung "Unnamed"
            locations = [loc for loc in all_locations_in_cols if "Unnamed" not in loc and len(loc) > 2]


            if locations:
                ref_loc = locations[0] 
                for col in df.columns:
                    if col.startswith(ref_loc + '_'):
                        commodities_set.add("_".join(col.split('_')[1:]))
                commodities = sorted(list(commodities_set))
            else: # Fallback jika tidak ada lokasi terdeteksi dengan benar
                locations = ["Lokasi Tidak Terdeteksi"]
                commodities = ["Komoditas Tidak Terdeteksi"]
        else:
            locations = ["Data Kosong"]
            commodities = ["Data Kosong"]
            
        last_data_date = df.index[-1] if not df.empty else None
        print("Data loaded successfully.")
        return df, locations, commodities, last_data_date
    except Exception as e:
        st.error(f"Error loading/parsing data: {e}")
        return None, None, None, None

def predict_recursive(model, initial_sequence_scaled, n_steps_out, scaler):
    current_sequence = initial_sequence_scaled.copy().reshape(1, LOOK_BACK, 1)
    predictions_scaled = []
    for _ in range(n_steps_out):
        next_pred_scaled = model.predict(current_sequence, verbose=0)[0, 0] # Asumsi model univariate atau multi-output yang dimodifikasi
        predictions_scaled.append(next_pred_scaled)
        next_pred_scaled_reshaped = np.array([[next_pred_scaled]])
        current_sequence = np.vstack((current_sequence[0, 1:, :], next_pred_scaled_reshaped))
        current_sequence = current_sequence.reshape(1, LOOK_BACK, 1)
    predictions_actual = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))
    return predictions_actual.flatten()


# --- Fungsi untuk Peta ---
@st.cache_data(ttl=3600 * 24) # Cache GeoJSON untuk 24 jam
def load_geojson_data(geojson_path):
    if not os.path.exists(geojson_path):
        st.error(f"File GeoJSON tidak ditemukan di: {geojson_path}")
        return None
    try:
        with open(geojson_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        st.error(f"Error saat membaca file GeoJSON. Pastikan formatnya valid: {geojson_path}. Detail: {e}")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan lain saat memuat GeoJSON: {e}")
        return None

@st.cache_data(ttl=3600 * 24)
def filter_geojson_for_java(geojson_data_full):
    if geojson_data_full is None or 'features' not in geojson_data_full:
        st.warning("Data GeoJSON tidak valid atau kosong, tidak dapat memfilter untuk Pulau Jawa.")
        return None
    
    provinsi_jawa = [
        "Banten", "DKI Jakarta", "Jawa Barat", 
        "Jawa Tengah", "Daerah Istimewa Yogyakarta", "Jawa Timur"
    ]
    
    # KUNCI PROPERTI DARI FILE GEOJSON ANDA (SESUAIKAN!)
    KABKOT_PROPERTY_KEY = "WADMKK"  # Nama properti untuk Kabupaten/Kota
    PROVINSI_PROPERTY_KEY = "WADMPR" # Nama properti untuk Provinsi

    filtered_features = []
    for feature in geojson_data_full['features']:
        props = feature.get('properties', {})
        nama_provinsi = props.get(PROVINSI_PROPERTY_KEY)
        nama_kabkot = props.get(KABKOT_PROPERTY_KEY)

        if nama_provinsi in provinsi_jawa and nama_kabkot:
            simplified_feature = {
                "type": "Feature",
                "geometry": feature["geometry"],
                "properties": {
                    "id_lokasi": nama_kabkot # Ini akan menjadi ID untuk matching
                }
            }
            filtered_features.append(simplified_feature)
            
    if not filtered_features:
        st.warning(f"Tidak ada fitur ditemukan untuk provinsi di Pulau Jawa dengan kunci provinsi '{PROVINSI_PROPERTY_KEY}' dan kabupaten/kota '{KABKOT_PROPERTY_KEY}'.")
        # Untuk debug, tampilkan contoh properti dari GeoJSON jika gagal
        if geojson_data_full['features']:
            st.json({"Contoh Properti Fitur Pertama GeoJSON": geojson_data_full['features'][0].get('properties', {})})
        return None
        
    return {"type": "FeatureCollection", "features": filtered_features}

# SESUAIKAN PEMETAAN INI DENGAN SANGAT HATI-HATI!
# Kunci: Nama lokasi dari df_wide (misal "Kota Bekasi")
# Nilai: Nama lokasi yang SAMA PERSIS dengan nilai di properti WADMKK di GeoJSON Anda
jabodetabek_map_ids = {
    "Kota Bekasi": "Kota Bekasi", # Ganti "Kota Bekasi" jika di GeoJSON hanya "Bekasi"
    "Kota Bogor": "Kota Bogor",   # Ganti "Kota Bogor" jika di GeoJSON hanya "Bogor"
    "Jakarta Pusat": "Kota Administrasi Jakarta Pusat", # Contoh jika di GeoJSON lebih lengkap
    "Kota Tangerang": "Kota Tangerang", # Ganti "Kota Tangerang" jika di GeoJSON hanya "Tangerang"
    "Kota Depok": "Kota Depok"    # Ganti "Kota Depok" jika di GeoJSON hanya "Depok"
}

def get_latest_prices_for_map_display(df_wide, target_commodity, map_ids_config, date_to_display):
    price_data = []
    for app_loc_name, map_id_value in map_ids_config.items():
        column_name = f"{app_loc_name}_{target_commodity}"
        if column_name in df_wide.columns:
            price = df_wide.loc[date_to_display, column_name]
            price_data.append({
                "id_lokasi": map_id_value, # Ini ID untuk matching di peta
                "Harga": price,
                "NamaTampil": app_loc_name # Nama yang akan ditampilkan di hover/tabel
            })
        # else:
            # Jika tidak ada data, biarkan saja, akan difilter dengan dropna nanti
    df = pd.DataFrame(price_data)
    df.dropna(subset=['Harga'], inplace=True) # Hanya tampilkan yang ada harga
    return df

# --- Streamlit App Layout ---
st.set_page_config(page_title="Harga Pangan Jabodetabek", page_icon="📈", layout="wide")

# Load Data dan GeoJSON sekali di awal
df_wide, locations, commodities, last_data_date = load_data(DATA_FILE_PATH)
geojson_data_full = load_geojson_data(GEOJSON_FILE_PATH)
geojson_java_filtered = None
if geojson_data_full:
    geojson_java_filtered = filter_geojson_for_java(geojson_data_full)


# Header Aplikasi
col_header1, col_header2 = st.columns([1, 6]) # Sesuaikan rasio jika perlu
with col_header1:
     st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Flag_of_Indonesia.svg/120px-Flag_of_Indonesia.svg.png", width=70) # Ukuran disesuaikan
with col_header2:
    st.title("📈 Analisis & Prediksi Harga Pangan Jabodetabek")
    if last_data_date:
        st.caption(f"Data terakhir diperbarui: {last_data_date.strftime('%A, %d %B %Y')}")
    else:
        st.caption("Data historis tidak termuat.")

# Informasi Penting
if last_data_date:
    st.info(f"""
        ℹ️ **Informasi Proyek:**
        * Aplikasi ini adalah **Proyek Portofolio (Work In Progress v0.3 - Peta Ditambahkan)**.
        * Data historis yang digunakan untuk analisis dan model terakhir diperbarui pada **{last_data_date.strftime('%d %B %Y')}**.
        * Prediksi harga dihasilkan berdasarkan data hingga tanggal tersebut. Akurasi prediksi menurun untuk horizon waktu yang lebih panjang.
        * Fitur Peta menampilkan harga terakhir yang tercatat untuk komoditas terpilih di wilayah Jabodetabek.
    """)
st.divider()

# Cek jika data gagal dimuat
if df_wide is None or not locations or not commodities or last_data_date is None:
    st.error("❌ Gagal memuat data historis atau data tidak lengkap. Beberapa fitur mungkin tidak berfungsi.")
    # Jangan hentikan aplikasi sepenuhnya agar pesan error bisa terlihat
    # st.stop()
else:
    # --- Sidebar untuk Input Prediksi (MASIH ADA) ---
    st.sidebar.header("📌 Parameter Analisis & Prediksi")
    selected_location = st.sidebar.selectbox("1. Pilih Lokasi (untuk Grafik & Prediksi Detail):", locations, key="loc_select")
    selected_commodity_pred = st.sidebar.selectbox("2. Pilih Komoditas (untuk Grafik & Prediksi Detail):", commodities, key="com_pred_select")

    st.sidebar.header("🗓️ Tanggal Prediksi (untuk Prediksi Detail)")
    min_pred_date = last_data_date.date() + timedelta(days=1)
    max_pred_date = last_data_date.date() + timedelta(days=MAX_HORIZON)

    selected_pred_date = st.sidebar.date_input(
        "Prediksi hingga tanggal:",
        value=min_pred_date,
        min_value=min_pred_date,
        max_value=max_pred_date,
        key="pred_date_select",
        help=f"Pilih tanggal akhir prediksi (maks. {MAX_HORIZON} hari setelah {last_data_date.strftime('%d %b %Y')})"
    )
    days_to_predict = (selected_pred_date - last_data_date.date()).days
    st.sidebar.caption(f"Prediksi untuk {days_to_predict} hari ke depan.")

    predict_button = st.sidebar.button(f"🔮 Prediksi Harga Detail", type="primary", use_container_width=True)
    st.sidebar.markdown("---")
    with st.sidebar.expander("ℹ️ Tentang Model & Data"):
        st.markdown(f"""
        * **Model Prediksi:** LSTM Multi-Output (Direct)
        * **Input Model:** Histori {LOOK_BACK} hari terakhir
        * **Horizon Prediksi:** Hingga {MAX_HORIZON} hari
        * **Data Source:** PIHPS Nasional (diolah)
        * **Data Cutoff:** {last_data_date.strftime('%d %b %Y')}
        """)

    # --- Pengaturan Tab Utama ---
    tab_titles_main = ["📊 Analisis Historis", "🚀 Prediksi Harga"]
    if geojson_java_filtered and geojson_java_filtered['features']:
        tab_titles_main.insert(0, "🗺️ Peta Harga Jabodetabek") # Peta di tab pertama

    tabs_main = st.tabs(tab_titles_main)
    
    current_tab_index = 0

    # --- TAB PETA HARGA ---
    if geojson_java_filtered and geojson_java_filtered['features']:
        with tabs_main[current_tab_index]:
            st.subheader(f"Peta Sebaran Harga Komoditas di Jabodetabek")
            
            col_map_filter1, col_map_filter2 = st.columns([3,2])
            with col_map_filter1:
                default_commodity_map = "Beras Kualitas Medium I" # Sesuaikan jika perlu
                try:
                    default_index_map = commodities.index(default_commodity_map)
                except ValueError:
                    default_index_map = 0
                
                selected_commodity_map_tab = st.selectbox(
                    "Pilih Komoditas untuk Ditampilkan di Peta:",
                    commodities,
                    index=default_index_map,
                    key="map_commodity_tab_select"
                )
            with col_map_filter2:
                 st.info(f"Menampilkan harga per: **{last_data_date.strftime('%d %B %Y')}**")


            df_map_prices = get_latest_prices_for_map_display(df_wide, selected_commodity_map_tab, jabodetabek_map_ids, last_data_date)

            if not df_map_prices.empty:
                FEATURE_ID_KEY_MAP = "properties.id_lokasi" # Sesuai dengan output filter_geojson_for_java
                
                min_price_map = df_map_prices['Harga'].min()
                max_price_map = df_map_prices['Harga'].max()
                color_range_map = None
                if pd.notna(min_price_map) and pd.notna(max_price_map) and min_price_map != max_price_map:
                    color_range_map = [min_price_map, max_price_map]

                try:
                    fig_map = px.choropleth_mapbox(
                        df_map_prices,
                        geojson=geojson_java_filtered,
                        locations="id_lokasi",
                        featureidkey=FEATURE_ID_KEY_MAP,
                        color="Harga",
                        color_continuous_scale="YlOrRd",
                        mapbox_style="carto-positron",
                        zoom=8, # Zoom lebih dekat ke Jabodetabek
                        center={"lat": -6.35, "lon": 106.8}, # Center di sekitar Depok/Selatan Jakarta
                        opacity=0.7,
                        hover_name="NamaTampil",
                        hover_data={"Harga": ":,.0f Rp", "id_lokasi": False},
                        range_color=color_range_map,
                        labels={'Harga':'Harga (Rp)'}
                    )
                    fig_map.update_layout(
                        # mapbox_accesstoken=st.secrets.get("MAPBOX_TOKEN"), # Uncomment jika pakai token Mapbox
                        margin={"r":10,"t":10,"l":10,"b":10},
                        height=500,
                        coloraxis_colorbar=dict(
                            title=f"Harga<br>{selected_commodity_map_tab.replace('_',' ')}",
                            thicknessmode="pixels", thickness=12,
                            lenmode="fraction", len=0.75,
                            yanchor="middle", y=0.5,
                            xanchor="right", x=0.99
                        )
                    )
                    st.plotly_chart(fig_map, use_container_width=True)
                    
                    # Tabel Detail di bawah peta
                    st.markdown(f"###### Detail Harga {selected_commodity_map_tab.replace('_', ' ')} ({last_data_date.strftime('%d %B %Y')})")
                    df_display_map_table = df_map_prices.copy()
                    df_display_map_table['Harga (Rp)'] = df_display_map_table['Harga'].apply(
                        lambda x: f"Rp {x:,.0f}" if pd.notnull(x) else "N/A"
                    )
                    df_table_display_final = df_display_map_table[['NamaTampil', 'Harga (Rp)']].rename(
                        columns={'NamaTampil': 'Wilayah'}
                    ).set_index('Wilayah')
                    st.dataframe(df_table_display_final, use_container_width=True)

                except Exception as e:
                    st.error(f"Gagal membuat peta untuk {selected_commodity_map_tab}: {e}")
                    st.caption("Debug Info:")
                    st.write("Data untuk Peta:", df_map_prices)
                    if geojson_java_filtered and geojson_java_filtered['features']:
                         st.write("Contoh Properti GeoJSON Filtered:", geojson_java_filtered['features'][0]['properties'])
            else:
                st.warning(f"Tidak ada data harga valid ditemukan untuk '{selected_commodity_map_tab}' di Jabodetabek pada tanggal {last_data_date.strftime('%d %B %Y')}.")
        current_tab_index +=1

    # --- TAB ANALISIS HISTORIS ---
    with tabs_main[current_tab_index]:
        target_column_name_hist = f"{selected_location}_{selected_commodity_pred}"
        if target_column_name_hist not in df_wide.columns:
            st.error(f"Data untuk '{selected_commodity_pred}' di '{selected_location}' (historis) tidak tersedia.")
        else:
            ts_data_hist = df_wide[[target_column_name_hist]].copy()
            ts_data_hist.rename(columns={target_column_name_hist: 'Harga Aktual'}, inplace=True)
            last_actual_value_hist = ts_data_hist.iloc[-1, 0]

            st.subheader(f"Analisis Harga Historis: {selected_commodity_pred.replace('_',' ')}")
            st.caption(f"Lokasi: {selected_location}")

            time_periods = {
                "30 Hari Terakhir": 30, "90 Hari Terakhir": 90,
                "6 Bulan Terakhir": 180, "1 Tahun Terakhir": 365,
                "Semua Waktu": len(ts_data_hist)
            }
            selected_period_key = st.selectbox("Tampilkan Periode:", options=list(time_periods.keys()), index=1, key="hist_period")
            days_to_show = time_periods[selected_period_key]
            plot_data_hist = ts_data_hist.iloc[-min(days_to_show, len(ts_data_hist)):]

            fig_hist = px.line(
                plot_data_hist, x=plot_data_hist.index, y='Harga Aktual',
                labels={'index': 'Tanggal', 'Harga Aktual': 'Harga (Rp)'},
                template="plotly_white" # Desain minimalis
            )
            fig_hist.update_traces(line=dict(color='royalblue', width=2.5))
            fig_hist.update_layout(hovermode='x unified', height=400, margin=dict(l=10, r=10, t=30, b=10),
                                   title_text=f"Tren Harga {selected_commodity_pred.replace('_',' ')} - {selected_location}", title_x=0.5)
            st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("---")
            st.markdown("##### Ringkasan Statistik Historis")
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                 st.metric("Harga Terakhir", f"Rp {last_actual_value_hist:,.0f}", help=f"Per {ts_data_hist.index[-1].strftime('%d %b %Y')}")
            with col_stat2:
                 st.metric(f"Harga Min ({selected_period_key})", f"Rp {plot_data_hist['Harga Aktual'].min():,.0f}")
                 st.metric(f"Harga Rata-rata ({selected_period_key})", f"Rp {plot_data_hist['Harga Aktual'].mean():,.0f}")
            with col_stat3:
                 st.metric(f"Harga Max ({selected_period_key})", f"Rp {plot_data_hist['Harga Aktual'].max():,.0f}")
        current_tab_index +=1

    # --- TAB PREDIKSI HARGA ---
    with tabs_main[current_tab_index]:
        st.subheader(f"Prediksi Harga: {selected_commodity_pred.replace('_',' ')}")
        st.caption(f"Lokasi: {selected_location} | Prediksi untuk {days_to_predict} hari ke depan (hingga {selected_pred_date.strftime('%d %B %Y')})")
        
        target_column_name_pred = f"{selected_location}_{selected_commodity_pred}"

        if target_column_name_pred not in df_wide.columns:
            st.error(f"Data untuk '{selected_commodity_pred}' di '{selected_location}' (prediksi) tidak tersedia.")
        else:
            ts_data_pred_base = df_wide[[target_column_name_pred]].copy()
            ts_data_pred_base.rename(columns={target_column_name_pred: 'Harga Aktual'}, inplace=True)
            last_actual_date_pred = ts_data_pred_base.index[-1]
            last_actual_value_pred = ts_data_pred_base.iloc[-1, 0]

            if not predict_button:
                st.info("Tekan tombol '🔮 Prediksi Harga Detail' di sidebar untuk melihat hasil forecast.")
            else:
                with st.spinner(f"Memuat model & scaler untuk {target_column_name_pred}..."):
                    model = load_keras_model(target_column_name_pred)
                    scaler = load_joblib_scaler(target_column_name_pred)

                if model is None or scaler is None:
                    st.error("Gagal memuat file model/scaler. Prediksi dibatalkan.")
                else:
                    with st.spinner(f"Menjalankan prediksi untuk {days_to_predict} hari..."):
                        last_sequence_actual = ts_data_pred_base.iloc[-LOOK_BACK:]
                        if len(last_sequence_actual) < LOOK_BACK:
                             st.warning(f"Data historis tidak cukup ({len(last_sequence_actual)} dari {LOOK_BACK} hari dibutuhkan).")
                        else:
                            initial_sequence_scaled = scaler.transform(last_sequence_actual.values)
                            predictions_actual = predict_recursive(model, initial_sequence_scaled, days_to_predict, scaler)
                            prediction_dates = pd.date_range(start=last_actual_date_pred + timedelta(days=1), periods=days_to_predict, freq='D')
                            pred_df = pd.DataFrame({'Harga Prediksi': predictions_actual}, index=prediction_dates)

                            st.markdown("##### Ringkasan Prediksi")
                            col_pred1, col_pred2, col_pred3 = st.columns(3)
                            delta_vs_today = predictions_actual[0] - last_actual_value_pred
                            delta_sign = "+" if delta_vs_today >=0 else ""
                            with col_pred1:
                                st.metric(
                                    label=f"Prediksi Besok ({prediction_dates[0].strftime('%d %b')})",
                                    value=f"Rp {predictions_actual[0]:,.0f}",
                                    delta=f"{delta_sign}{delta_vs_today:,.0f} Rupiah"
                                )
                            if days_to_predict > 1:
                                delta_vs_today_end = predictions_actual[-1] - last_actual_value_pred
                                delta_sign_end = "+" if delta_vs_today_end >=0 else ""
                                with col_pred2:
                                     st.metric(
                                        label=f"Prediksi Akhir ({prediction_dates[-1].strftime('%d %b')})",
                                        value=f"Rp {predictions_actual[-1]:,.0f}",
                                        delta=f"{delta_sign_end}{delta_vs_today_end:,.0f} Rupiah"
                                    )
                            with col_pred3:
                                 avg_pred = predictions_actual.mean()
                                 st.metric(label=f"Rata-rata Prediksi ({days_to_predict} Hari)", value=f"Rp {avg_pred:,.0f}")

                            st.markdown("##### Grafik Prediksi vs Historis")
                            fig_pred = go.Figure()
                            hist_plot_data_pred = ts_data_pred_base.iloc[-min(90, len(ts_data_pred_base)):] # Tampilkan 90 hari historis
                            fig_pred.add_trace(go.Scatter(x=hist_plot_data_pred.index, y=hist_plot_data_pred['Harga Aktual'],
                                                        mode='lines', name='Harga Aktual', line=dict(color='royalblue', width=2)))
                            fig_pred.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Harga Prediksi'],
                                                        mode='lines+markers', name='Harga Prediksi', line=dict(color='firebrick', dash='dash'), marker=dict(size=5)))

                            fig_pred.update_layout(
                                title_text=f'Prediksi Harga {selected_commodity_pred.replace("_"," ")} - {selected_location}', title_x=0.5,
                                xaxis_title='Tanggal', yaxis_title='Harga (Rp)',
                                hovermode='x unified', height=450, margin=dict(l=10, r=10, t=40, b=10),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            st.plotly_chart(fig_pred, use_container_width=True)

                            with st.expander("Lihat Detail Prediksi Harian"):
                                pred_df_display = pred_df.copy()
                                pred_df_display.index = pred_df_display.index.strftime('%A, %d %b %Y')
                                pred_df_display['Harga Prediksi'] = pred_df_display['Harga Prediksi'].apply(lambda x: f"Rp {x:,.0f}")
                                st.dataframe(pred_df_display, use_container_width=True)

                            if days_to_predict > 10: # Peringatan akurasi untuk prediksi > 10 hari
                                st.warning("⚠️ Akurasi prediksi cenderung menurun signifikan untuk horizon waktu yang lebih panjang (>10 hari). Gunakan sebagai indikasi tren.")
                            st.caption(f"Model: LSTM Multi-Output (Recursive Prediction from Univariate-like Model). Input: {LOOK_BACK} hari.")
        current_tab_index +=1


# Footer Aplikasi (jika ada)
st.divider()
st.caption(f"Sumber Data Utama: PIHPS Nasional (Bank Indonesia) | Aplikasi dikembangkan sebagai Proyek Portofolio.")
if last_data_date:
    st.caption(f"Versi Data: {last_data_date.strftime('%Y%m%d')}")
