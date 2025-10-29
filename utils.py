# utils.py - Fungsi Utilitas untuk Streamlit App
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def to_numeric_keep_none(s):
    """Bersihkan string angka yang mengandung koma ribuan atau titik desimal"""
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float, np.number)):
        return s
    try:
        s2 = str(s).strip()
        if s2.count(',')==1 and s2.count('.')>0 and s2.rfind('.') < s2.rfind(','):
            s2 = s2.replace('.', '').replace(',', '.')
        else:
            s2 = s2.replace(',', '')
        return float(s2)
    except:
        return np.nan

def standardize_provinsi_name(df, col='Provinsi'):
    """Standarisasi nama provinsi: strip, uppercase, normalize spacing"""
    df[col] = df[col].astype(str).str.strip().str.upper().str.replace(r'\s+',' ', regex=True)
    return df

def ensure_year_int(df, col='Tahun'):
    """Konversi kolom tahun menjadi integer dengan handling missing values"""
    if col not in df.columns:
        return df
    
    try:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        try:
            df[col] = df[col].astype('Int64')
        except:
            df[col] = df[col].fillna(0).astype(int)
            df.loc[df[col] == 0, col] = np.nan
    except Exception as e:
        print(f"Warning: Tidak dapat mengkonversi kolom {col} ke integer: {e}")
    
    return df

def clean_pdrb_data(df_pdrb):
    """Membersihkan data PDRB"""
    df_clean = df_pdrb.copy()
    
    if 'nama_provinsi' in df_clean.columns:
        df_clean = standardize_provinsi_name(df_clean, 'nama_provinsi')
        df_clean.rename(columns={'nama_provinsi':'Provinsi'}, inplace=True)
    
    if 'pdrb_per_kapita' in df_clean.columns:
        df_clean['pdrb_per_kapita'] = df_clean['pdrb_per_kapita'].apply(to_numeric_keep_none)
    
    if 'tahun' in df_clean.columns:
        df_clean = ensure_year_int(df_clean, 'tahun')
        df_clean.rename(columns={'tahun':'Tahun'}, inplace=True)
    
    if 'satuan' in df_clean.columns:
        df_clean['satuan'] = df_clean['satuan'].astype(str)
        mask_ribu = df_clean['satuan'].str.upper().str.contains('RIBU', na=False)
        df_clean.loc[mask_ribu,'pdrb_per_kapita'] = df_clean.loc[mask_ribu,'pdrb_per_kapita'] * 1000
        df_clean.rename(columns={'pdrb_per_kapita':'PDRB_per_kapita_Rupiah'}, inplace=True)
    
    return df_clean

def clean_waste_data(df_waste):
    """Membersihkan data sampah"""
    df_clean = df_waste.copy()
    df_clean.columns = df_clean.columns.str.strip().str.replace(r'\s+',' ', regex=True)
    
    rename_map = {
        'Timbulan Sampah Harian(ton)': 'Timbulan_Harian_ton',
        'Timbulan Sampah Tahunan(ton)': 'Timbulan_Tahunan_ton',
        'Provinsi': 'Provinsi',
        'Kabupaten/Kota': 'Kabupaten_Kota',
        'Tahun': 'Tahun'
    }
    
    for k,v in rename_map.items():
        if k in df_clean.columns:
            df_clean.rename(columns={k:v}, inplace=True)
    
    for col in ['Timbulan_Harian_ton','Timbulan_Tahunan_ton']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(to_numeric_keep_none)
    
    df_clean = standardize_provinsi_name(df_clean, 'Provinsi')
    df_clean = ensure_year_int(df_clean, 'Tahun')
    
    return df_clean

def clean_population_data(df_pop):
    """Membersihkan data populasi"""
    df_clean = df_pop.copy()
    
    col_names = df_clean.columns.tolist()
    if len(col_names) >= 3:
        df_clean.columns = ['Provinsi', 'Jumlah_Penduduk_Ribu', 'Tahun']
    
    if 'Provinsi' in df_clean.columns:
        df_clean = standardize_provinsi_name(df_clean, 'Provinsi')
    
    if 'Jumlah_Penduduk_Ribu' in df_clean.columns:
        df_clean['Jumlah_Penduduk_Ribu'] = df_clean['Jumlah_Penduduk_Ribu'].apply(to_numeric_keep_none)
        df_clean['Jumlah_Penduduk'] = df_clean['Jumlah_Penduduk_Ribu'] * 1000
    
    if 'Tahun' in df_clean.columns:
        df_clean = ensure_year_int(df_clean, 'Tahun')
    
    if 'Provinsi' in df_clean.columns:
        df_clean = df_clean[~df_clean['Provinsi'].str.upper().isin(['INDONESIA', 'NASIONAL', 'TOTAL'])]
    
    return df_clean

def clean_infrastructure_data(df_infra):
    """Membersihkan data infrastruktur"""
    df_clean = df_infra.copy()
    df_clean.columns = df_clean.columns.str.strip()
    
    possible_prov = [c for c in df_clean.columns if 'prov' in c.lower()]
    if possible_prov:
        df_clean.rename(columns={possible_prov[0]:'Provinsi'}, inplace=True)
        df_clean = standardize_provinsi_name(df_clean, 'Provinsi')
    
    if 'Tahun' not in df_clean.columns:
        possible_year = [c for c in df_clean.columns if 'tahun' in c.lower()]
        if possible_year:
            df_clean.rename(columns={possible_year[0]:'Tahun'}, inplace=True)
    
    df_clean = ensure_year_int(df_clean, 'Tahun')
    
    if 'Provinsi' in df_clean.columns and 'Tahun' in df_clean.columns:
        fac_count = df_clean.groupby(['Provinsi','Tahun']).size().reset_index(name='Jumlah_Fasilitas')
        
        sampah_in_cols = [c for c in df_clean.columns if 'sampahmasuk' in c.lower() or 'sampah masuk' in c.lower()]
        if sampah_in_cols:
            samp_col = sampah_in_cols[0]
            agg_sampah = df_clean.groupby(['Provinsi','Tahun'])[samp_col].sum().reset_index()
            agg_sampah.rename(columns={samp_col:'Infra_Sampahmasuk_ton'}, inplace=True)
            infra_summary = fac_count.merge(agg_sampah, on=['Provinsi','Tahun'], how='left')
        else:
            infra_summary = fac_count.copy()
            infra_summary['Infra_Sampahmasuk_ton'] = np.nan
    else:
        infra_summary = pd.DataFrame()
    
    return infra_summary

# ============================================================================
# MODELING FUNCTIONS
# ============================================================================

def prepare_modeling_data(df_waste, df_pdrb, df_pop, df_infra):
    """Menggabungkan dan mempersiapkan data untuk modeling"""
    
    # Agregasi waste data
    df_waste_agg = df_waste.groupby(['Provinsi', 'Tahun']).agg({
        'Timbulan_Tahunan_ton': 'sum',
        'Timbulan_Harian_ton': 'sum'
    }).reset_index()
    
    # Merge semua dataset
    df_model = df_waste_agg.copy()
    df_model = df_model.merge(
        df_pdrb[['Provinsi', 'Tahun', 'PDRB_per_kapita_Rupiah']],
        on=['Provinsi', 'Tahun'],
        how='left'
    )
    df_model = df_model.merge(
        df_infra[['Provinsi', 'Tahun', 'Jumlah_Fasilitas', 'Infra_Sampahmasuk_ton']],
        on=['Provinsi', 'Tahun'],
        how='left'
    )
    df_model = df_model.merge(
        df_pop[['Provinsi', 'Tahun', 'Jumlah_Penduduk']],
        on=['Provinsi', 'Tahun'],
        how='left'
    )
    
    # Imputasi missing values
    num_cols = df_model.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in num_cols:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
    
    # Imputasi berdasarkan median per Provinsi
    cols_impute = ['PDRB_per_kapita_Rupiah', 'Jumlah_Fasilitas', 'Infra_Sampahmasuk_ton']
    for col in cols_impute:
        if col in df_model.columns:
            df_model[col] = df_model.groupby('Provinsi')[col].transform(
                lambda x: x.fillna(x.median())
            )
    
    if 'Jumlah_Penduduk' in df_model.columns:
        df_model['Jumlah_Penduduk'] = df_model.groupby('Provinsi')['Jumlah_Penduduk'].transform(
            lambda x: x.fillna(x.mean())
        )
    
    # Fill remaining with global median
    df_model = df_model.fillna(df_model.median(numeric_only=True))
    
    # Remove duplicates
    df_model = df_model.drop_duplicates().reset_index(drop=True)
    
    return df_model

def train_models(df_model):
    """Melatih 3 model regresi dan mengembalikan hasil"""
    
    feature_cols = [
        'PDRB_per_kapita_Rupiah',
        'Jumlah_Fasilitas',
        'Infra_Sampahmasuk_ton',
        'Jumlah_Penduduk',
    ]
    target_col = 'Timbulan_Tahunan_ton'
    
    # Filter columns yang ada
    available_features = [f for f in feature_cols if f in df_model.columns]
    
    # Prepare X and y
    X = df_model[available_features].dropna()
    y = df_model.loc[X.index, target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_test': y_test,
            'R2': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'scaler': scaler,
            'features': available_features
        }
    
    return results

def save_model(model, scaler, features, filename='best_model.pkl'):
    """Menyimpan model terbaik"""
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': features
    }
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)

def load_model(filename='best_model.pkl'):
    """Memuat model yang telah disimpan"""
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def predict_waste(pdrb, population, facilities, infra_capacity, model_data):
    """Melakukan prediksi timbulan sampah"""
    
    # Prepare input
    input_data = pd.DataFrame({
        'PDRB_per_kapita_Rupiah': [pdrb],
        'Jumlah_Penduduk': [population],
        'Jumlah_Fasilitas': [facilities],
        'Infra_Sampahmasuk_ton': [infra_capacity]
    })
    
    # Scale input
    input_scaled = model_data['scaler'].transform(input_data[model_data['features']])
    
    # Predict
    prediction = model_data['model'].predict(input_scaled)[0]
    
    # Calculate confidence interval (simplified)
    confidence_lower = prediction * 0.90
    confidence_upper = prediction * 1.10
    
    return {
        'prediction': prediction,
        'lower_bound': confidence_lower,
        'upper_bound': confidence_upper
    }

# ============================================================================
# VISUALIZATION HELPER FUNCTIONS
# ============================================================================

def get_feature_importance(model, feature_names):
    """Mendapatkan feature importance dari model"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
    return None

def calculate_statistics(df):
    """Menghitung statistik deskriptif"""
    stats = {
        'mean': df.mean(),
        'median': df.median(),
        'std': df.std(),
        'min': df.min(),
        'max': df.max(),
        'count': df.count()
    }
    return pd.DataFrame(stats)

def get_correlation_matrix(df, columns):
    """Menghitung correlation matrix"""
    return df[columns].corr()

# ============================================================================
# DATA VALIDATION FUNCTIONS
# ============================================================================

def validate_input_data(pdrb, population, facilities, infra_capacity):
    """Validasi input data dari user"""
    errors = []
    
    if pdrb < 10000000 or pdrb > 200000000:
        errors.append("PDRB per kapita harus antara 10 juta - 200 juta Rupiah")
    
    if population < 100000 or population > 100000000:
        errors.append("Jumlah penduduk harus antara 100 ribu - 100 juta")
    
    if facilities < 10 or facilities > 1000:
        errors.append("Jumlah fasilitas harus antara 10 - 1000")
    
    if infra_capacity < 1000 or infra_capacity > 10000000:
        errors.append("Kapasitas infrastruktur harus antara 1 ribu - 10 juta ton")
    
    return errors

def check_data_quality(df):
    """Mengecek kualitas data"""
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': len(df.select_dtypes(include=['number']).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns)
    }
    return quality_report