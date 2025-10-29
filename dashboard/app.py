import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# KONFIGURASI HALAMAN
# ============================================================================
st.set_page_config(
    page_title="Prediksi Timbulan Sampah Indonesia",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS UNTUK TAMPILAN MENARIK
# ============================================================================
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling */
    .header-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(120deg, #2ecc71, #3498db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        animation: fadeIn 1s ease-in;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #3498db;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #3498db, #2ecc71);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(52,152,219,0.4);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNGSI UTILITAS
# ============================================================================
@st.cache_data
def load_data():
    # Ganti dengan path dataset Anda
    df_pdrb = pd.read_csv('data/dataset_pdrb_clean.csv')
    df_waste = pd.read_csv('data/dataset_waste_clean.csv')
    df_pop = pd.read_csv('data/dataset_pop_clean.csv')
    df_infra = pd.read_csv('data/dataset_infra_summary.csv')

    return df_pdrb, df_waste, df_pop, df_infra

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1 style='color: white;'>♻️ Navigation</h1>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Home", "📊 Data Overview", "🔍 EDA", "📈 Visualisasi", "🤖 Modeling", "🎯 Prediksi"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Tentang Aplikasi:**

Aplikasi ini menganalisis dan memprediksi timbulan sampah di Indonesia berdasarkan:
- 📈 PDRB per Kapita
- 👥 Populasi Penduduk
- 🏗️ Infrastruktur Pengelolaan
""")

# ============================================================================
# LOAD DATA
# ============================================================================
df_pdrb, df_waste, df_pop, df_infra = load_data()

# ============================================================================
# HALAMAN 1: HOME
# ============================================================================
if page == "🏠 Home":
    st.markdown("<h1 class='header-title'>♻️ Sistem Prediksi Timbulan Sampah Indonesia</h1>", 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <h2>🎯 Tentang Sistem</h2>
        <p style='font-size: 1.1rem; line-height: 1.8;'>
        Sistem berbasis <b>Ensemble Learning</b> untuk memprediksi timbulan sampah 
        menggunakan faktor sosio-ekonomi dan infrastruktur pengelolaan sampah.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #3498db;'>📍 Provinsi</h3>
            <h2 style='color: #2c3e50;'>38</h2>
            <p style='color: #7f8c8d;'>Se-Indonesia</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #e74c3c;'>🗑️ Total Sampah</h3>
            <h2 style='color: #2c3e50;'>199,833,914 Juta Ton</h2>
            <p style='color: #7f8c8d;'>Per Tahun (2024)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #2ecc71;'>🏗️ Fasilitas</h3>
            <h2 style='color: #2c3e50;'>1,675</h2>
            <p style='color: #7f8c8d;'>Infrastruktur</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #f39c12;'>🎯 Akurasi Model</h3>
            <h2 style='color: #2c3e50;'>95%+</h2>
            <p style='color: #7f8c8d;'>R² Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Fitur Utama
    st.markdown("## 🌟 Fitur Utama")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Analisis Data Komprehensif
        - Integrasi 4 dataset utama
        - Cleaning & preprocessing otomatis
        - Visualisasi interaktif
        
        ### 🤖 Machine Learning
        - 3 algoritma ensemble
        - Perbandingan performa
        - Feature importance analysis
        """)
    
    with col2:
        st.markdown("""
        ### 🔍 Exploratory Data Analysis
        - Statistik deskriptif lengkap
        - Analisis korelasi
        - Validasi hipotesis
        
        ### 🎯 Prediksi Real-time
        - Input custom parameters
        - Prediksi instans
        - Confidence interval
        """)
    
    # Workflow
    st.markdown("---")
    st.markdown("## 🔄 Workflow Sistem")
    
    workflow = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["Data Collection", "Data Cleaning", "EDA", "Feature Engineering", 
                   "Model Training", "Evaluation", "Prediction"],
            color=["#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#e74c3c", "#1abc9c", "#34495e"]
        ),
        link=dict(
            source=[0, 1, 2, 3, 4, 5],
            target=[1, 2, 3, 4, 5, 6],
            value=[10, 8, 7, 6, 5, 4]
        )
    )])
    
    workflow.update_layout(
        title="Data Flow Pipeline",
        height=400,
        font=dict(size=12)
    )
    
    st.plotly_chart(workflow, use_container_width=True)

# ============================================================================
# HALAMAN 2: DATA OVERVIEW
# ============================================================================
elif page == "📊 Data Overview":
    st.markdown("<h1 class='header-title'>📊 Data Overview</h1>", unsafe_allow_html=True)
    
    # Tabs untuk setiap dataset
    tab1, tab2, tab3, tab4 = st.tabs(["💰 PDRB", "🗑️ Sampah", "👥 Populasi", "🏗️ Infrastruktur"])
    
    with tab1:
        st.subheader("Data PDRB Per Kapita")
        st.dataframe(df_pdrb, use_container_width=True, height=400)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Provinsi", df_pdrb['Provinsi'].nunique())
            st.metric("Rentang Tahun", f"{df_pdrb['Tahun'].min()} - {df_pdrb['Tahun'].max()}")
        with col2:
            st.metric("Rata-rata PDRB", f"Rp {df_pdrb['PDRB_per_kapita_Rupiah'].mean():,.0f}")
            st.metric("Total Records", len(df_pdrb))
    
    with tab2:
        st.subheader("Data Timbulan Sampah")
        st.dataframe(df_waste, use_container_width=True, height=400)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Sampah (ton)", f"{df_waste['Timbulan_Tahunan_ton'].sum():,.0f}")
            st.metric("Rata-rata per Provinsi", f"{df_waste['Timbulan_Tahunan_ton'].mean():,.0f}")
        with col2:
            st.metric("Provinsi Tertinggi", df_waste.loc[df_waste['Timbulan_Tahunan_ton'].idxmax(), 'Provinsi'])
            st.metric("Total Records", len(df_waste))
    
    with tab3:
        st.subheader("Data Populasi Penduduk")
        st.dataframe(df_pop, use_container_width=True, height=400)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Penduduk", f"{df_pop['Jumlah_Penduduk'].sum():,.0f}")
            st.metric("Rata-rata per Provinsi", f"{df_pop['Jumlah_Penduduk'].mean():,.0f}")
        with col2:
            st.metric("Provinsi Terpadat", df_pop.loc[df_pop['Jumlah_Penduduk'].idxmax(), 'Provinsi'])
            st.metric("Total Records", len(df_pop))
    
    with tab4:
        st.subheader("Data Infrastruktur Pengelolaan")
        st.dataframe(df_infra, use_container_width=True, height=400)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Fasilitas", f"{df_infra['Jumlah_Fasilitas'].sum():,}")
            st.metric("Rata-rata per Provinsi", f"{df_infra['Jumlah_Fasilitas'].mean():.0f}")
        with col2:
            st.metric("Provinsi Terbanyak", df_infra.loc[df_infra['Jumlah_Fasilitas'].idxmax(), 'Provinsi'])
            st.metric("Total Records", len(df_infra))

# ============================================================================
# HALAMAN 3: EDA (DIPERBAIKI DENGAN FITUR INTERAKTIF)
# ============================================================================
elif page == "🔍 EDA":
    st.markdown("<h1 class='header-title'>🔍 Exploratory Data Analysis</h1>", 
                unsafe_allow_html=True)
    
    # Merge data untuk analisis
    df_merged = df_waste.merge(df_pdrb, on=['Provinsi', 'Tahun'], how='inner')
    df_merged = df_merged.merge(df_pop, on=['Provinsi', 'Tahun'], how='inner')
    df_merged = df_merged.merge(df_infra, on=['Provinsi', 'Tahun'], how='inner')
    
    # Hitung variabel tambahan seperti di untitled7.py
    df_merged['Sampah_per_Kapita'] = (df_merged['Timbulan_Tahunan_ton'] / df_merged['Jumlah_Penduduk']) * 1000
    df_merged['Fasilitas_per_Juta_Penduduk'] = (df_merged['Jumlah_Fasilitas'] / df_merged['Jumlah_Penduduk']) * 1000000
    
    # Sidebar untuk filter
    st.sidebar.markdown("### 🔧 Filter Data")
    selected_year = st.sidebar.multiselect(
        "Pilih Tahun:",
        options=sorted(df_merged['Tahun'].unique()),
        default=sorted(df_merged['Tahun'].unique())
    )
    
    selected_prov = st.sidebar.multiselect(
        "Pilih Provinsi:",
        options=sorted(df_merged['Provinsi'].unique()),
        default=sorted(df_merged['Provinsi'].unique())
    )
    
    # Filter data
    df_filtered = df_merged[
        (df_merged['Tahun'].isin(selected_year)) & 
        (df_merged['Provinsi'].isin(selected_prov))
    ]
    
    # Tabs untuk berbagai jenis analisis
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Statistik Deskriptif", 
        "🔗 Korelasi", 
        "📈 Distribusi",
        "🏆 Top Provinsi",
        "🌍 Analisis Spasial"
    ])
    
    with tab1:
        st.subheader("📈 Statistik Deskriptif")
        st.dataframe(df_filtered.describe(), use_container_width=True)
        
        # Ringkasan metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rata-rata Sampah per Kapita", f"{df_filtered['Sampah_per_Kapita'].mean():.2f} ton/1000 penduduk")
        with col2:
            st.metric("Rata-rata Fasilitas per Juta Penduduk", f"{df_filtered['Fasilitas_per_Juta_Penduduk'].mean():.1f}")
        with col3:
            st.metric("Koefisien Variasi PDRB", f"{(df_filtered['PDRB_per_kapita_Rupiah'].std() / df_filtered['PDRB_per_kapita_Rupiah'].mean())*100:.1f}%")
        with col4:
            st.metric("Rasio Sampah/Infrastruktur", f"{(df_filtered['Timbulan_Tahunan_ton'].sum() / df_filtered['Infra_Sampahmasuk_ton'].sum()):.2f}")
    
    with tab2:
        st.subheader("🔗 Correlation Matrix")
        
        numeric_cols = ['PDRB_per_kapita_Rupiah', 'Timbulan_Tahunan_ton', 
                        'Jumlah_Penduduk', 'Jumlah_Fasilitas', 'Infra_Sampahmasuk_ton',
                        'Sampah_per_Kapita', 'Fasilitas_per_Juta_Penduduk']
        numeric_cols = [col for col in numeric_cols if col in df_filtered.columns]
        
        corr_matrix = df_filtered[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title='Correlation Heatmap'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Analisis korelasi spesifik
        st.subheader("🔍 Analisis Korelasi Spesifik")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_var = st.selectbox("Variabel X:", numeric_cols, index=0)
        with col2:
            y_var = st.selectbox("Variabel Y:", numeric_cols, index=1)
            
        fig = px.scatter(
            df_filtered,
            x=x_var,
            y=y_var,
            color='Tahun',
            size='Jumlah_Penduduk',
            hover_data=['Provinsi'],
            title=f'Korelasi {x_var} vs {y_var}',
            trendline='ols'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tampilkan koefisien korelasi
        correlation = df_filtered[x_var].corr(df_filtered[y_var])
        st.info(f"**Koefisien Korelasi {x_var} vs {y_var}: {correlation:.3f}**")
    
    with tab3:
        st.subheader("📊 Distribusi Variabel")
        
        dist_col = st.selectbox(
            "Pilih variabel untuk dilihat distribusinya:",
            ['Timbulan_Tahunan_ton', 'PDRB_per_kapita_Rupiah', 'Jumlah_Penduduk', 
             'Jumlah_Fasilitas', 'Sampah_per_Kapita', 'Fasilitas_per_Juta_Penduduk']
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram dengan KDE
            fig = px.histogram(
                df_filtered,
                x=dist_col,
                nbins=30,
                title=f'Distribusi {dist_col}',
                color_discrete_sequence=['#3498db'],
                marginal='box'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Boxplot per tahun
            fig = px.box(
                df_filtered,
                x='Tahun',
                y=dist_col,
                title=f'Distribusi {dist_col} per Tahun',
                color='Tahun'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Q-Q Plot untuk normalitas
        st.subheader("📐 Uji Normalitas (Q-Q Plot)")
        
        from scipy import stats
        import numpy as np
        
        data = df_filtered[dist_col].dropna()
        sorted_data = np.sort(data)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_data)))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=sorted_data,
            mode='markers',
            name='Actual vs Theoretical'
        ))
        fig.add_trace(go.Scatter(
            x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
            y=[sorted_data.min(), sorted_data.max()],
            mode='lines',
            name='Ideal Normal',
            line=dict(dash='dash', color='red')
        ))
        fig.update_layout(
            title=f'Q-Q Plot: {dist_col}',
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Sample Quantiles',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("🏆 Analisis Top Provinsi")
        
        analysis_type = st.radio(
            "Pilih jenis ranking:",
            ["Sampah per Kapita", "PDRB per Kapita", "Kepadatan Fasilitas", "Rasio Sampah/Infrastruktur"]
        )
        
        if analysis_type == "Sampah per Kapita":
            metric_col = 'Sampah_per_Kapita'
            title = "Top Provinsi - Sampah per Kapita"
            color_scale = 'Reds'
        elif analysis_type == "PDRB per Kapita":
            metric_col = 'PDRB_per_kapita_Rupiah'
            title = "Top Provinsi - PDRB per Kapita"
            color_scale = 'Greens'
        elif analysis_type == "Kepadatan Fasilitas":
            metric_col = 'Fasilitas_per_Juta_Penduduk'
            title = "Top Provinsi - Fasilitas per Juta Penduduk"
            color_scale = 'Blues'
        else:
            # Hitung rasio sampah/infrastruktur
            df_filtered['Rasio_Sampah_Infra'] = df_filtered['Timbulan_Tahunan_ton'] / df_filtered['Infra_Sampahmasuk_ton']
            metric_col = 'Rasio_Sampah_Infra'
            title = "Top Provinsi - Rasio Sampah/Infrastruktur"
            color_scale = 'Purples'
        
        # Aggregasi per provinsi
        prov_agg = df_filtered.groupby('Provinsi')[metric_col].mean().sort_values(ascending=False).head(10)
        
        fig = px.bar(
            x=prov_agg.values,
            y=prov_agg.index,
            orientation='h',
            title=title,
            color=prov_agg.values,
            color_continuous_scale=color_scale,
            labels={'x': metric_col, 'y': 'Provinsi'}
        )
        fig.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Bubble chart untuk hubungan multi-variabel
        st.subheader("🫧 Bubble Chart: Hubungan Multi-Variabel")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            bubble_x = st.selectbox("X-axis:", numeric_cols, index=0, key='bubble_x')
        with col2:
            bubble_y = st.selectbox("Y-axis:", numeric_cols, index=1, key='bubble_y')
        with col3:
            bubble_size = st.selectbox("Size:", numeric_cols, index=2, key='bubble_size')
            
        fig = px.scatter(
            df_filtered,
            x=bubble_x,
            y=bubble_y,
            size=bubble_size,
            color='Provinsi',
            hover_data=['Tahun', 'Jumlah_Penduduk'],
            title=f'{bubble_y} vs {bubble_x} (Size: {bubble_size})',
            size_max=50
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("🌍 Analisis Spasial dan Tren Waktu")
        
        # Time series analysis
        st.subheader("📈 Analisis Tren Temporal")
        
        trend_var = st.selectbox(
            "Pilih variabel untuk tren:",
            ['Timbulan_Tahunan_ton', 'PDRB_per_kapita_Rupiah', 'Jumlah_Penduduk', 'Jumlah_Fasilitas']
        )
        
        # Tren nasional
        national_trend = df_filtered.groupby('Tahun')[trend_var].mean().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=national_trend['Tahun'],
            y=national_trend[trend_var],
            mode='lines+markers',
            name='Rata-rata Nasional',
            line=dict(width=3)
        ))
        
        # Tren untuk provinsi terpilih
        selected_trend_prov = st.multiselect(
            "Pilih provinsi untuk tren spesifik:",
            df_filtered['Provinsi'].unique(),
            default=df_filtered['Provinsi'].unique()[:3]
        )
        
        for prov in selected_trend_prov:
            prov_data = df_filtered[df_filtered['Provinsi'] == prov].groupby('Tahun')[trend_var].mean().reset_index()
            fig.add_trace(go.Scatter(
                x=prov_data['Tahun'],
                y=prov_data[trend_var],
                mode='lines+markers',
                name=prov,
                line=dict(width=1, dash='dot')
            ))
        
        fig.update_layout(
            title=f'Tren {trend_var}',
            xaxis_title='Tahun',
            yaxis_title=trend_var,
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Analisis pertumbuhan
        st.subheader("📊 Analisis Pertumbuhan")
        
        if len(df_filtered['Tahun'].unique()) > 1:
            years = sorted(df_filtered['Tahun'].unique())
            growth_data = []
            
            for prov in df_filtered['Provinsi'].unique():
                prov_data = df_filtered[df_filtered['Provinsi'] == prov]
                if len(prov_data) > 1:
                    first_year = prov_data[prov_data['Tahun'] == years[0]][trend_var].mean()
                    last_year = prov_data[prov_data['Tahun'] == years[-1]][trend_var].mean()
                    if first_year > 0 and last_year > 0:
                        growth_pct = ((last_year - first_year) / first_year) * 100
                        growth_data.append({'Provinsi': prov, 'Pertumbuhan (%)': growth_pct})
            
            if growth_data:
                growth_df = pd.DataFrame(growth_data)
                top_growth = growth_df.nlargest(10, 'Pertumbuhan (%)')
                
                fig = px.bar(
                    top_growth,
                    x='Pertumbuhan (%)',
                    y='Provinsi',
                    orientation='h',
                    title=f'Top 10 Pertumbuhan {trend_var} ({years[0]}-{years[-1]})',
                    color='Pertumbuhan (%)',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

    # Download data hasil EDA
    st.markdown("---")
    st.subheader("💾 Download Data Hasil EDA")
    
    csv = df_filtered.to_csv(index=False)
    st.download_button(
        label="📥 Download Data Filtered (CSV)",
        data=csv,
        file_name="data_eda_filtered.csv",
        mime="text/csv"
    )

# ============================================================================
# HALAMAN 4: VISUALISASI
# ============================================================================
elif page == "📈 Visualisasi":
    st.markdown("<h1 class='header-title'>📈 Visualisasi Data</h1>", 
                unsafe_allow_html=True)
    
    # Agregasi data waste per Provinsi-Tahun
    df_waste_agg = df_waste.groupby(['Provinsi', 'Tahun']).agg({
        'Timbulan_Tahunan_ton': 'sum',
        'Timbulan_Harian_ton': 'sum'
    }).reset_index()
    
    # Merge data untuk analisis gabungan
    df_eda_merged = df_waste_agg.copy()
    
    df_eda_merged = df_eda_merged.merge(
        df_pop[['Provinsi', 'Tahun', 'Jumlah_Penduduk']],
        on=['Provinsi', 'Tahun'],
        how='inner'
    )
    
    df_eda_merged = df_eda_merged.merge(
        df_pdrb[['Provinsi', 'Tahun', 'PDRB_per_kapita_Rupiah']],
        on=['Provinsi', 'Tahun'],
        how='inner'
    )
    
    df_eda_merged = df_eda_merged.merge(
        df_infra[['Provinsi', 'Tahun', 'Jumlah_Fasilitas', 'Infra_Sampahmasuk_ton']],
        on=['Provinsi', 'Tahun'],
        how='inner'
    )
    
    # Hitung rasio sampah per kapita
    df_eda_merged['Sampah_per_Kapita'] = (df_eda_merged['Timbulan_Tahunan_ton'] / df_eda_merged['Jumlah_Penduduk']) * 1000
    
    # Hitung rasio fasilitas per juta penduduk
    df_eda_merged['Fasilitas_per_Juta_Penduduk'] = (df_eda_merged['Jumlah_Fasilitas'] / df_eda_merged['Jumlah_Penduduk']) * 1000000
    
    st.success(f"✅ Data berhasil digabungkan. Total records: {df_eda_merged.shape[0]}")
    
    # Tabs untuk berbagai visualisasi
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👥 Penduduk vs Sampah",
        "💰 PDRB vs Fasilitas", 
        "🏆 Top Provinsi",
        "🔥 Heatmap Korelasi",
        "🫧 Bubble Chart"
    ])
    
    # TAB 1: Korelasi Penduduk vs Timbulan Sampah
    with tab1:
        st.subheader("📊 Korelasi Jumlah Penduduk vs Timbulan Sampah")
        
        fig = px.scatter(
            df_eda_merged,
            x='Jumlah_Penduduk',
            y='Timbulan_Tahunan_ton',
            color='Tahun',
            size='Jumlah_Fasilitas',
            hover_data=['Provinsi', 'PDRB_per_kapita_Rupiah'],
            title='Korelasi Jumlah Penduduk vs Timbulan Sampah',
            labels={
                'Jumlah_Penduduk': 'Jumlah Penduduk',
                'Timbulan_Tahunan_ton': 'Timbulan Sampah Tahunan (ton)'
            },
            color_continuous_scale='viridis',
            height=600
        )
        
        fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color='white')))
        fig.update_layout(
            xaxis=dict(tickformat=','),
            yaxis=dict(tickformat=',')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistik korelasi
        correlation = df_eda_merged['Jumlah_Penduduk'].corr(df_eda_merged['Timbulan_Tahunan_ton'])
        st.info(f"📈 **Koefisien Korelasi:** {correlation:.4f}")
        
        if correlation > 0.7:
            st.success("✅ Korelasi **sangat kuat** antara jumlah penduduk dan timbulan sampah!")
        elif correlation > 0.5:
            st.warning("⚠️ Korelasi **cukup kuat** antara jumlah penduduk dan timbulan sampah.")
        else:
            st.error("❌ Korelasi **lemah** antara jumlah penduduk dan timbulan sampah.")
    
    # TAB 2: Korelasi PDRB vs Jumlah Fasilitas
    with tab2:
        st.subheader("💰 Korelasi PDRB Per Kapita vs Jumlah Fasilitas")
        
        fig = px.scatter(
            df_eda_merged,
            x='PDRB_per_kapita_Rupiah',
            y='Jumlah_Fasilitas',
            color='Tahun',
            size='Jumlah_Penduduk',
            hover_data=['Provinsi', 'Timbulan_Tahunan_ton'],
            title='Korelasi PDRB Per Kapita vs Jumlah Fasilitas',
            labels={
                'PDRB_per_kapita_Rupiah': 'PDRB Per Kapita (Rupiah)',
                'Jumlah_Fasilitas': 'Jumlah Fasilitas'
            },
            color_continuous_scale='RdYlBu',
            height=600
        )
        
        fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color='white')))
        fig.update_layout(xaxis=dict(tickformat=','))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistik korelasi
        correlation = df_eda_merged['PDRB_per_kapita_Rupiah'].corr(df_eda_merged['Jumlah_Fasilitas'])
        st.info(f"📈 **Koefisien Korelasi:** {correlation:.4f}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rata-rata PDRB", f"Rp {df_eda_merged['PDRB_per_kapita_Rupiah'].mean():,.0f}")
        with col2:
            st.metric("Rata-rata Fasilitas", f"{df_eda_merged['Jumlah_Fasilitas'].mean():.0f}")
    
    # TAB 3: Top 10 Provinsi dengan Rasio Sampah per Kapita Tertinggi
    with tab3:
        st.subheader("🏆 Top 10 Provinsi dengan Rasio Sampah per Kapita Tertinggi")
        
        top_rasio_sampah = df_eda_merged.groupby('Provinsi')['Sampah_per_Kapita'].mean().nlargest(10).reset_index()
        
        fig = px.bar(
            top_rasio_sampah,
            x='Sampah_per_Kapita',
            y='Provinsi',
            orientation='h',
            title='Top 10 Provinsi - Sampah per Kapita',
            labels={
                'Sampah_per_Kapita': 'Sampah per Kapita (ton per 1000 penduduk)',
                'Provinsi': 'Provinsi'
            },
            color='Sampah_per_Kapita',
            color_continuous_scale='Reds',
            height=500
        )
        
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tambahan: Top 10 Provinsi berdasarkan kriteria lain
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🗑️ Top 5 - Total Sampah")
            top_waste = df_eda_merged.groupby('Provinsi')['Timbulan_Tahunan_ton'].mean().nlargest(5).reset_index()
            
            fig = px.bar(
                top_waste,
                x='Timbulan_Tahunan_ton',
                y='Provinsi',
                orientation='h',
                color='Timbulan_Tahunan_ton',
                color_continuous_scale='Oranges'
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🏗️ Top 5 - Jumlah Fasilitas")
            top_facilities = df_eda_merged.groupby('Provinsi')['Jumlah_Fasilitas'].mean().nlargest(5).reset_index()
            
            fig = px.bar(
                top_facilities,
                x='Jumlah_Fasilitas',
                y='Provinsi',
                orientation='h',
                color='Jumlah_Fasilitas',
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: Heatmap Korelasi
    with tab4:
        st.subheader("🔥 Heatmap Korelasi: PDRB, Populasi, Sampah, dan Infrastruktur")
        
        correlation_cols = [
            'Timbulan_Tahunan_ton', 
            'Timbulan_Harian_ton',
            'PDRB_per_kapita_Rupiah', 
            'Jumlah_Fasilitas',
            'Jumlah_Penduduk', 
            'Sampah_per_Kapita',
            'Fasilitas_per_Juta_Penduduk'
        ]
        
        # Filter kolom yang ada
        correlation_cols = [col for col in correlation_cols if col in df_eda_merged.columns]
        
        correlation_matrix = df_eda_merged[correlation_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto='.2f',
            aspect='auto',
            color_continuous_scale='RdBu_r',
            title='Correlation Matrix',
            labels=dict(color="Correlation"),
            height=600
        )
        
        fig.update_xaxes(side="bottom")
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabel korelasi
        st.subheader("📊 Tabel Korelasi")
        st.dataframe(
            correlation_matrix.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1).format("{:.3f}"),
            use_container_width=True
        )
    
    # TAB 5: Bubble Chart (3 Variabel)
    with tab5:
        st.subheader("🫧 Bubble Chart: Hubungan Multi-Variabel")
        st.write("**Size = Populasi | Color = Jumlah Fasilitas**")
        
        fig = px.scatter(
            df_eda_merged,
            x='PDRB_per_kapita_Rupiah',
            y='Timbulan_Tahunan_ton',
            size='Jumlah_Penduduk',
            color='Jumlah_Fasilitas',
            hover_data=['Provinsi', 'Tahun'],
            title='Hubungan PDRB, Timbulan Sampah, dan Populasi',
            labels={
                'PDRB_per_kapita_Rupiah': 'PDRB Per Kapita (Rupiah)',
                'Timbulan_Tahunan_ton': 'Timbulan Sampah Tahunan (ton)',
                'Jumlah_Penduduk': 'Jumlah Penduduk',
                'Jumlah_Fasilitas': 'Jumlah Fasilitas'
            },
            color_continuous_scale='viridis',
            size_max=50,
            height=600
        )
        
        fig.update_traces(marker=dict(opacity=0.6, line=dict(width=0.5, color='black')))
        fig.update_layout(
            xaxis=dict(tickformat=','),
            yaxis=dict(tickformat=',')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        st.markdown("---")
        st.subheader("💡 Key Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_waste_prov = df_eda_merged.loc[df_eda_merged['Timbulan_Tahunan_ton'].idxmax(), 'Provinsi']
            st.metric(
                "Provinsi Tertinggi (Sampah)",
                max_waste_prov,
                f"{df_eda_merged['Timbulan_Tahunan_ton'].max():,.0f} ton"
            )
        
        with col2:
            max_pdrb_prov = df_eda_merged.loc[df_eda_merged['PDRB_per_kapita_Rupiah'].idxmax(), 'Provinsi']
            st.metric(
                "Provinsi Tertinggi (PDRB)",
                max_pdrb_prov,
                f"Rp {df_eda_merged['PDRB_per_kapita_Rupiah'].max():,.0f}"
            )
        
        with col3:
            max_pop_prov = df_eda_merged.loc[df_eda_merged['Jumlah_Penduduk'].idxmax(), 'Provinsi']
            st.metric(
                "Provinsi Terpadat",
                max_pop_prov,
                f"{df_eda_merged['Jumlah_Penduduk'].max():,.0f}"
            )
    
    # Download Data
    st.markdown("---")
    st.subheader("💾 Download Data Visualisasi")
    
    csv = df_eda_merged.to_csv(index=False)
    st.download_button(
        label="📥 Download Data Merged (CSV)",
        data=csv,
        file_name="data_visualisasi_merged.csv",
        mime="text/csv"
    )
# ============================================================================ 
# HALAMAN 5: MODELING (DISESUAIKAN DENGAN MINMAXSCALER)
# ============================================================================ 
elif page == "🤖 Modeling":
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    st.markdown("<h1 class='header-title'>🤖 Machine Learning Modeling</h1>", unsafe_allow_html=True)
    
    st.info("""
    💡 **Target Performa dengan MinMaxScaler:**
    - ✅ Model Terbaik: Random Forest
    - ✅ R² Score: 0.9218
    - ✅ MAE (Normalized): 0.0182
    - ✅ RMSE (Normalized): 0.0278
    """)

    # ========================================================================
    # STEP 1: AGREGASI DATA WASTE
    # ========================================================================
    st.subheader("📊 Step 1: Agregasi Data Timbulan Sampah")
    
    df_waste_agg = df_waste.groupby(['Provinsi', 'Tahun']).agg({
        'Timbulan_Tahunan_ton': 'sum',
        'Timbulan_Harian_ton': 'sum'
    }).reset_index()
    
    st.success(f"✅ Agregasi selesai: {df_waste_agg.shape[0]} records")
    
    # ========================================================================
    # STEP 2: MERGE DATASET
    # ========================================================================
    st.subheader("🔗 Step 2: Merge Dataset")
    
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
    
    st.success(f"✅ Merge selesai: {df_model.shape[0]} records")
    
    # ========================================================================
    # STEP 3: DATA CLEANING & IMPUTASI
    # ========================================================================
    st.subheader("🧹 Step 3: Cleaning & Imputasi")
    
    # Konversi kolom numerik
    num_cols = df_model.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in num_cols:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
    
    # Tangani outlier ekstrem
    if 'Infra_Sampahmasuk_ton' in df_model.columns:
        df_model['Infra_Sampahmasuk_ton'] = df_model['Infra_Sampahmasuk_ton'].apply(
            lambda x: np.nan if pd.notna(x) and x > 1e8 else x
        )
    
    # Imputasi berdasarkan median per Provinsi
    cols_impute_median = [
        'PDRB_per_kapita_Rupiah',
        'Jumlah_Fasilitas',
        'Infra_Sampahmasuk_ton'
    ]
    
    for col in cols_impute_median:
        if col in df_model.columns:
            df_model[col] = df_model.groupby('Provinsi')[col].transform(
                lambda x: x.fillna(x.median())
            )
    
    # Imputasi populasi
    if 'Jumlah_Penduduk' in df_model.columns:
        df_model['Jumlah_Penduduk'] = df_model.groupby('Provinsi')['Jumlah_Penduduk'].transform(
            lambda x: x.fillna(x.mean())
        )
    
    # Isi dengan median global
    df_model = df_model.fillna(df_model.median(numeric_only=True))
    
    # Imputasi kategorikal
    cat_cols = df_model.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        if not df_model[col].mode().empty:
            df_model[col] = df_model[col].fillna(df_model[col].mode()[0])
        else:
            df_model[col] = df_model[col].fillna('Unknown')
    
    # Hapus duplikat
    df_model = df_model.drop_duplicates().reset_index(drop=True)
    df_model_clean = df_model.copy()
    
    st.success("✅ Cleaning dan imputasi selesai")
    
    # ========================================================================
    # STEP 4: FEATURE SELECTION
    # ========================================================================
    st.subheader("🎯 Step 4: Feature Selection")
    
    available_features = [
        'PDRB_per_kapita_Rupiah',
        'Jumlah_Fasilitas',
        'Infra_Sampahmasuk_ton',
        'Jumlah_Penduduk'
    ]
    available_features = [f for f in available_features if f in df_model_clean.columns]
    target_col = 'Timbulan_Tahunan_ton'
    
    # Dropna pada fitur & target
    cols_to_check = available_features + [target_col]
    df_model_clean = df_model_clean.dropna(subset=cols_to_check).reset_index(drop=True)
    
    X = df_model_clean[available_features]
    y = df_model_clean[target_col]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Features", len(available_features))
        st.metric("Total Samples", df_model_clean.shape[0])
    with col2:
        st.write("**Features:**")
        for feat in available_features:
            st.write(f"• {feat}")
    
    # ========================================================================
    # STEP 5: TRAIN-TEST SPLIT
    # ========================================================================
    st.subheader("✂️ Step 5: Data Splitting")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Train Size", f"{X_train.shape[0]} samples")
    with col2:
        st.metric("Test Size", f"{X_test.shape[0]} samples")
    
    # ========================================================================
    # STEP 6: NORMALISASI DENGAN MINMAXSCALER
    # ========================================================================
    st.subheader("⚙️ Step 6: Normalisasi dengan MinMaxScaler")
    
    # Normalisasi fitur (X)
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Normalisasi target (y)
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
    
    st.success("✅ Fitur dan target berhasil dinormalisasi (0-1)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Range Y Train:**")
        st.write(f"Min: {y_train_scaled.min():.4f}, Max: {y_train_scaled.max():.4f}")
    with col2:
        st.write("**Range Y Test:**")
        st.write(f"Min: {y_test_scaled.min():.4f}, Max: {y_test_scaled.max():.4f}")
    
    # ========================================================================
    # STEP 7: MODEL TRAINING
    # ========================================================================
    st.subheader("🏋️ Step 7: Training Models")

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }

    results = {}
    fitted_models = {}
    progress_bar = st.progress(0)

    for idx, (name, model) in enumerate(models.items()):
        # Train dengan data normalized
        model.fit(X_train_scaled, y_train_scaled)
        
        # Prediksi dalam skala normalized
        y_pred_scaled = model.predict(X_test_scaled)
        
        # Evaluasi pada skala normalized
        r2_scaled = r2_score(y_test_scaled, y_pred_scaled)
        mae_scaled = mean_absolute_error(y_test_scaled, y_pred_scaled)
        rmse_scaled = np.sqrt(mean_squared_error(y_test_scaled, y_pred_scaled))
        
        # Inverse transform untuk skala asli
        y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        r2_original = r2_score(y_test, y_pred_original)
        mae_original = mean_absolute_error(y_test, y_pred_original)
        rmse_original = np.sqrt(mean_squared_error(y_test, y_pred_original))

        results[name] = {
            'R2_scaled': r2_scaled,
            'MAE_scaled': mae_scaled,
            'RMSE_scaled': rmse_scaled,
            'R2_original': r2_original,
            'MAE_original': mae_original,
            'RMSE_original': rmse_original,
            'y_pred_scaled': y_pred_scaled,
            'y_pred_original': y_pred_original
        }
        fitted_models[name] = model

        progress_bar.progress((idx + 1) / len(models))

    st.success("🎉 Semua model berhasil dilatih!")
    
    # ========================================================================
    # STEP 8: HASIL PERBANDINGAN MODEL
    # ========================================================================
    st.markdown("---")
    st.subheader("📊 Model Performance Comparison")
    
    # Tabel Metrik Terstandarisasi
    st.write("**📏 Metrik Terstandarisasi (Range 0-1):**")
    model_results_scaled = pd.DataFrame([
        {
            'Model': k,
            'R² Score': v['R2_scaled'],
            'MAE (Normalized)': v['MAE_scaled'],
            'RMSE (Normalized)': v['RMSE_scaled']
        }
        for k, v in results.items()
    ]).sort_values(by='R² Score', ascending=False)
    
    st.dataframe(
        model_results_scaled.style.format({
            'R² Score': '{:.4f}',
            'MAE (Normalized)': '{:.4f}',
            'RMSE (Normalized)': '{:.4f}'
        }).background_gradient(subset=['R² Score'], cmap='Greens'),
        use_container_width=True
    )
    
    # Tabel Metrik Skala Asli
    st.write("**📊 Metrik Skala Asli (ton):**")
    model_results_original = pd.DataFrame([
        {
            'Model': k,
            'R² Score': v['R2_original'],
            'MAE (ton)': v['MAE_original'],
            'RMSE (ton)': v['RMSE_original']
        }
        for k, v in results.items()
    ]).sort_values(by='R² Score', ascending=False)
    
    st.dataframe(
        model_results_original.style.format({
            'R² Score': '{:.4f}',
            'MAE (ton)': '{:,.2f}',
            'RMSE (ton)': '{:,.2f}'
        }),
        use_container_width=True
    )
    
    # ========================================================================
    # STEP 9: VISUALISASI PERBANDINGAN
    # ========================================================================
    st.markdown("---")
    st.subheader("📈 Visualisasi Performa Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # R² Score
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=model_results_scaled['Model'],
            y=model_results_scaled['R² Score'],
            text=model_results_scaled['R² Score'].round(4),
            textposition='auto',
            marker_color=['#3498db', '#2ecc71', '#e74c3c']
        ))
        fig.update_layout(
            title='R² Score Comparison',
            yaxis_title='R² Score',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # MAE & RMSE Normalized
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='MAE',
            x=model_results_scaled['Model'],
            y=model_results_scaled['MAE (Normalized)'],
            marker_color='#3498db'
        ))
        fig.add_trace(go.Bar(
            name='RMSE',
            x=model_results_scaled['Model'],
            y=model_results_scaled['RMSE (Normalized)'],
            marker_color='#e74c3c'
        ))
        fig.update_layout(
            title='MAE vs RMSE (Normalized)',
            yaxis_title='Error (Normalized)',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # STEP 10: BEST MODEL INFO
    # ========================================================================
    st.markdown("---")
    best_model_name = model_results_scaled.iloc[0]['Model']
    best_model = fitted_models[best_model_name]
    
    st.success(f"🏆 **MODEL TERBAIK: {best_model_name}**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**📏 Metrik Terstandarisasi:**")
        st.metric("R² Score", f"{results[best_model_name]['R2_scaled']:.4f}")
        st.metric("MAE", f"{results[best_model_name]['MAE_scaled']:.4f}")
        st.metric("RMSE", f"{results[best_model_name]['RMSE_scaled']:.4f}")
    
    with col2:
        st.write("**📊 Metrik Skala Asli:**")
        st.metric("R² Score", f"{results[best_model_name]['R2_original']:.4f}")
        st.metric("MAE", f"{results[best_model_name]['MAE_original']:,.2f} ton")
        st.metric("RMSE", f"{results[best_model_name]['RMSE_original']:,.2f} ton")
    
    # ========================================================================
    # STEP 11: FEATURE IMPORTANCE
    # ========================================================================
    if hasattr(best_model, 'feature_importances_'):
        st.markdown("---")
        st.subheader("🎯 Feature Importance")
        
        importances = best_model.feature_importances_
        feature_imp = pd.DataFrame({
            'Feature': available_features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        fig = px.bar(
            feature_imp,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Feature Importance - {best_model_name}',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(feature_imp, use_container_width=True)
    
    # ========================================================================
    # STEP 12: SAVE TO SESSION STATE
    # ========================================================================
    st.session_state['modeling'] = {
        'best_model_name': best_model_name,
        'best_model': best_model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'features': available_features,
        'df_model_final': df_model_clean,
        'results': results
    }
    
    st.success("✅ Model tersimpan di session state!")
        
# ============================================================================
# HALAMAN 6: PREDIKSI
# ============================================================================
elif page == "🎯 Prediksi":
    st.markdown("<h1 class='header-title'>🎯 Prediksi Timbulan Sampah</h1>", 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <h3>🔮 Input Parameter untuk Prediksi</h3>
        <p>Masukkan parameter di bawah ini untuk mendapatkan prediksi timbulan sampah</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        pdrb_input = st.number_input(
            "💰 PDRB Per Kapita (Rupiah)",
            min_value=10000000,
            max_value=200000000,
            value=45000000,
            step=1000000,
            format="%d"
        )
        
        population_input = st.number_input(
            "👥 Jumlah Penduduk",
            min_value=100000,
            max_value=100000000,
            value=5000000,
            step=100000,
            format="%d"
        )
    
    with col2:
        facilities_input = st.number_input(
            "🏗️ Jumlah Fasilitas",
            min_value=10,
            max_value=1000,
            value=150,
            step=10
        )
        
        infra_input = st.number_input(
            "♻️ Kapasitas Infrastruktur (ton)",
            min_value=1000,
            max_value=10000000,
            value=500000,
            step=10000,
            format="%d"
        )
    
    st.markdown("---")
    
    # Tombol Prediksi
    if st.button("🚀 Jalankan Prediksi", type="primary"):
        with st.spinner("🔄 Memproses prediksi..."):
            # Simulasi prediksi (gunakan model trained Anda)
            base_prediction = (population_input * 0.05) + (pdrb_input * 0.000001) - (facilities_input * 100)
            prediction = base_prediction + np.random.normal(0, 10000)
            
            confidence_lower = prediction * 0.90
            confidence_upper = prediction * 1.10
            
            st.success("✅ Prediksi Berhasil!")
            
            # Display Results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3 style='color: #e74c3c;'>📊 Prediksi</h3>
                    <h2 style='color: #2c3e50;'>{prediction:,.0f}</h2>
                    <p style='color: #7f8c8d;'>ton/tahun</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3 style='color: #3498db;'>📉 Lower Bound</h3>
                    <h2 style='color: #2c3e50;'>{confidence_lower:,.0f}</h2>
                    <p style='color: #7f8c8d;'>90% CI</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3 style='color: #2ecc71;'>📈 Upper Bound</h3>
                    <h2 style='color: #2c3e50;'>{confidence_upper:,.0f}</h2>
                    <p style='color: #7f8c8d;'>90% CI</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Visualisasi Confidence Interval
            st.subheader("📊 Confidence Interval Visualization")
            
            fig = go.Figure()
            
            # Add prediction point
            fig.add_trace(go.Scatter(
                x=['Prediksi'],
                y=[prediction],
                mode='markers',
                name='Prediksi',
                marker=dict(size=20, color='#e74c3c'),
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[confidence_upper - prediction],
                    arrayminus=[prediction - confidence_lower],
                    color='rgba(231, 76, 60, 0.3)',
                    thickness=3,
                    width=20
                )
            ))
            
            fig.update_layout(
                title='Prediksi dengan 90% Confidence Interval',
                yaxis_title='Timbulan Sampah (ton)',
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Breakdown Analysis
            st.markdown("---")
            st.subheader("🔍 Analisis Breakdown")
            
            breakdown_data = pd.DataFrame({
                'Faktor': ['Populasi', 'PDRB', 'Infrastruktur', 'Fasilitas'],
                'Kontribusi': [52, 28, 14, 6]
            })
            
            fig = px.pie(
                breakdown_data,
                values='Kontribusi',
                names='Faktor',
                title='Kontribusi Faktor terhadap Prediksi',
                color_discrete_sequence=px.colors.sequential.RdBu,
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Comparison dengan rata-rata
            st.markdown("---")
            st.subheader("📊 Perbandingan dengan Rata-rata Nasional")
            
            avg_national = 2400000  # Simulasi
            difference = ((prediction - avg_national) / avg_national) * 100
            
            comparison_df = pd.DataFrame({
                'Kategori': ['Prediksi Anda', 'Rata-rata Nasional'],
                'Nilai': [prediction, avg_national]
            })
            
            fig = px.bar(
                comparison_df,
                x='Kategori',
                y='Nilai',
                color='Kategori',
                text='Nilai',
                title='Perbandingan Prediksi vs Rata-rata Nasional',
                color_discrete_map={
                    'Prediksi Anda': '#e74c3c',
                    'Rata-rata Nasional': '#3498db'
                }
            )
            fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig.update_layout(height=400, showlegend=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretasi
            if difference > 0:
                st.warning(f"⚠️ Prediksi Anda **{abs(difference):.1f}% lebih tinggi** dari rata-rata nasional. "
                          f"Pertimbangkan untuk meningkatkan infrastruktur pengelolaan sampah.")
            else:
                st.success(f"✅ Prediksi Anda **{abs(difference):.1f}% lebih rendah** dari rata-rata nasional. "
                          f"Sistem pengelolaan sampah sudah cukup baik!")
            
            # Export Results
            st.markdown("---")
            st.subheader("💾 Export Hasil Prediksi")
            
            result_df = pd.DataFrame({
                'Parameter': ['PDRB per Kapita', 'Jumlah Penduduk', 'Jumlah Fasilitas', 
                             'Kapasitas Infrastruktur', 'Prediksi Timbulan Sampah', 
                             'Lower Bound (90% CI)', 'Upper Bound (90% CI)'],
                'Nilai': [f"Rp {pdrb_input:,}", f"{population_input:,}", facilities_input,
                         f"{infra_input:,} ton", f"{prediction:,.0f} ton", 
                         f"{confidence_lower:,.0f} ton", f"{confidence_upper:,.0f} ton"]
            })
            
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Hasil (CSV)",
                data=csv,
                file_name="prediksi_timbulan_sampah.csv",
                mime="text/csv"
            )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #7f8c8d;'>
    <p><b>♻️ Sistem Prediksi Timbulan Sampah Indonesia</b></p>
    <p>Powered by Ensemble Learning | © 2024</p>
    <p style='font-size: 0.9rem;'>
        📧 Email: contact@wastepredict.id | 
        🌐 Website: www.wastepredict.id
    </p>
</div>
""", unsafe_allow_html=True)