import streamlit as st
import mysql.connector
import pandas as pd
from datetime import datetime

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Klinik Hewan",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling khusus
st.markdown(
    """
    <style>
    /* Judul utama */
    .main-title {
        font-size: 40px;
        color: #3D6CB9;
        text-align: center;
        font-weight: bold;
    }
    /* Deskripsi */
    .description {
        color: #6B7280;
        font-size: 18px;
        text-align: center;
        margin-bottom: 30px;
    }
    /* Subheader dan hasil filter */
    .subheader, .results-count {
        font-size: 24px;
        font-weight: bold;
        color: #2A4858;
        margin-top: 10px;
    }
    /* Tabel */
    .stTable table {
        border-collapse: collapse;
        width: 100%;
    }
    /* Header tabel */
    .stTable thead tr th {
        background-color: #3D6CB9;
        color: white;
        padding: 10px;
        font-size: 16px;
    }
    /* Baris ganjil */
    .stTable tbody tr:nth-child(odd) {
        background-color: #f8f9fa;
    }
    /* Baris genap */
    .stTable tbody tr:nth-child(even) {
        background-color: #ffffff;
    }
    /* Sel dan batas sel */
    .stTable tbody tr td, .stTable thead tr th {
        border: 1px solid #ddd;
        padding: 10px;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Judul dan deskripsi halaman
st.markdown('<div class="main-title">Dashboard Admin Klinik Hewan 🐾</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Selamat datang di halaman administrasi klinik hewan. Lihat dan kelola data janji temu dengan mudah!</div>', unsafe_allow_html=True)

# Fungsi untuk koneksi ke database MySQL
def get_database_connection():
    connection = mysql.connector.connect(
        host="localhost",
        database="klinik_hewan",
        user="root",
        password=""
    )
    return connection

# Fungsi untuk mengambil data janji temu dari database
def load_appointments():
    connection = get_database_connection()
    query = "SELECT id, nama_pemilik, jenis_hewan, waktu, keluhan,status FROM janji_temu"
    df = pd.read_sql(query, connection)
    connection.close()
    return df

# Subheader untuk data janji temu
st.markdown('<div class="subheader">Data Janji Temu</div>', unsafe_allow_html=True)

# Memuat data dari database
df_appointments = load_appointments()
df_appointments = df_appointments.rename(columns={
    'nama_pemilik': 'Nama Pemilik',
    'jenis_hewan': 'Jenis Hewan',
    'keluhan': 'Keluhan',
    'status': 'Status'
})

# Menghapus kolom 'id'
df_appointments = df_appointments.drop(columns=['id'])

# menghapus index tabel
# st.dataframe(df, hide_index=True)



# Sidebar untuk filter data
with st.sidebar:
    st.subheader("Filter Data Janji Temu")

    # Filter berdasarkan jenis hewan
    filter_pet = st.text_input("Cari berdasarkan jenis hewan:")

    # Filter tanggal
    start_date = st.date_input("Pilih tanggal mulai:", value=pd.to_datetime("2024-01-01"))
    end_date = st.date_input("Pilih tanggal akhir:", value=pd.to_datetime("2024-12-31"))

    # Filter waktu
    start_time = st.time_input("Pilih waktu mulai:", value=datetime.strptime("00:00", "%H:%M").time())
    end_time = st.time_input("Pilih waktu akhir:", value=datetime.strptime("23:59", "%H:%M").time())

# Menerapkan filter ke DataFrame
df_filtered = df_appointments.copy()

if filter_pet:
    df_filtered = df_filtered[df_filtered['Jenis Hewan'].str.contains(filter_pet, case=False)]

df_filtered = df_filtered[
    (df_filtered['waktu'].dt.date >= start_date) &
    (df_filtered['waktu'].dt.date <= end_date) &
    (df_filtered['waktu'].dt.time >= start_time) &
    (df_filtered['waktu'].dt.time <= end_time)
]


# Tampilkan data hasil filter dalam tabel tanpa kolom indeks
if not df_filtered.empty:
    st.table(df_filtered.reset_index(drop=True))
else:
    st.warning("Tidak ada janji temu yang sesuai dengan filter.")
