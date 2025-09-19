from flask import Flask, render_template, request, jsonify
import mysql.connector
import pandas as pd
from datetime import datetime

app = Flask(__name__)


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
    query = "SELECT id, nama_pemilik, jenis_hewan, waktu, keluhan, status FROM janji_temu"
    df = pd.read_sql(query, connection)
    connection.close()
    return df


# Route untuk halaman utama
@app.route('/', methods=['GET', 'POST'])
def dashboard():
    # Memuat data janji temu
    df = load_appointments()
    df = df.rename(columns={
        'nama_pemilik': 'Nama Pemilik',
        'jenis_hewan': 'Jenis Hewan',
        'waktu': 'Waktu',
        'keluhan': 'Keluhan',
        'status': 'Status'
    })

    # Menangani filter data jika form disubmit
    if request.method == 'POST':
        # Ambil data filter dari form
        filter_pet = request.form.get('filter_pet', '').strip()
        start_date = request.form.get('start_date', '')
        end_date = request.form.get('end_date', '')
        start_time = request.form.get('start_time', '')
        end_time = request.form.get('end_time', '')

        # Filter data hanya jika ada input filter
        if filter_pet or start_date or end_date or start_time or end_time:
            # Konversi kolom Waktu ke tipe datetime
            df['Waktu'] = pd.to_datetime(df['Waktu'])

            # Filter berdasarkan jenis hewan
            if filter_pet:
                df = df[df['Jenis Hewan'].str.contains(filter_pet, case=False)]

            # Filter berdasarkan tanggal
            if start_date:
                df = df[df['Waktu'].dt.date >= pd.to_datetime(start_date).date()]
            if end_date:
                df = df[df['Waktu'].dt.date <= pd.to_datetime(end_date).date()]

            # Filter berdasarkan waktu
            if start_time:
                df = df[df['Waktu'].dt.time >= datetime.strptime(start_time, "%H:%M").time()]
            if end_time:
                df = df[df['Waktu'].dt.time <= datetime.strptime(end_time, "%H:%M").time()]

    # Mengirim data ke template
    return render_template('dashboard.html', appointments=df)


# Endpoint untuk memperbarui status janji temu
@app.route('/update_status', methods=['POST'])
def update_status():
    data = request.get_json()
    appointment_id = data.get('id')
    new_status = data.get('status')

    if not appointment_id or not new_status:
        return jsonify({'error': 'Invalid data'}), 400

    try:
        connection = get_database_connection()
        cursor = connection.cursor()
        cursor.execute(
            "UPDATE janji_temu SET status = %s WHERE id = %s",
            (new_status, appointment_id)
        )
        connection.commit()
        cursor.close()
        connection.close()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
