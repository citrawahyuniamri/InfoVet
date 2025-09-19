# Dataframe Module
import pandas as pd

# Stopword Removal Module
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords

# Tokenizer Module
from nltk.tokenize import word_tokenize

# Stemmer Module
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Padding Module
from keras_preprocessing.sequence import pad_sequences

# Import/Export Object Module
import pickle

# load_model Module
from keras.models import load_model

# fuzzywuzzy Module (Untuk Function Fuzzy_Recommend)
from fuzzywuzzy import process, fuzz

# pertanyaan_oov Module
from csv import writer

# Connect to Telegram Module
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)
import logging

# Inline keyboard
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackQueryHandler
# Load json file Module
import json

#database mysql
import mysql.connector
from mysql.connector import Error

#datetime
import re
from datetime import datetime

# Module untuk menghitung QA execution time
import time


class OOV():

    def Preprocessing(self, data):
        # Case Folding
        data['lower'] = data['Pertanyaan'].str.lower()

        # Punctual Removal
        data['punctual'] = data['lower'].str.replace('[^a-zA-Z0-9]+', ' ', regex=True)

        # Normalization
        kamus_baku = pd.read_csv('kata_baku.csv', sep=";")
        dict_kamus_baku = kamus_baku[['slang', 'baku']].to_dict('list')
        dict_kamus_baku = dict(zip(dict_kamus_baku['slang'], dict_kamus_baku['baku']))
        norm = []
        for i in data['punctual']:
            res = " ".join(dict_kamus_baku.get(x, x) for x in str(i).split())
            norm.append(str(res))
        data['normalize'] = norm

        # Stopword Removal
        stop_words = set(stopwords.words('indonesian'))
        swr = []
        for i in data['normalize']:
            tokens = word_tokenize(i)
            filtered = [word for word in tokens if word not in stop_words]
            swr.append(" ".join(filtered))
        data['stopwords'] = swr

        # Stemming
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        stem = []
        for i in data['stopwords']:
            stem.append(stemmer.stem(str(i)))
        data['stemmed'] = stem

        return data

    def Model_Sequencing(self, data):
        model_tokenizer = pickle.load(open('model_tokenizer.pkl', 'rb'))
        model_sequences = model_tokenizer.texts_to_sequences(data['stemmed'])
        max_seq_len = pickle.load(open('max_seq_len.pkl', 'rb'))
        model_sequences_padded = pad_sequences(model_sequences, maxlen=max_seq_len)
        return model_sequences_padded

    def Predict_Label(self, model_sequences_padded):
        model = load_model('chatbot_model.h5')
        categorical_predicted_label = []
        onehot_predicted_label = model.predict(model_sequences_padded)
        for i in range(0, len(model_sequences_padded)):
            categorical_predicted_label.append(onehot_predicted_label[i].argmax())
        return categorical_predicted_label

    def Menambahkan_Pertanyaan_OOV(self):
        try :
            data = pd.read_csv('pertanyaan_oov.csv', header=None)
            # print(data.shape)
            if (data.shape[1]==1):
                print("Terdapat Pertanyaan OOV yang belum dijawab")
        except pd.errors.EmptyDataError:
            print("Tidak terdapat Pertanyaan OOV yang perlu ditambahkan")

class QA():

    def Preprocessing(self,input):
        print("Tahap Preprocessing Dimulai")

        # Case Folding
        data = pd.DataFrame([input], columns=['Pertanyaan'])
        data['lower'] = data['Pertanyaan'].str.lower()
        print("Tahap Case Folding Berhasil :",data['lower'].iloc[0])

        # Punctual Removal
        data['punctual'] = data['lower'].str.replace('[^a-zA-Z0-9]+', ' ', regex=True)
        print("Tahap Punctual Removal Berhasil :",data['punctual'].iloc[0])

        # Normalization
        kamus_baku = pd.read_csv('kata_baku.csv', sep=";")
        dict_kamus_baku = kamus_baku[['slang', 'baku']].to_dict('list')
        dict_kamus_baku = dict(zip(dict_kamus_baku['slang'], dict_kamus_baku['baku']))
        norm = []
        for i in data['punctual']:
            res = " ".join(dict_kamus_baku.get(x, x) for x in str(i).split())
            norm.append(str(res))
        data['normalize'] = norm
        print("Tahap Normalisasi Berhasil :",data['normalize'].iloc[0])

        # Stopword Removal
        stop_words = set(stopwords.words('indonesian'))
        swr = []
        for i in data['normalize']:
            tokens = word_tokenize(i)
            filtered = [word for word in tokens if word not in stop_words]
            swr.append(" ".join(filtered))
        data['stopwords'] = swr
        print("Tahap Stopword Removal Berhasil :",data['stopwords'].iloc[0])

        # Stemming
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        stem = []
        for i in data['stopwords']:
            stem.append(stemmer.stem(str(i)))
        data['stemmed'] = stem
        print("Tahap Stemming Berhasil :",data['stemmed'].iloc[0])
        return data

    def Find_oov(self,list_to_check, item_to_find):
        oov_index = []
        for idx, value in enumerate(list_to_check):
            if value == item_to_find:
                oov_index.append(idx)
        return oov_index

    def OOV_Checking(self,data):
        corpus_tokenizer = pickle.load(open('corpus_tokenizer.pkl', 'rb'))
        corpus_sequence = corpus_tokenizer.texts_to_sequences(data['stemmed'])
        oov_index = self.Find_oov(corpus_sequence[0],1)
        oov_words = []
        input_splitted = data['stemmed'].iloc[0].split()
        for i in oov_index:
            oov_words.append(input_splitted[i])
        return oov_words

    def Fuzzy_Recommend(self,oov_words,input):
        print("Tahap Fuzzy Recommend Dimulai")
        corpus_word_index = pickle.load(open('corpus_word_index.pkl', 'rb'))
        list_corpus_word_index = list(corpus_word_index.keys())
        recommended_input = []
        for oov_word in oov_words:
            recommended_input.append(process.extract(oov_word, list_corpus_word_index, scorer=fuzz.token_sort_ratio)[0][0])
            print(process.extract(oov_word, list_corpus_word_index, scorer=fuzz.token_sort_ratio))
        with open('pertanyaan_oov.csv', 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow([input])
            f_object.close()
        return recommended_input

    def Add_To_Pertanyaan_OOV(self,input):
        print("Pertanyaan Yang Diberikan Tidak Terdapat Dalam Corpus")
        with open('pertanyaan_oov.csv', 'a') as f_object:
            print("Proses Memasukkan Pertanyaan Kedalam Pertanyaan_OOV.csv Dimulai")
            writer_object = writer(f_object)
            writer_object.writerow([input])
            f_object.close()
            print("Proses Memasukkan Pertanyaan Kedalam Pertanyaan_OOV.csv Selesai")
        return 0

    def Model_Sequencing(self,data):
        model_tokenizer = pickle.load(open('model_tokenizer.pkl', 'rb'))
        model_sequence = model_tokenizer.texts_to_sequences(data['stemmed'])
        max_seq_len = pickle.load(open('max_seq_len.pkl', 'rb'))
        model_sequence_padded = pad_sequences(model_sequence, maxlen=max_seq_len) #maxlen mengikuti maxlen model training
        print("Tahap Model Sequencing Berhasil :", model_sequence_padded)
        return model_sequence_padded

    def Predict_Label(self,sequence):
        model = load_model('chatbot_model.h5')
        label = int(model.predict(sequence).argmax())
        if (label == 0):
            type_label = 'Kesehatan'
        elif (label == 1):
            type_label = 'Klinik'
        elif (label == 2):
            type_label = 'Perilaku'
        print("Tahap Predict Label Berhasil :",type_label)
        return label

    def Corpus_Sequencing(self,data):
        corpus_tokenizer = pickle.load(open('corpus_tokenizer.pkl', 'rb'))
        corpus_sequence = corpus_tokenizer.texts_to_sequences(data['stemmed'])
        print("Tahap Corpus Sequencing Berhasil :",corpus_sequence)
        return corpus_sequence

    def Matching(self,sequence,predicted_label):
        print("Tahap Matching Dimulai")
        print("Sequence Pertanyaan: ",sequence)
        df_Kesehatan = pickle.load(open('df_Kesehatan.pkl', 'rb'))
        df_Klinik = pickle.load(open('df_Klinik.pkl', 'rb'))
        df_Perilaku = pickle.load(open('df_Perilaku.pkl', 'rb'))
        if (predicted_label == 0):
            check_df = df_Kesehatan
        elif (predicted_label == 1):
            check_df = df_Klinik
        elif (predicted_label == 2):
            check_df = df_Perilaku

        Compatibility = [0] * len(check_df)
        # print("sudah sampai sini 2")
        # Looping tiap baris Sequences df corpus pilihan
        index = 0
        for check_sequences in check_df['Sequences']:
            # Looping tiap element Sequences df testing per baris
            for element in sequence[0]:
                # print(element)
                if (element in check_sequences):
                    # print("didalam if")
                    Compatibility[index] += 1
            #                     print(Compatibility)
            # Compatibility[index] = Compatibility[index] / len(check_sequences)
            Compatibility[index] = Compatibility[index] / len(sequence[0])
            index += 1
        print("Tahap Pengecekan Compatibility Berhasil")
        print("Hasil Pengecekan Compatibilty :",Compatibility)

        # Jika presentase compatibility sama maka diambil jawaban yang len corpus_sequence-nya terpanjang
        # Dimana hal ini menandakan kecocokan yang lebih menyeluruh
        # Penggunaan teknik ini untuk mencegah input panjang tetapi ada kata yang terdapat pada pertanyaan corpus yang pendek
        
        print("Tahap Max Compatibility Dimulai")
        index_max_compatibility = []
        for idx, value in enumerate(Compatibility):
            if value == max(Compatibility):
                index_max_compatibility.append(idx)

        print("Index yang memiliki Max Compatibility :",index_max_compatibility)

        perfect_compatibilty_sequence = []
        perfect_compatibilty_index = 0
        for idx in index_max_compatibility:
            if (idx == index_max_compatibility[0]):
                perfect_compatibilty_sequence = check_df['Sequences'].iloc[idx]
                perfect_compatibilty_index = idx
            else:
                if(len(check_df['Sequences'].iloc[idx]) <= len(perfect_compatibilty_sequence)):
                    perfect_compatibilty_sequence = check_df['Sequences'].iloc[idx]
                    perfect_compatibilty_index = idx

        # Prediksi_Jawaban = check_df['Jawaban'].loc[Compatibility.index(max(Compatibility))]
        Prediksi_Jawaban = check_df['Jawaban'].iloc[perfect_compatibilty_index]
        print("Tahap Pengambilan Compatibility Maksimum Berhasil")
        print("Index dengan Compatibility Maksimum yang akan diambil adalah index ke :",perfect_compatibilty_index)
        return Prediksi_Jawaban

    def Handle_Response(self,input):
        data = self.Preprocessing(input)
        oov_words = self.OOV_Checking(data)
        if(len(oov_words)==0):
            print("Pertanyaan yang diberikan adalah :",input)
            corpus_sequence = self.Corpus_Sequencing(data)
            model_sequence_padded = self.Model_Sequencing(data)
            predicted_label = self.Predict_Label(model_sequence_padded)
            prediksi_jawaban = self.Matching(corpus_sequence,predicted_label)
            print("Tahap Pencarian Jawaban Berhasil")
            print("Jawaban atas pertanyaan '"+input+"' adalah '"+prediksi_jawaban+"'")
            return prediksi_jawaban
        else:
            Jawaban = "Maaf, saat ini chatbot masih tidak memiliki pengetahuan mengenai '" + ' '.join(oov_words) + "'\n" + "Jawaban atas pertanyaan tersebut akan segera ditambahkan."
            self.Add_To_Pertanyaan_OOV(input)
            # recommended_input = self.Fuzzy_Recommend(oov_words, input)
            # Jawaban = Jawaban + "Apakah yang anda maksud adalah '" + ' '.join(recommended_input) + "'?"
            # Jawaban = "Maaf, saya tidak memiliki pengetahuan mengenai "+ x for x in (data['stemmed'].iloc[0].split())
            return Jawaban

TOKEN = '7569695626:AAFGk0JLygaVtb6WqNgXjLpbrB11TGXvT6Y'
BOT_USERNAME = '@klinikhewansumut_bot'

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Selamat datang di InfoVet!\n'
                                    'Saya adalah sebuah Chatbot Layanan Informasi UPTD Klinik Hewan Sumatera Utara.\n'
                                    'Saya dapat menjawab pertanyaan seputar layanan klinik dan kesehatan hewan peliharaan anda.\n'
                                    'Saya juga dapat membantu anda membuat janji temu di klinik.\n'
                                    'Gunakan perintah /list_perintah untuk mengakses semua perintah yang dapat digunakan')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Saya akan menjawab pertanyaan seputar layanan klinik\n'
                                    'serta informasi kesehatan hewan kucing dan anjing.\n'
                                    'Silahkan ajukan pertanyaan anda, saya akan menjawab')

async def tanya_chatbot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Cek apakah pengguna sedang membuat janji
    if context.user_data.get('is_creating_appointment', False):
        await update.message.reply_text(
            "Anda sedang dalam proses pembuatan janji. Selesaikan dulu proses ini sebelum menanyakan pertanyaan lain."
        )
        return

    text_diterima: str = update.message.text

    print('Text diterima : ', text_diterima)
    if text_diterima == '/tanya':
        await update.message.reply_text("Silahkan langsung menanyakan pertanyaan.\n\n"
                                        "Contoh : Apa saja jenis vaksin untuk kucing?")
    else:
        qa_start = time.monotonic()
        response = qa.Handle_Response(text_diterima)
        qa_end = time.monotonic()
        exec_time = round(qa_end - qa_start, 2)
        print("Execution Time :", exec_time, "Detik")
        await update.message.reply_text(response)

# Define state constants
NAMA, JENIS_HEWAN, WAKTU, KELUHAN = range(4)

# Start making an appointment
async def buat_janji(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['is_creating_appointment'] = True  # Set status pembuatan janji
    keyboard = [
        [InlineKeyboardButton("Isi Form", callback_data='form_start')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        'Terima kasih telah menghubungi UPT Klinik Hewan Sumatera Utara. '
        'Untuk membuat janji temu, klik tombol di bawah untuk memulai.'
        'Untuk membatalkan klik atau masukkan perintah /cancel.',
        reply_markup=reply_markup
    )
    return NAMA

# Ask for owner's name
async def form_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Silakan masukkan nama pemilik:")
    return NAMA

# Handle phone input and ask for pet type
async def input_nama(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['name'] = update.message.text
    await update.message.reply_text(f"Nama: {context.user_data['name']}\n"
                                    "Sekarang silakan masukkan jenis hewan peliharaan:")
    return JENIS_HEWAN

# Handle pet type and complete the form
async def input_jenis_hewan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['pet'] = update.message.text
    await update.message.reply_text(f"Jenis hewan: {context.user_data['pet']}\n"
                                    "Sekarang silakan masukkan tanggal dan waktu janji temu\n"
                                    "(Contoh : 27-10-2024, 10.00")
    return WAKTU


async def input_waktu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    time_input = update.message.text

    # Pola regex untuk memvalidasi format "DD-MM-YYYY, HH.MM"
    time_pattern = r"^([0-9]{1,2})-([0-9]{1,2})-([0-9]{4}),\s([0-9]{1,2})\.([0-9]{2})$"
    match = re.match(time_pattern, time_input)

    if match:
        # Konversi teks input ke datetime
        try:
            # Gabungkan input menjadi format yang bisa dibaca oleh datetime
            date_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)} {match.group(4)}:{match.group(5)}"
            appointment_time = datetime.strptime(date_str, "%d-%m-%Y %H:%M")

            # Simpan datetime ke context
            context.user_data['time'] = appointment_time

            await update.message.reply_text(f"Waktu janji temu: {appointment_time.strftime('%d-%m-%Y, %H.%M')}\n"
                                            "Silakan masukkan keluhan yang dialami hewan Anda.")
            return KELUHAN

        except ValueError as e:
            # Jika konversi ke datetime gagal
            await update.message.reply_text("Format waktu tidak valid. Silakan masukkan ulang sesuai format: "
                                            "DD-MM-YYYY, HH.MM\n(Contoh: 27-10-2024, 10.00)")
            return WAKTU
    else:
        # Jika format tidak sesuai dengan regex
        await update.message.reply_text("Format tidak sesuai. Silakan masukkan ulang dengan format yang benar: "
                                        "DD-MM-YYYY, HH.MM\n(Contoh: 27-10-2024, 10.00)")
        return WAKTU

async def input_keluhan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['problem'] = update.message.text

    # Data yang akan disimpan
    name = context.user_data['name']
    pet = context.user_data['pet']
    time = context.user_data['time']
    problem = context.user_data['problem']

    # Tampilkan pesan janji temu selesai
    await update.message.reply_text(f"Nama pemilik        : {name}\n"
                                    f"Jenis hewan           : {pet}\n"
                                    f"Waktu janji temu  : {time}\n"
                                    f"Keluhan             : {problem}\n\n"
                                    "Terima kasih, janji temu Anda telah dibuat.")

    # Reset status setelah janji selesai
    context.user_data['is_creating_appointment'] = False

    # Simpan data ke database MySQL
    try:
        # Buat koneksi ke MySQL
        connection = mysql.connector.connect(
            host='localhost',
            database='klinik_hewan',
            user='root',
            password=''
        )

        if connection.is_connected():
            cursor = connection.cursor()
            # Query untuk memasukkan data ke dalam tabel
            insert_query = """INSERT INTO janji_temu (nama_pemilik, jenis_hewan, waktu, keluhan, status)
                              VALUES (%s, %s, %s, %s,%s)"""
            # Eksekusi query
            cursor.execute(insert_query, (name, pet, time, problem, "Dijadwalkan"))
            connection.commit()  # Simpan perubahan

            print("Data janji temu berhasil disimpan ke database.")

    except Error as e:
        print(f"Error saat menyimpan ke database: {e}")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("Koneksi ke MySQL ditutup.")

    # Akhiri percakapan
    return ConversationHandler.END

# If user cancels, we can stop the conversation
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Pembuatan janji dibatalkan.")
    return ConversationHandler.END


async def text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_diterima: str = update.message.text

    print('Text diterima : ', text_diterima)
    qa_start = time.monotonic()
    response = qa.Handle_Response(text_diterima)
    qa_end = time.monotonic()
    exec_time = round(qa_end - qa_start, 2)
    print("Execution Time :", exec_time, "Detik")

    await update.message.reply_text(response)

async def list_perintah(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Perintah-perintah yang dapat digunakan yakni :\n'
                                    '/start - Perintah untuk memulai chatbot\n'
                                    '/help - Perintah untuk menampilkan deskripsi chatbot\n'
                                    '/tanya - Perintah untuk menanyakan langsung pertanyaan\n'
                                    '/buat_janji - Perintah untuk membuat janji temu di klinik')


def main():
    global qa
    global oov

    # Initialize QA and OOV objects
    qa = QA()
    oov = OOV()
    oov.Menambahkan_Pertanyaan_OOV()
    print('Bot dimulai...')

    # Create the application instance
    application = Application.builder().token(TOKEN).build()

    # Add command handlers
    application.add_handler(CommandHandler('start', start_command))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('tanya', tanya_chatbot))
    application.add_handler(CommandHandler('list_perintah', list_perintah))

    # Define the ConversationHandler for appointment booking
    appointment_handler = ConversationHandler(
        entry_points=[CommandHandler('buat_janji', buat_janji)],
        states={
            NAMA: [MessageHandler(filters.TEXT & ~filters.COMMAND, input_nama)],
            JENIS_HEWAN: [MessageHandler(filters.TEXT & ~filters.COMMAND, input_jenis_hewan)],
            WAKTU: [MessageHandler(filters.TEXT & ~filters.COMMAND, input_waktu)],
            KELUHAN: [MessageHandler(filters.TEXT & ~filters.COMMAND, input_keluhan)],
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    # Add the ConversationHandler
    application.add_handler(appointment_handler)

    # Add handler for callback queries
    application.add_handler(CallbackQueryHandler(form_start, pattern='form_start'))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, tanya_chatbot))

    # Start polling to listen for updates
    application.run_polling()

# Run the main function
if __name__ == '__main__':
    main()