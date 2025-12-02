from flask import Flask, render_template, request       # mengimpor Flask dan fungsi untuk render HTML & info request
from flask_socketio import SocketIO, emit               # mengimpor library WebSocket untuk komunikasi real-time
import joblib                                           # untuk memuat model dan vectorizer dari file .pkl
import time                                             # untuk membuat timestamp (waktu ketika pesan dikirim)

app = Flask(__name__)                                   # inisialisasi aplikasi Flask
socketio = SocketIO(app, cors_allowed_origins="*")      # menghubungkan Flask dengan SocketIO (mengaktifkan WebSocket)

# ===== LOAD MODEL =====
model = joblib.load("model.pkl")                        # memuat model machine learning
vectorizer = joblib.load("vectorizer.pkl")              # memuat TF-IDF vectorizer

connected_users = {}                                    # menyimpan daftar user yang online (sid → username)
user_count = 0                                          # nomor user agar menghasilkan nama User 1, User 2, dst

def is_spam(text):                                      # fungsi untuk mendeteksi apakah pesan spam
    vec = vectorizer.transform([text])                  # mengubah teks menjadi vektor numerik
    pred = model.predict(vec)[0]                        # memprediksi kategori pesan (spam / ham)
    return pred == "spam"                               # return True jika kategori spam

# ===== ROUTE =====
@app.route("/")                                         # route halaman utama
def index():
    return render_template("index.html")                # membuka file HTML (UI chat)

# ===== KETIKA USER TERHUBUNG =====
@socketio.on("connect")                                 # event saat user terhubung ke WebSocket
def on_connect():
    global user_count                                   # ambil variabel global
    user_count += 1                                     # user baru → tambah jumlah user
    user_id = f"User {user_count}"                      # buat nama user baru
    connected_users[request.sid] = user_id              # simpan username berdasarkan session ID
    emit("set_username", user_id)                       # kirim username ke user yang baru masuk
    emit("user_list", list(connected_users.values()), broadcast=True)  # perbarui daftar user ke semua client

@socketio.on("disconnect")                              # event saat user keluar
def on_disconnect():
    if request.sid in connected_users:                  # jika user yang keluar ada dalam daftar
        del connected_users[request.sid]                # hapus user dari daftar
        emit("user_list", list(connected_users.values()), broadcast=True)  # update daftar user ke semua client

# ===== PESAN MASUK =====
@socketio.on("message")                                 # event saat pesan dikirimkan oleh user
def handle_message(msg):
    user = connected_users.get(request.sid, "User")     # ambil username berdasarkan session ID
    timestamp = time.strftime("%H:%M")                  # waktu saat pesan dikirim
    spam_flag = is_spam(msg)                            # cek apakah pesan itu spam

    data = {                                            # buat data pesan untuk dikirim ke client
        "user": user,
        "text": msg,
        "time": timestamp,
        "isSpam": spam_flag
    }

    emit("chat_message", data, broadcast=True)          # kirim pesan ke semua user secara realtime (broadcast)

if __name__ == "__main__":                               # menjalankan server jika file dieksekusi langsung
    print("Server running at http://127.0.0.1:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)   # menjalankan server WebSocket + Flask dengan mode debug
