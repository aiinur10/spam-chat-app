from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import joblib
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ===== LOAD MODEL =====
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

connected_users = {}
user_count = 0

def is_spam(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return pred == "spam"

# ===== ROUTE =====
@app.route("/")
def index():
    return render_template("index.html")

# ===== KETIKA USER TERHUBUNG =====
@socketio.on("connect")
def on_connect():
    global user_count
    user_count += 1
    user_id = f"User {user_count}"
    connected_users[request.sid] = user_id
    emit("set_username", user_id)
    emit("user_list", list(connected_users.values()), broadcast=True)

@socketio.on("disconnect")
def on_disconnect():
    if request.sid in connected_users:
        del connected_users[request.sid]
        emit("user_list", list(connected_users.values()), broadcast=True)

# ===== PESAN MASUK =====
@socketio.on("message")
def handle_message(msg):
    user = connected_users.get(request.sid, "User")
    timestamp = time.strftime("%H:%M")
    spam_flag = is_spam(msg)

    data = {
        "user": user,
        "text": msg,
        "time": timestamp,
        "isSpam": spam_flag
    }

    # Kirim langsung ke semua user
    emit("chat_message", data, broadcast=True)


if __name__ == "__main__":
    print("Server running at http://127.0.0.1:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
