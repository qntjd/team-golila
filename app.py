from flask import Flask, render_template, request, jsonify, send_from_directory
import os, time, json
from model.generator import generate_music

app = Flask(__name__)

# outputs 폴더 자동 생성
out_dir = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(out_dir, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_music', methods=['POST'])
def generate_music_api():
    data = request.get_json()
    genre = data.get('genre', 'pop')
    mood = data.get('mood', 'happy')
    duration = data.get('duration', 10)
    
    file_path = generate_music(genre, mood, duration)
    return jsonify({"message": "Music generated successfully!", "file": os.path.basename(file_path)})

@app.route('/latest')
def latest():
    files = [f for f in os.listdir(out_dir) if f.endswith('.mid') or f.endswith('.wav')]
    if not files:
        return jsonify([])
    files.sort(key=lambda f: os.path.getmtime(os.path.join(out_dir, f)), reverse=True)
    return jsonify(files)

@app.route('/outputs/<filename>')
def download_file(filename):
    return send_from_directory(out_dir, filename)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000, debug=True)
