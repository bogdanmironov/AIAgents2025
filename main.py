import os
import sqlite3
import tempfile
import torch
import torch.nn as nn
import whisper
import joblib
from flask import Flask, request

class MultinomialLogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MultinomialLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        logits = self.linear(x)
        return logits

def load_classifier_components():
    vectorizer = joblib.load("vectorizer.joblib")
    label_mapping = joblib.load("label_mapping.joblib")
    input_dim = vectorizer.max_features
    num_classes = len(label_mapping)
    model = MultinomialLogisticRegression(input_dim, num_classes)
    model.load_state_dict(torch.load("classifier_model.pt", map_location=torch.device('cpu')))
    model.eval()
    return vectorizer, label_mapping, model

vectorizer, label_mapping, classifier_model = load_classifier_components()
audio_model = whisper.load_model("base")

def get_db_connection():
    conn = sqlite3.connect('videos.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            data BLOB
        )
    ''')
    conn.commit()
    conn.close()

def save_file_to_db(file_storage):
    conn = get_db_connection()
    blob_data = file_storage.read()
    conn.execute("INSERT INTO videos (filename, data) VALUES (?, ?)", (file_storage.filename, blob_data))
    conn.commit()
    video_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return video_id

def retrieve_file_from_db(video_id):
    conn = get_db_connection()
    row = conn.execute("SELECT filename, data FROM videos WHERE id = ?", (video_id,)).fetchone()
    conn.close()
    if row:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(row['data'])
        temp_file.close()
        return temp_file.name
    else:
        return None

def transcribe_and_classify(audio_path, audio_model, vectorizer, classifier_model, language="en"):
    result = audio_model.transcribe(audio_path, language=language)
    transcribed_text = result["text"]
    print("Transcribed text:", transcribed_text)
    X_new = vectorizer.transform([transcribed_text]).toarray()
    X_new_tensor = torch.tensor(X_new, dtype=torch.float32)
    with torch.no_grad():
        logits = classifier_model(X_new_tensor)
        predicted_class = torch.argmax(logits, dim=1).item()
    predicted_label = label_mapping[predicted_class]
    print("Predicted label:", predicted_label)
    return transcribed_text, predicted_label

app = Flask(__name__)

upload_form = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Upload MP4 File</title>
  </head>
  <body>
    <h1>Upload MP4 File for Transcription & Classification</h1>
    <form method="post" enctype="multipart/form-data" action="/upload">
      <input type="file" name="video" accept="video/mp4,video/*,audio/mp4,audio/x-m4a">
      <input type="submit" value="Upload">
    </form>
  </body>
</html>
"""

@app.route('/')
def index():
    return upload_form

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return "No file part", 400
    file = request.files['video']
    if file.filename == '':
        return "No selected file", 400
    video_id = save_file_to_db(file)
    temp_filepath = retrieve_file_from_db(video_id)
    if not temp_filepath:
        return "Error retrieving the file from the database.", 500
    transcribed_text, predicted_label = transcribe_and_classify(temp_filepath, audio_model, vectorizer, classifier_model, language="en")
    os.remove(temp_filepath)
    response_html = f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <title>Result</title>
      </head>
      <body>
        <h1>Transcription and Classification Result</h1>
        <p><strong>Transcribed Text:</strong> {transcribed_text}</p>
        <p><strong>Predicted Label:</strong> {predicted_label}</p>
        <a href="/">Upload another file</a>
      </body>
    </html>
    """
    return response_html

if __name__ == '__main__':
    init_db()
    app.run()