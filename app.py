import os
import re
import fitz  # PyMuPDF
import spacy
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, send_file, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER
app.secret_key = 'your_secret_key_here'  # <-- Add a secret key for sessions!

nlp = spacy.load("en_core_web_sm")

# -------- Hardcoded User --------
users = {
    "admin": generate_password_hash("admin123")
}

# -------- Login Decorator --------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ---------- Helper Functions ----------
def extract_text_from_pdf(path):
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return "Name Not Found"

def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else "Email Not Found"

# ---------- Routes ----------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and check_password_hash(users[username], password):
            session['username'] = username
            return redirect(url_for('index'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        if not os.path.exists(REPORT_FOLDER):
            os.makedirs(REPORT_FOLDER)

        # Job description PDF
        if 'job_description_file' not in request.files:
            return "Job description file missing", 400
        job_desc_file = request.files['job_description_file']
        if job_desc_file.filename == '':
            return "No job description file selected", 400

        job_desc_path = os.path.join(UPLOAD_FOLDER, job_desc_file.filename)
        job_desc_file.save(job_desc_path)
        job_desc_text = extract_text_from_pdf(job_desc_path)

        # Resumes
        files = request.files.getlist('resumes')
        resume_texts = []
        filenames = []

        for file in files:
            if file.filename.endswith('.pdf'):
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)
                text = extract_text_from_pdf(filepath)
                resume_texts.append(text)
                filenames.append(file.filename)

        # Scoring
        texts = [preprocess(job_desc_text)] + [preprocess(text) for text in resume_texts]
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(texts)
        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # Matched keywords
        feature_names = tfidf.get_feature_names_out()
        job_vec = tfidf_matrix[0].toarray().flatten()

        results = []
        for i in range(len(resume_texts)):
            resume_vec = tfidf_matrix[i+1].toarray().flatten()
            matched_indices = np.where((job_vec > 0) & (resume_vec > 0))[0]
            matched_keywords = ", ".join([feature_names[idx] for idx in matched_indices])

            name = extract_name(resume_texts[i])
            email = extract_email(resume_texts[i])
            score_percent = round(scores[i]*100, 2)

            results.append((filenames[i], name, email, f"{score_percent}%", matched_keywords))

        results.sort(key=lambda x: float(x[3].rstrip('%')), reverse=True)

        # Save to CSV
        df = pd.DataFrame(results, columns=['File Name', 'Candidate Name', 'Email', 'Score', 'Matched Keywords'])
        report_path = os.path.join(REPORT_FOLDER, 'hr_report.csv')
        df.to_csv(report_path, index=False)

        return render_template('index.html', results=results, report_generated=True)

    return render_template('index.html', results=None, report_generated=False)

@app.route('/download')
@login_required
def download():
    path = os.path.join(REPORT_FOLDER, 'hr_report.csv')
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
