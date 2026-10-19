# 23CSE301 ML Capstone — Interactive Streamlit Dashboard

A state-of-the-art interactive Machine Learning dashboard built with Streamlit and Plotly, covering all three tracks:
1. **🚗 Track 1: Interstate Traffic Volume Regression (10 Models)**
2. **🏦 Track 2: Bank Term Deposit Classification (10 Models)**
3. **🛍️ Track 3: Online Retail Customer Segmentation (K-Means & Hierarchical)**
4. **📋 Technical Viva & Evaluation Rubric Guide**

---

## 🚀 Running Locally

### 1. Prerequisites
Ensure dependencies from `requirements.txt` are installed:
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
Run the Streamlit server from the project root:
```bash
streamlit run app/app.py
```
Or from within the `app/` folder:
```bash
cd app
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 🌐 One-Click Public Deployment (Bonus: +1 to +2 Marks)

This application is fully production-ready and pre-configured for instant zero-configuration deployment to **Streamlit Community Cloud** or **Render**.

### Option A: Streamlit Community Cloud (Recommended)
1. Fork or push this repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
3. Click **"New App"**.
4. Select your repository: `SH-Nihil-Mukkesh-25/ml_capstone` (or your personal fork).
5. Set:
   - **Main file path:** `app/app.py`
   - **Branch:** `main`
6. Click **"Deploy!"**.
   - Streamlit Cloud will automatically install `requirements.txt`, load the serialized models in `models/`, and provide a public HTTPS URL (e.g. `https://ml-capstone-amrita.streamlit.app`).

### Option B: Render / Hugging Face Spaces
- **Render:** Select **Web Service**, Build Command: `pip install -r requirements.txt`, Start Command: `streamlit run app/app.py --server.port $PORT --server.address 0.0.0.0`.
- **Hugging Face Spaces:** Create a new Space with the **Streamlit** SDK, upload repository files, and set `app_file: app/app.py`.

---

## 📦 Features & Interactive Capabilities
- **Real-Time Traffic Simulator:** Live prediction of vehicle volume and congestion classification using the trained Gradient Boosting Regressor pipeline.
- **Client Lead Scoring Engine:** Real-time customer subscription probability, dynamic decision threshold slider (0.05–0.95), and lead tier qualification (Platinum, Gold, Silver, Bronze).
- **Customer Persona Classifier:** Live assignment of customer RFM parameters into 4 strategic retail personas with tailored business retention action plans.
- **3D PCA Space Explorer:** Interactive 3D visualization of customer segments in orthogonal principal component space.
- **Viva Voce Cheat Sheet:** Technical answers and theoretical justifications for all major rubric viva questions.
