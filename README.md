# 🛍️ Customer Segmentation using Clustering
**Data Mining — AIE323 | Alamein University | Project 1**

---

## 📁 Project Structure

```
customer-segmentation/
├── customer_segmentation.ipynb   ← Full pipeline notebook (Milestones 1–3)
├── app.py                        ← Streamlit dashboard (Milestone 4)
├── requirements.txt              ← Python dependencies
├── Mall_Customers.csv            ← Dataset (optional — auto-generated if missing)
└── README.md
```

---

## ⚙️ Setup (One-Time)

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Step 1 — Run the Jupyter Notebook
```bash
jupyter notebook customer_segmentation.ipynb
```
Run all cells top to bottom. This will:
- Load / generate the dataset
- Perform EDA and preprocessing
- Engineer features (RFM proxy scores)
- Apply PCA dimensionality reduction
- Train K-Means, DBSCAN, and Agglomerative Clustering
- Evaluate all models (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- Profile customer segments and assign personas
- Save `kmeans_model.pkl` and `scaler.pkl`

### Step 2 — Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at **http://localhost:8501**

---

## 📊 Dashboard Features

| Tab | Description |
|-----|-------------|
| 📍 Cluster View | 2D scatter + PCA projection + cluster size bars |
| 📊 Profiles | Segment personas, bar comparisons, radar chart |
| 🔍 Customer Lookup | Enter Customer ID → get segment + persona |
| 📤 Upload & Predict | Upload new CSV → download predictions |
| 📈 EDA | Distributions, correlation heatmap, gender split |

**Sidebar Controls:**
- Choose algorithm: K-Means / Agglomerative / DBSCAN
- Adjust number of clusters (k)
- Tune DBSCAN epsilon & min_samples
- Select feature axes for scatter plot

---

## 📂 Dataset

**Mall Customer Segmentation** (Kaggle)
- Download from: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
- Place `Mall_Customers.csv` in the project folder
- If not present, the code **auto-generates synthetic data** with similar characteristics

| Column | Description |
|--------|-------------|
| CustomerID | Unique ID |
| Gender | Male / Female |
| Age | Customer age |
| Annual Income (k$) | Annual income in thousands |
| Spending Score (1-100) | Store-assigned spending score |

---

## 🧠 Algorithms Used

| Algorithm | Why Chosen |
|-----------|------------|
| **K-Means** | Fast, interpretable, works well on compact clusters |
| **DBSCAN** | Detects noise/outliers; no need to specify k |
| **Agglomerative** | Hierarchical; dendrogram shows natural groupings |

---

## 📦 Deployment (Optional)

**Streamlit Community Cloud:**
1. Push to a GitHub repo
2. Visit https://streamlit.io/cloud → New app → connect repo
3. Set main file: `app.py`

**Local only:**
```bash
streamlit run app.py --server.port 8501
```

---

## 👥 Team
- Student 1
- Student 2
- Student 3
- Student 4

*Alamein University — Faculty of Computer Science & Engineering*
