# 🏠 London Property Price Prediction App (2025 Edition)

Welcome to a smart, explainable ML-powered real estate valuation app built using **Streamlit**. This app predicts **London property prices** in 2024 using a historical UK housing dataset (1995–1999), adjusted with a 4.2x inflation factor. It combines geolocation, sentiment, description patterns (via NLP), and traditional features to produce an interpretable price prediction.

> 🚀 Repo: [github.com/ace2016/real-estate-app](https://github.com/ace2016/real-estate-app)
---

## 🔍 Core Features

- 🔢 **Price Estimation** based on location, sentiment, property description, and more.
- 🌐 **Benchmark API Integration** with the UK Property Data API (RapidAPI).
- 🧠 **SHAP Explainability**: Waterfall plots + Top Feature Contributions, understand what features influenced your prediction.
- 📝 **Sentiment Analysis** of descriptions to impact prediction.
- 📉 **Text & Structured Feature Modeling** with TF-IDF + Truncated SVD.
- 📍 **Interactive Map View** to visualize the property location.
- 🧮 **Inflation Adjusted** predictions (x4.2 for 1995–1999 to 2024).
- 📈 **Confidence Intervals** around each prediction.
- 💡 **Custom CSS styling** for polished UI.
- ✅ **Fallback Logic** for unknown postcodes or districts.

---

### 🗂️ Dataset Used

[UK HM Land Registry Price Paid Data](https://www.gov.uk/guidance/about-the-price-paid-data)

**Dataset:** UK House Price data (1995–1999), geocoded and enriched with OSM data.

---
### 🗁 Folder structure

├── app.py                      # Main Streamlit application\
├── real_estate_model.pkl       # Trained ML pipeline (preprocessor + model)\
├── district_trends.csv         # Generated using the rolling mean\
├── model_metadata.json         # Inflation factor, expected features, training years\
├── predictions_log.csv         # Auto-generated log of predictions\
├── requirements.txt\
└── README.md

---

## 📂 Dataset and Fallback Logic

Although `district_trends.csv` was generated using rolling means across postcode districts, it was **not used** in this version of the app.

Instead, a **universal fallback value of £500,000** was used across all London postcodes for the feature `district_price_trend`. This ensures consistent predictions even when district-level data is missing or ambiguous.

---

## 🧠 Model Details

- Pipeline: `TfidfVectorizer → TruncatedSVD → ColumnTransformer → XGBoost`
- Model: Gradient Boosted Trees via `XGBoost`, `CatBoost`, `LightGBM` with `CatBoost` as the best model at 0.726 R square
- Text Features: Extracted from property descriptions, reduced to 50 latent topics
- Structured Features: Property type, ownership, distance to city center, amenities, etc.
- Explainability: Integrated via **SHAP** (`shap.Explainer + shap.waterfall + summary_plot`)

---

## ⚙️ How to Run the App Locally

Clone the repository and install dependencies inside a virtual environment.

```bash
# 1. Clone the repo
git clone https://github.com/ace2016/real-estate-app.git
cd real-estate-app

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install the required dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py

```
### **NOTE:** *The model training code is not attached to this repo, I will upload that at a later date. Alongside the enriched dataset of 1M records.*
