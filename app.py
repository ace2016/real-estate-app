import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import json
import pgeocode
import matplotlib.pyplot as plt
import requests
import cloudpickle
from datetime import datetime
from geopy.distance import geodesic
from shap import explainers
import re # Import regex for postcode formatting
import os # For file operations (logging)
import io # For in-memory file handling for downloads

# Set Matplotlib to use the 'Agg' backend to prevent issues with Streamlit
plt.switch_backend('Agg')

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="London Property Price App", layout="wide", page_icon="🏠")

# RapidAPI Key and Host for UK Property Data API
RAPIDAPI_KEY = st.secrets["api"]["rapidapi_key"]  # RapidAPI key
RAPIDAPI_HOST = "uk-property-data.p.rapidapi.com"

# Define log file path
PREDICTIONS_LOG_FILE = "predictions_log.csv"

# Custom CSS for styling
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main content styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* Header styling */
    h1 {
        color: #2E4053;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .st-emotion-cache-ue6h4q { /* Targeting st.caption */
        text-align: center;
        font-size: 0.9rem;
        color: #607D8B;
        margin-bottom: 2rem;
    }

    /* Container styling for sections */
    .st-emotion-cache-eczf16 { /* Targets st.container */
        background-color: #f8f9fa; /* Light grey background for sections */
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e0e0e0;
    }

    /* Metric styling */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: #1A5276 !important; /* Darker blue for values */
        font-weight: 600;
    }
    [data-testid="stMetricLabel"] {
        color: #495057 !important;
        font-size: 0.9rem !important;
    }

    /* Info, Success, Warning boxes */
    .st-emotion-cache-1f81016 { /* Targets st.info */
        background-color: #e0f7fa; /* Light blue */
        border-left: 5px solid #00BCD4; /* Cyan border */
        border-radius: 8px;
    }
    .st-emotion-cache-1f81016.st-emotion-cache-1f81016-hover:hover { /* Targets st.success */
        background-color: #e8f5e9; /* Light green */
        border-left: 5px solid #4CAF50; /* Green border */
        border-radius: 8px;
    }
    .st-emotion-cache-1f81016.st-emotion-cache-1f81016-warning { /* Targets st.warning */
        background-color: #fff3e0; /* Light orange */
        border-left: 5px solid #FF9800; /* Orange border */
        border-radius: 8px;
    }

    /* Button Styling */
    .st-emotion-cache-l9bibm { /* Targets primary button */
        background-color: #1A5276; /* Dark blue */
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        transition: background-color 0.3s ease, box-shadow 0.3s ease;
    }
    .st-emotion-cache-l9bibm:hover {
        background-color: #2874A6; /* Lighter blue on hover */
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .st-emotion-cache-l9bibm:active {
        background-color: #154360;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Secondary button styling (for "Try Another") */
    .st-emotion-cache-1c7y2gy { /* Targets secondary button */
        background-color: #6c757d; /* Grey */
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: background-color 0.3s ease, box-shadow 0.3s ease;
    }
    .st-emotion-cache-1c7y2gy:hover {
        background-color: #5a6268;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Sidebar styling */
    .st-emotion-cache-vk3305 { /* Targets sidebar content */
        background-color: #f0f2f6; /* Slightly darker grey for sidebar */
        padding: 1.5rem;
        border-right: 1px solid #e0e0e0;
        border-radius: 0 12px 12px 0;
        box-shadow: 2px 0 5px rgba(0,0,0,0.05);
    }
    .st-emotion-cache-1wivf5 { /* Targets sidebar title */
        color: #2E4053;
        font-weight: 600;
        margin-bottom: 1rem;
    }
</style>
"""

# Load model and metadata
try:
    # Ensure 'real_estate_model.pkl' and 'model_metadata.json' are in the same directory
    # as this script or provide the full path.
    with open("real_estate_model.pkl", "rb") as f:
        model = cloudpickle.load(f)
    st.write("✅ Loaded model type:", type(model))
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)

    INFLATION_FACTOR = metadata.get("inflation_factor", 4.2)
    EXPECTED_FEATURES = metadata.get("expected_features", [])
    TRAINING_YEARS = metadata.get("training_years", "1995-1999")

except FileNotFoundError:
    st.error("🚨 Error: Model files (real_estate_model.pkl or model_metadata.json) not found.")
    st.stop()
except Exception as e:
    st.error(f"🚨 Model or metadata loading failed: {str(e)}")
    st.stop()


# Helper functions

def format_postcode_for_display(postcode_raw):
    """
    Formats a UK postcode string for consistent display (e.g., 'E16AN' to 'E1 6AN').
    """
    # Remove all whitespace and convert to uppercase
    cleaned_postcode = re.sub(r'\s+', '', postcode_raw).upper()
    if len(cleaned_postcode) > 3:
        # Take the last 3 characters as the inward code
        inward_code = cleaned_postcode[-3:]
        # The rest is the outward code
        outward_code = cleaned_postcode[:-3]
        return f"{outward_code} {inward_code}"
    return cleaned_postcode # Return as is if too short to format

@st.cache_data
def get_postcode_info(postcode):
    """
    Fetches latitude, longitude, and district for a given UK postcode.
    Uses pgeocode for reliable postcode lookup.
    """
    nomi = pgeocode.Nominatim("gb")
    location = nomi.query_postal_code(postcode)
    return {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "district": location.county_name or "Unknown" # Fallback if county_name is empty
    }

def get_uk_property_data_benchmark(postcode, paon):
    """
    Fetches estimated current value from UK Property Data API (RapidAPI) for benchmarking.
    Requires both postcode and house number/name (PAON).
    """
    url = f"https://{RAPIDAPI_HOST}/propertytools.api.v1.Public/GetPropertyReport"
    
    payload = json.dumps({
        "postcode": postcode,
        "paon": paon
    })

    headers = {
        'x-rapidapi-key': RAPIDAPI_KEY,
        'x-rapidapi-host': RAPIDAPI_HOST,
        'Content-Type': "application/json"
    }
    
    debug_messages = []
    
    try:
        response = requests.post(url, data=payload, headers=headers, timeout=10)
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        # --- Debugging outputs for UK Property Data API response ---
        debug_messages.append(f"UK Property Data API URL: {url}")
        debug_messages.append(f"UK Property Data API Status Code: {response.status_code}")
        debug_messages.append(f"UK Property Data API Request Payload: {payload}")
        debug_messages.append(f"Full JSON Response: {json.dumps(data, indent=2)}") 
        # --- End Debugging outputs ---

        # Extract estimated current value
        estimated_value = data.get("property", {}).get("estimatedCurrentValue", {}).get("value")
        
        if estimated_value is not None:
            return int(estimated_value), debug_messages
        else:
            debug_messages.append(f"⚠️ UK Property Data: 'estimatedCurrentValue' not found in response for '{paon}, {postcode}'. This often means the exact address is not in the API's database.")
            return None, debug_messages
            
    except requests.exceptions.Timeout:
        debug_messages.append(f"⚠️ UK Property Data API request timed out for '{paon}, {postcode}'.")
    except requests.exceptions.RequestException as req_err:
        debug_messages.append(f"⚠️ UK Property Data API network error for '{paon}, {postcode}': {req_err}")
    except json.JSONDecodeError as json_err:
        debug_messages.append(f"⚠️ UK Property Data API JSON decoding error for '{paon}, {postcode}': {json_err}. Response might be malformed.")
    except Exception as e: # Catch any other unexpected errors
        debug_messages.append(f"⚠️ An unexpected error occurred with UK Property Data benchmark for '{paon}, {postcode}': {str(e)}")
    return None, debug_messages

def analyze_sentiment(text):
    """
    Performs a simple rule-based sentiment analysis on the property description.
    Assigns a score and a sentiment label.
    """
    text = text.lower()
    positive_keywords = ['luxury', 'spacious', 'modern', 'bright', 'excellent', 'stunning', 'beautiful', 'generous', 'well-maintained', 'desirable', 'prime', 'fantastic', 'convenient', 'ideal', 'charming', 'immaculate', 'vibrant', 'quiet', 'peaceful', 'secure', 'newly renovated', 'high specification']
    negative_keywords = ['small', 'cramped', 'dated', 'noisy', 'needs work', 'basic', 'limited', 'compact', 'requires renovation', 'overlooked']

    pos_count = sum(text.count(word) for word in positive_keywords)
    neg_count = sum(text.count(word) for word in negative_keywords)

    score = pos_count - neg_count

    if score > 0:
        sentiment = 'positive'
    elif score < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return score, sentiment

def generate_description(property_type, luxury, proximity, garden, parking, transport, sentiment):
    """
    Generates a descriptive string for the property based on selected features and sentiment.
    """
    phrases = []
    if luxury: phrases.append("luxury finish")
    if proximity: phrases.append("close to central London")
    if garden: phrases.append("includes a private garden")
    if parking: phrases.append("private parking space")
    if transport: phrases.append("great transport links")
    
    # Add sentiment-based phrasing
    if sentiment == 'positive':
        sentiment_phrase = "a truly desirable"
    elif sentiment == 'negative':
        sentiment_phrase = "a functional but basic"
    else:
        sentiment_phrase = "a standard"

    base = f"{property_type.lower()} property"
    
    if phrases:
        return f"This {sentiment_phrase} {base} features " + ", ".join(phrases) + "."
    else:
        return f"This is {sentiment_phrase} {base}."


def make_prediction(input_data: dict):
    """
    Preprocesses input data, performs feature engineering, and makes a price prediction.
    """
    df = pd.DataFrame([input_data])

    # Feature engineering (must match features used during model training)
    df['clean_description'] = df['description'].str.lower()
    df['desc_length'] = df['clean_description'].str.len()
    df['word_count'] = df['clean_description'].str.split().str.len()
    df['has_garden'] = df['clean_description'].str.contains('garden').astype(int)
    df['has_parking'] = df['clean_description'].str.contains('parking').astype(int)
    df['has_transport'] = df['clean_description'].str.contains('transport').astype(int)
    df['has_luxury'] = df['clean_description'].str.contains('luxury').astype(int)
    
    # Integrate sentiment analysis
    sentiment_score, sentiment_label = analyze_sentiment(df['description'].iloc[0])
    df['sentiment_score'] = sentiment_score
    df['sentiment'] = sentiment_label

    df['postcode_prefix'] = df['postcode'].str.extract(r'^([A-Z]{1,2})')[0].fillna('UNK')
    df['is_london'] = df['postcode_prefix'].isin(['E', 'EC', 'N', 'NW', 'SE', 'SW', 'W', 'WC']).astype(int)
    
    df['year'] = datetime.now().year
    df['month'] = datetime.now().month
    df['season'] = (df['month'] % 12 // 3 + 1).astype(int) # 1=Winter, 2=Spring, 3=Summer, 4=Autumn
    
    # London city center coordinates
    center = (51.5074, -0.1278) 
    df['distance_to_center'] = df.apply(lambda row: geodesic((row['latitude'], row['longitude']), center).km, axis=1)

    # Initialize fallback status for this prediction run.
    # This will always be False as there is no dynamic district trend data to fall back on.
    st.session_state['district_fallback_used'] = False 

    # 'district_price_trend' is taken directly from input_data, which is 500000 by default.
    # No dynamic lookup for district trend from CSV.
    df['district_price_trend'] = input_data.get("district_price_trend", 0) # Ensure its present, default to 0 if not in input_data

    # Ensure all expected features are present, fill with 0 if missing
    # This is crucial if your model was trained on a specific set of features
    for feat in EXPECTED_FEATURES:
        if feat not in df.columns:
            df[feat] = 0
    
    # Select and reorder columns to match the training data feature order
    try:
        df = df[EXPECTED_FEATURES]
    except KeyError as ke:
        st.error(f"🚨 Feature mismatch: Missing expected feature '{ke}'. Please check `model_metadata.json` and feature engineering.")
        return None, None, None, None, None # Return sentiment_label as well

    try:
        st.write("DEBUG: model type =>", type(model))
        log_pred = model.predict(df)[0]
        pred_price = np.exp(log_pred) * INFLATION_FACTOR
        lower, upper = pred_price * 0.85, pred_price * 1.15 # 15% confidence interval
        
        # Return the prediction results, df_input, sentiment, and the fixed values for logging
        return round(pred_price), round(lower), round(upper), df, sentiment_label
    except Exception as e:
        st.error(f"🚨 Prediction failed: {str(e)}")
        return None, None, None, None, None # Return sentiment_label as well

def log_prediction(input_data, predicted_price, benchmark_price, fallback_status, district_trend_value_used):
    """
    Logs prediction details to a CSV file.
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "postcode": input_data.get("postcode"),
        "property_type": input_data.get("property_type"),
        "duration": input_data.get("duration"),
        "description": input_data.get("description"),
        "latitude": input_data.get("latitude"),
        "longitude": input_data.get("longitude"),
        "district": input_data.get("district"),
        "district_price_trend_used": district_trend_value_used, # This will be 500000 as per input_data
        "predicted_price": predicted_price,
        "benchmark_price": benchmark_price,
        "district_fallback_applied": fallback_status # This will be False as per make_prediction
    }
    
    log_df = pd.DataFrame([log_entry])

    # Check if file exists, if not, create with header. Otherwise, append without header.
    if not os.path.exists(PREDICTIONS_LOG_FILE):
        log_df.to_csv(PREDICTIONS_LOG_FILE, index=False)
    else:
        log_df.to_csv(PREDICTIONS_LOG_FILE, mode='a', header=False, index=False)
    st.success(f"✅ Prediction logged to {PREDICTIONS_LOG_FILE}")


def show_map(lat, lon):
    """
    Displays a map centered at the given latitude and longitude.
    """
    df = pd.DataFrame({'lat': [lat], 'lon': [lon]})
    st.map(df, zoom=12) # Added zoom for better visibility

def explain_shap(input_df):
    st.subheader("📊 Model Explanation Report")
    
    try:
        # Extract Model Components 
        preprocessor = model.named_steps['preprocessor']
        regressor = model.named_steps['regressor']
        
        # Build Text Feature Dominance Analysis 
        tfidf = preprocessor.named_transformers_['text'].named_steps['tfidf']
        svd = preprocessor.named_transformers_['text'].named_steps['svd']
        
        # Calculate text feature importance in the model
        text_feature_indices = [i for i, name in enumerate(preprocessor.get_feature_names_out()) 
                              if name.startswith("text_truncatedsvd")]
        
        if hasattr(regressor, 'coef_'):
            text_coef_sum = np.sum(np.abs(regressor.coef_[text_feature_indices]))
            total_coef_sum = np.sum(np.abs(regressor.coef_))
            text_dominance = text_coef_sum / total_coef_sum
        else:
            # Fallback for non-linear models
            text_dominance = len(text_feature_indices) / len(preprocessor.get_feature_names_out())
        
        # SHAP Calculation (for visualization only)
        transformed_input = preprocessor.transform(input_df)
        feature_names = preprocessor.get_feature_names_out()
        explainer = shap.Explainer(regressor, transformed_input)
        shap_values = explainer(transformed_input)
        shap_row = shap_values[0]
        tfidf_terms = tfidf.get_feature_names_out()

        # Generate readable names for text_truncatedsvd features 
        svd_feature_map = {}
        for i, component in enumerate(svd.components_):
            top_idxs = component.argsort()[-3:][::-1]
            keywords = [tfidf_terms[j] for j in top_idxs]
            svd_feature_map[f"text_truncatedsvd{i}"] = f"Topic: {' / '.join(keywords)}"

        # Create final readable names for all features
        readable_feature_names = []
        for name in feature_names:
            if name in svd_feature_map:
                readable_feature_names.append(svd_feature_map[name])
            elif name.startswith("text_truncatedsvd"):
                readable_feature_names.append(f"Topic: {name}")
            else:
                readable_feature_names.append(name.replace("_", " ").title())

        
        # Generate Stakeholder Summary 
        st.markdown("### 🔍 Model Behavior Summary")
        
        if text_dominance > 0.05:
            conclusion = (
                "This analysis reveals that <b>textual descriptions</b> dominate the model's decision-making process, "
                f"accounting for {text_dominance:.0%} of the model's learned patterns. "
                "The specific wording in property descriptions is the primary driver of predictions."
            )
        elif text_dominance > 0.0:
            conclusion = (
                "The model uses a <b>balanced combination</b> of textual descriptions and traditional features, "
                f"with text accounting for {text_dominance:.0%} of the learned patterns."
            )
        else:
            conclusion = (
                "Traditional property features dominate this model, "
                f"with text descriptions contributing most of the learned patterns."
            )
        
        # Text Feature Deep Dive 
        st.markdown(f"#### Text Analysis")
        
        # Get top 3 most important text features from model coefficients
        if hasattr(regressor, 'coef_'):
            top_text_indices = np.argsort(np.abs(regressor.coef_[text_feature_indices]))[-3:][::-1]
            top_text_features = [preprocessor.get_feature_names_out()[text_feature_indices[i]] 
                              for i in top_text_indices]
        else:
            top_text_features = [f"text_truncatedsvd{i}" for i in range(3)]
        
        # Map to keywords
        
        text_patterns = {}
        for i, feat in enumerate(top_text_features):
            component_idx = int(feat.replace("text_truncatedsvd", ""))
            top_idxs = svd.components_[component_idx].argsort()[-3:][::-1]
            text_patterns[feat] = {
                'display_name': f"Description Pattern {i+1} ({feat})",
                'keywords': [tfidf_terms[idx] for idx in top_idxs],
                'direction': "decreases" if (hasattr(regressor, 'coef_')) and 
                              (regressor.coef_[text_feature_indices[component_idx]] < 0) 
                              else "increases"
            }
        
        # Build the report
        st.markdown("### Key Findings")
        st.write(conclusion)

        st.markdown("#### Most Significant Text Patterns")
        for feat in top_text_features:
            pattern = text_patterns[feat]
            st.write(
                f"- **{pattern['display_name']}**: {', '.join(pattern['keywords'])} "
                f"(typically {pattern['direction']} price)"
            )

        st.markdown("#### What This Means")
        st.write(
            f'Properties containing phrases like "{", ".join(text_patterns[top_text_features[0]]["keywords"])}" '
            "are most influential in the model's predictions."
        )

        # Visualizations 
        st.markdown("### Feature Impact Analysis")
        
        # Plot 1: Waterfall Plot
        st.markdown("#### Waterfall Plot (Detailed Impact)")
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_row,
                base_values=explainer.expected_value,
                feature_names=readable_feature_names,
                data=transformed_input[0]
            ),
            max_display=10,
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        # Add this right after your SHAP calculation section (after getting shap_row)

        # Create Top Features Table 
        st.markdown("### Top 5 Most Impactful Features")

        # Create a DataFrame of feature impacts
        impact_df = pd.DataFrame({
            'Feature': readable_feature_names,
            'SHAP Value': shap_row.values,
            'Absolute Impact': np.abs(shap_row.values)
        }).sort_values('Absolute Impact', ascending=False).head(5)

        # Display the table with color coding
        st.dataframe(
            impact_df.style.format({
                'SHAP Value': '{:.3f}',
                'Absolute Impact': '{:.3f}'
            }).apply(
                lambda x: ['color: red' if v < 0 else 'color: green' for v in x] 
                if x.name == 'SHAP Value' else [''] * len(x),
                axis=1
            ),
            height=200
        )
        
        # Interpretation Guide 
        st.markdown("### How to Interpret This")
        st.write("""
        1. **Base Value**: The model's average prediction across all properties
        2. **Red Bars**: Features increasing the predicted price
        3. **Blue Bars**: Features decreasing the predicted price
        4. **f(x)**: Final prediction for this specific property
        """)
        
    except Exception as e:
        st.error(f"Explanation failed: {str(e)}")



# Streamlit UI
st.markdown(custom_css, unsafe_allow_html=True) # Inject custom CSS

st.markdown("<h1 style='color:#2E4053; text-align: center;'>London Property Price Estimator (2024)</h1>", unsafe_allow_html=True) 
st.caption(f"Smart valuation using historical UK housing data ({TRAINING_YEARS}) adjusted for 2024 using inflation and market trends.")

# Initialize session state for manual_description if not already set
if "manual_description_text" not in st.session_state:
    st.session_state["manual_description_text"] = ""
# Initialize description_choice in session state
if "description_choice_state" not in st.session_state:
    st.session_state["description_choice_state"] = "Generate from Checkboxes (Simple)"
# Initialize fallback status (will always be False now as no dynamic trend data)
if 'district_fallback_used' not in st.session_state:
    st.session_state['district_fallback_used'] = False


# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("🏡 Property Details")
    with st.form("predict_form"):
        house_number_name = st.text_input("🏠 House Number/Name (e.g., 59 or 'South Quay Plaza, Canary Wharf, London')", value="36").strip()
        postcode_raw = st.text_input("📍 Postcode (e.g., E14 9SH)", value="CR0 2NG").strip()
        postcode = format_postcode_for_display(postcode_raw) # Format for internal use and display

        property_type = st.selectbox("🏢 Property Type", ['Flat', 'Terraced', 'Semi-Detached', 'Detached'])
        duration = st.selectbox("⏳ Ownership Type", ['Leasehold', 'Freehold'])
        
        st.subheader("✨ Key Features")
        luxury = st.checkbox("Luxury Finish")
        garden = st.checkbox("Private Garden")
        parking = st.checkbox("Private Parking")
        transport = st.checkbox("Near Transport Links")
        proximity = st.checkbox("Close to Central London")
        
        st.markdown("---")
        st.subheader("📝 Description Input")
        # Removed the on_change callback as it's not allowed inside st.form
        description_choice = st.radio("How would you like to provide the description?", 
                                      ("Generate from Checkboxes (Simple)", "Enter Manually (Detailed)"),
                                      key="description_choice_radio_button")

        manual_description = ""
        if description_choice == "Enter Manually (Detailed)": # Use description_choice directly after form submission
            manual_description = st.text_area(
                "✍️ Manual Property Description",
                value=st.session_state["manual_description_text"], # Pre-fill from session state
                height=150,
                help="Provide a detailed description similar to real property listings to better influence text-based features.",
                key="manual_description_text_area" # Add a unique key
            )
            # Update session state when text area changes (this happens automatically with key)
            st.session_state["manual_description_text"] = manual_description
        else:
            # If "Generate from Checkboxes" is selected, clear manual description from session state
            if st.session_state["manual_description_text"] != "":
                st.session_state["manual_description_text"] = ""
            manual_description = "" # Ensure manual_description is empty if not chosen
        
        st.markdown("<br>", unsafe_allow_html=True) # Add some space
        submit = st.form_submit_button("🔍 Predict Price", type="primary")

# --- Main Content Area for Results ---
if submit:
    if not postcode_raw or not house_number_name: # Ensure house number/name is also provided
        st.error("Please enter both a postcode and a house number/name to get a prediction.")
    else:
        with st.spinner("🔎 Fetching location and generating prediction..."):
            loc = get_postcode_info(postcode) # Use formatted postcode for pgeocode
            
            if pd.isna(loc['latitude']) or pd.isna(loc['longitude']):
                st.error("⚠️ Invalid postcode entered. Please check the postcode and try again.")
            else:
                # Determine which description to use based on the selected radio button value
                if description_choice == "Enter Manually (Detailed)" and manual_description.strip():
                    description_to_use = manual_description.strip()
                else:
                    # Generate description based on checkboxes if no manual input or choice is 'Generate'
                    temp_desc = generate_description(property_type, luxury, proximity, garden, parking, transport, sentiment='neutral') 
                    _, sentiment_label_for_prediction = analyze_sentiment(temp_desc)
                    description_to_use = generate_description(property_type, luxury, proximity, garden, parking, transport, sentiment_label_for_prediction)
                
                # Analyze sentiment for the chosen description
                _, final_sentiment_label = analyze_sentiment(description_to_use)

                input_data = {
                    "postcode": postcode.upper(), # Use formatted postcode for model input
                    "property_type": property_type,
                    "duration": duration,
                    "description": description_to_use, # Use the chosen description
                    "latitude": loc['latitude'],
                    "longitude": loc['longitude'],
                    # Provide a default or more robust way to get district if 'Unknown' is common
                    "district": loc['district'] if loc['district'] != "Unknown" else "CAMDEN", 
                    "district_price_trend": 500000 # This value is fixed 
                }

                # Call make_prediction, now returning only the relevant prediction outputs
                y_pred, lower, upper, df_input, _ = make_prediction(input_data) 

                if y_pred is not None::
                    # Benchmarking using UK Property Data API
                    benchmark_price, uk_property_data_debug_messages = get_uk_property_data_benchmark(postcode, house_number_name)

                    # Log the prediction with all details.
                    # district_price_trend_used and district_fallback_applied will reflect the fixed values.
                    log_prediction(input_data, y_pred, benchmark_price, st.session_state['district_fallback_used'], input_data["district_price_trend"])

                    # Display fallback warning if applicable (will always be False now)
                    if st.session_state['district_fallback_used']:
                        st.warning("⚠️ The prediction for this postcode used a fallback district value, which might affect accuracy due to missing specific trend data.")

                    st.markdown("## 📈 Prediction Results")
                    with st.container():
                        st.subheader("💰 Your Property Valuation")
                        col_pred, col_ci = st.columns(2)
                        with col_pred:
                            st.metric(label="Estimated 2024 Price", value=f"£{y_pred:,.0f}")
                        with col_ci:
                            st.metric(label="Confidence Interval", value=f"£{lower:,.0f} – £{upper:,.0f}")
                    
                    with st.container():
                        st.subheader("📊 Market Benchmark")
                        if benchmark_price:
                            st.metric(label=f"UK Property Data Benchmark Price for '{house_number_name}, {postcode}'", value=f"£{benchmark_price:,.0f}")
                            diff = y_pred - benchmark_price
                            color = "#e0f7fa" if abs(diff) < 50000 else ("#ffe0b2" if diff > 0 else "#c8e6c9")
                            st.markdown(f"<div style='background-color:{color};padding:1rem;border-radius:8px;'>"
                                        f"<strong>🔎 Difference from Benchmark:</strong> £{diff:,.0f}</div>", unsafe_allow_html=True)
                        else:
                            st.warning(f"📊 UK Property Data Benchmark Price for '{house_number_name}, {postcode}' is currently unavailable. This often happens if the exact address is not found in the API's database.")
                        
                        with st.expander("ℹ️ UK Property Data API Debug Details (Click to expand)", expanded=False):
                            if uk_property_data_debug_messages:
                                for msg in uk_property_data_debug_messages:
                                    st.code(msg)
                            else:
                                st.info("No specific debug messages from UK Property Data API call.")

                    with st.container():
                        st.subheader("📝 Property Description Used")
                        st.info(f"**Sentiment**: {final_sentiment_label.capitalize()}. {description_to_use}")

                    with st.container():
                        st.subheader("📍 Location Overview")
                        col_map, col_panel = st.columns([1.2, 1.8])
                        with col_map:
                            st.markdown("#### Map View")
                            show_map(loc['latitude'], loc['longitude'])

                        with col_panel:
                            st.markdown("#### Rationale Behind Prediction")
                            st.markdown(f"""
                            - **District**: {loc['district']}
                            - **Adjusted to 2024** using inflation factor of ×{INFLATION_FACTOR}
                            - **Proximity to city center**: {round(df_input['distance_to_center'].values[0], 1)} km
                            - Model trained on historical data from: **{TRAINING_YEARS}**
                            - **Description Sentiment**: {final_sentiment_label.capitalize()}

                            Features like "luxury finish", "private garden", and "great transport links" positively influence the valuation. The model now incorporates a simple sentiment analysis based on keywords, which can also affect the price.
                            """)

                    with st.expander("Feature Impact (SHAP Explanation)", expanded=False):
                        if df_input is not None:
                            explain_shap(df_input)
                        else:
                            st.warning("Cannot generate SHAP explanation: Input data for prediction was not available.")
                    
                    st.markdown("---")
                    # "Try Another" button
                    if st.button("🔄 Try Another Property", type="secondary"):
                        st.session_state["rerun"] = True # Set a state variable to trigger rerun
                        st.experimental_rerun() # Rerun the app

# --- About This App Section (Always visible at the bottom) ---
st.markdown("---")
with st.expander("ℹ️ About This App", expanded=False):
    st.markdown(f"""
    This application provides an estimated valuation for properties in London, leveraging a machine learning model trained on historical UK housing data.

    **Key Features:**
    - **Smart Valuation:** Utilizes a trained model to predict property prices, adjusted for current market trends and inflation.
    - **Benchmark Comparison:** Compares the model's prediction against an estimated current value from the UK Property Data API.
    - **Explainable AI (SHAP):** Provides insights into which features most influence the price prediction.
    - **Sentiment Analysis:** Incorporates sentiment from property descriptions to refine valuations.

    **Model & Data:**
    - The underlying model was trained on historical data from **{TRAINING_YEARS}**.
    - An inflation factor of **×{INFLATION_FACTOR}** is applied to adjust predictions to 2024 values.
    - Location data is powered by `pgeocode`.
    - Benchmark data is sourced from the `UK Property Data API` via RapidAPI.

    **Disclaimer:**
    This tool provides estimates for informational purposes only and should not be used as the sole basis for financial decisions. Property values can be influenced by many factors not captured here. Always consult with a qualified real estate professional for accurate valuations.

    **Contact:**
    [Owhofasa Anslem]
    [github.com/ace2016; in/owhofasa-emrobowasan]
    """)

# Handle rerun for "Try Another Property" button
if "rerun" in st.session_state and st.session_state["rerun"]:
    del st.session_state["rerun"] # Clear the flag
    st.experimental_rerun() # This will actually rerun the script
