import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

import warnings
warnings.filterwarnings('ignore')

# Load Deployment Artifacts
@st.cache_resource
def load_artifacts():
    base_path= os.getenv("ARTIFACTS_PATH", "Deployment_Artifacts")

    try:
        model= joblib.load(os.path.join(base_path, "ridge_model.pkl"))
        scaler= joblib.load(os.path.join(base_path, "scaler.joblib"))
        ordinal_maps= joblib.load(os.path.join(base_path, "ordinal_mappings.joblib"))
        full_feature_list= joblib.load(os.path.join(base_path, "final_features_columns.joblib"))
        top_10_features= joblib.load(os.path.join(base_path, "top_10_input_features.joblib"))

        return model, scaler, ordinal_maps, full_feature_list, top_10_features
    
    except FileNotFoundError as e:
        st.error(f"❌ Deployment artifact not found: {e}")
        st.stop()

model, scaler, ordinal_mappings, FULL_FEATURE_LIST, top_10_features = load_artifacts()

# Preprocessing + Prediction Functions
HIGHLY_SKEWED_FEATURES= ["GrLivArea", "1stFlrSF", "TotalBsmtSF", "GarageArea"]

def apply_preprocessing(input_df, ordinal_maps, full_feature_list, scaler):
    df= input_df.copy()

    # Ordinal encoding (skip manually numeric cols)
    skip_cols= ["ExterQual", "KitchenQual"]
    for col, mapping in ordinal_maps.items():
        if col in df.columns and col not in skip_cols:
            df[col]= df[col].map(mapping).fillna(0)

    # Log-transform skewed features
    for col in HIGHLY_SKEWED_FEATURES:
        if col in df.columns:
            df[col]= np.log1p(df[col])

    # Ensure all model columns exist
    missing_cols= [col for col in full_feature_list if col not in df.columns]
    if missing_cols:
        df= pd.concat([df, pd.DataFrame(0, index=df.index, columns=missing_cols)], axis=1)

    df= df[full_feature_list]
    df= df.drop(columns=["Id", "SalePrice", "SalePrice_Log"], errors="ignore")

    # Scale numerical features
    try:
        df[scaler.feature_names_in_]= scaler.transform(df[scaler.feature_names_in_])
    except ValueError:
        st.error("⚠️ Scaling error: Feature mismatch detected")
        st.stop()

    return df


def predict_price(processed_data):
    log_prediction= model.predict(processed_data)
    final_price= np.expm1(log_prediction)

    return final_price


# Streamlit App Layout
st.set_page_config(
    page_title="🏡 Ames House Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
    <style>
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    .info-box p {
        color: #e2e8f0 !important;
    }
    
    .info-box strong {
        color: #ffffff !important;
    }
    
    /* Price display styling */
    .price-display {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
        margin: 2rem 0;
    }
    
    .price-display h2 {
        color: white;
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .price-label {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    
    /* Section headers */
    .section-header {
        color: #1e40af;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3b82f6;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #1e293b;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #e2e8f0;
    }
    
    /* Button styling improvements */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f5f7fa;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    /* Dark instruction box */
    .dark-instruction-box {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    .dark-instruction-box p {
        margin: 0;
        color: #e2e8f0 !important;
    }
    
    .dark-instruction-box strong {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>🏡 Ames House Price Predictor</h1>
        <p>AI-Powered Real Estate Valuation Tool</p>
    </div>
""", unsafe_allow_html=True)

# Info section
st.markdown("""
    <div class="info-box">
        <p style="margin: 0; font-size: 1.1rem;">
            <strong>📊 About the Model:</strong> This housing price predictor was developed after 
            experimenting with multiple regression algorithms, including Linear Regression, Ridge, 
            Lasso, ElasticNet, Decision Tree, Random Forest, and XGBoost. After fine-tuning and evaluating 
            the top-performing models, <strong>Ridge Regression</strong> emerged as the most consistent 
            and accurate, offering the best balance between bias and variance.
        </p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.95rem;">
            <strong>Model Performance:</strong> R² Score = <strong>0.915</strong> | 
            <strong>Explained Variance:</strong> 91.54% on test data
        </p>
    </div>
""", unsafe_allow_html=True)

# Feature Inputs
top_10_dict = {
    "OverallQual": ("Overall Material and Finish Quality (1-10)", 7, 1, 10, "slider"),
    "GrLivArea": ("Above Ground Living Area (Sq Ft)", 1500, 500, 4000, "number_input"),
    "GarageCars": ("Garage Capacity (0-4 Cars)", 2, 0, 4, "slider"),
    "1stFlrSF": ("First Floor Area (Sq Ft)", 1000, 500, 3000, "number_input"),
    "YearBuilt": ("Year Built", 2000, 1900, 2020, "slider"),
    "ExterQual": ("Exterior Material Quality (1-5)", 4, 1, 5, "slider"),
    "TotalBsmtSF": ("Total Basement Area (Sq Ft)", 1000, 0, 3000, "number_input"),
    "KitchenQual": ("Kitchen Quality (1-5)", 4, 1, 5, "slider"),
    "GarageArea": ("Garage Area (Sq Ft)", 480, 0, 1200, "number_input"),
    "FullBath": ("Full Bathrooms Above Grade", 2, 0, 4, "slider"),
}

# Session State Management- Initialize defaults once
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs= None

# Initialize widget values in session state if not present
for feature, (label, default, min_val, max_val, widget) in top_10_dict.items():
    key= f"input_{feature}"
    if key not in st.session_state:
        st.session_state[key]= default

# Reset callback function
def reset_inputs():
    for feature, (label, default, min_val, max_val, widget) in top_10_dict.items():
        key= f"input_{feature}"
        st.session_state[key]= default
    st.session_state["show_reset_spinner"]= True
    st.toast("Inputs have been reset to default values.", icon="🔄")

# Sidebar Inputs
st.sidebar.markdown("## ⚙️ Property Features")
st.sidebar.markdown("---")

user_inputs= {}
for feature, (label, default, min_val, max_val, widget) in top_10_dict.items():
    key= f"input_{feature}"
    
    if widget=="slider":
        user_inputs[feature]= st.sidebar.slider(
            label, min_val, max_val, key=key
        )
    else:
        user_inputs[feature]= st.sidebar.number_input(
            label, min_value=min_val, max_value=max_val, key=key
        )

# Buttons
col1, col2= st.columns([1, 1], gap="medium")
predict_clicked= col1.button("🚀 Predict House Price", width='stretch', type="primary")
col2.button("🔄 Reset to Defaults", width='stretch', on_click=reset_inputs)

# Show reset spinner if flag is set
if st.session_state.get("show_reset_spinner", False):
    with st.spinner("🔄 Resetting inputs..."):
        time.sleep(0.8)
    st.session_state["show_reset_spinner"] = False
    st.rerun()


# Prediction Logic
if predict_clicked:
    full_input_dict= {col: 0 for col in FULL_FEATURE_LIST}
    for col, value in user_inputs.items():
        full_input_dict[col]= value

    raw_input_df= pd.DataFrame([full_input_dict])

    with st.spinner("🔄 Processing data and generating prediction..."):
        time.sleep(1.2)
        processed_data= apply_preprocessing(raw_input_df, ordinal_mappings, FULL_FEATURE_LIST, scaler)
        predicted_price= predict_price(processed_data)

    price = predicted_price[0]
    
    # Price display
    st.markdown(f"""
        <div class="price-display">
            <div class="price-label">Estimated Property Value</div>
            <h2>${price:,.0f}</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Show confidence metrics
    col_a, col_b, col_c= st.columns(3)
    with col_a:
        st.metric("Model Confidence", "91.54%", help="Based on R² score")
    with col_b:
        st.metric("Price Range", f"${price*0.9:,.0f} - ${price*1.1:,.0f}", help="±10% estimate range")
    with col_c:
        st.metric("Prediction Status", "✅ Complete")
    
    st.session_state.last_inputs= user_inputs.copy()
    st.session_state.last_predicted_price= price


# Display Last Inputs
if st.session_state.last_inputs:
    st.markdown("---")
    st.markdown('<div class="section-header">🔍 Your Last Prediction</div>', unsafe_allow_html=True)
    
    # Show the last predicted price in an expander
    if 'last_predicted_price' in st.session_state:
        with st.expander("💰 View Last Predicted Price", expanded=False):
            st.markdown(f"""
                <div class="price-display">
                    <div class="price-label">Last Predicted Property Value</div>
                    <h2>${st.session_state.last_predicted_price:,.0f}</h2>
                </div>
            """, unsafe_allow_html=True)
    
    df_display= pd.DataFrame([st.session_state.last_inputs]).T
    df_display.columns= ["Value"]
    df_display.index.name= "Feature"
    df_display= df_display.reset_index()
    
    with st.expander("📋 View Input Details", expanded=False):
        st.dataframe(df_display, width='stretch', hide_index=True)


# Batch CSV Upload
st.markdown("---")
st.markdown('<div class="section-header">📂 Batch Predictions</div>', unsafe_allow_html=True)

st.markdown("""
    <div class="dark-instruction-box">
        <p>
            <strong>📥 Upload CSV File:</strong> Process multiple properties at once by uploading a CSV 
            file with the same feature columns. Download the results with predicted prices included.
        </p>
    </div>
""", unsafe_allow_html=True)

uploaded_file= st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Upload a CSV file containing property features for batch prediction"
)

if uploaded_file:
    try:
        input_csv= pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded successfully! Found {len(input_csv)} properties to analyze.")

        input_review= input_csv.copy()
        input_review.index= input_review.index+1
        input_review.index.name= "Property ID"
        input_review.rename(columns={
            "OverallQual": "Overall Quality",
            "GrLivArea": "Living Area (sqft)",
            "GarageCars": "Garage Capacity",
            "1stFlrSF": "First Floor Area (sqft)",
            "YearBuilt": "Year Built",
            "ExterQual": "Exterior Quality",
            "TotalBsmtSF": "Basement Area (sqft)",
            "KitchenQual": "Kitchen Quality",
            "GarageArea": "Garage Area (sqft)",
            "FullBath": "Full Bathrooms"

        }, inplace=True)

        with st.expander("📊 Preview Uploaded Data", expanded=True):
            st.dataframe(input_review.head(10), width='stretch')

        if st.button("🚀 Generate Predictions for All Properties", width='stretch', type="primary"):
            with st.spinner("🔄 Processing batch predictions..."):
                processed_csv= apply_preprocessing(input_csv, ordinal_mappings, FULL_FEATURE_LIST, scaler)
                predictions= predict_price(processed_csv)
                input_csv["Predicted_SalePrice"] = predictions
                time.sleep(1)

            st.success(f"✅ Successfully predicted prices for {len(input_csv)} properties!")
            
            # Show statistics
            col_x, col_y, col_z = st.columns(3)
            with col_x:
                st.metric("Total Properties", len(input_csv))
            with col_y:
                st.metric("Average Price", f"${input_csv['Predicted_SalePrice'].mean():,.0f}")
            with col_z:
                st.metric("Price Range", f"${input_csv['Predicted_SalePrice'].min():,.0f} - ${input_csv['Predicted_SalePrice'].max():,.0f}")

            results_preview= input_csv[["Predicted_SalePrice"]].copy()
            results_preview.index= results_preview.index+1
            results_preview.index.name= "Property ID"
            results_preview.rename(columns={"Predicted_SalePrice": "Predicted Sale Price"}, inplace=True)

            
            with st.expander("📈 Preview Results", expanded=True):
                st.dataframe(results_preview.head(10), width='stretch')

            csv_data= input_csv.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Complete Results (CSV)",
                data=csv_data,
                file_name="ames_housing_predictions.csv",
                mime="text/csv",
                width='stretch'
            )
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("💡 Please ensure your CSV has the correct column names and data format.")

# Footer
st.markdown("---")