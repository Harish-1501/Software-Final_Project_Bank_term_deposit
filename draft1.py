import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from typing import Dict, List, Tuple, Optional
import h2o
from h2o.frame import H2OFrame

import streamlit as st
import pandas as pd

MODEL_PATH = "saved_model/StackedEnsemble_AllModels_1_AutoML_1_20250807_154043"

def _init_h2o_and_load_model(model_path: str):
    # Initialize H2O (cached so it runs once per session)
    h2o.init()
    model = h2o.load_model(model_path)
    return model



def _predict_h2o(df: pd.DataFrame, model, ui_yes_label: str = "Yes") -> pd.DataFrame:
    """
    Runs H2O predictions on a pandas DataFrame and returns a DataFrame
    containing the original columns + 'prediction' (Yes/No) + 'confidence' (%).
    """
    from h2o.frame import H2OFrame
    import numpy as np

    hf = H2OFrame(df)
    raw = model.predict(hf).as_data_frame()  # typically has 'predict' + prob cols

    # --- figure out class domain (order of classes as trained) ---
    try:
        classes = model._model_json["output"]["domains"][-1] or []
    except Exception:
        classes = []
    classes_lower = [str(c).lower() for c in classes]

    # --- identify probability columns (numeric only) ---
    num_cols = raw.select_dtypes(include="number").columns.tolist()

    # helper: normalize a label to Yes/No for UI
    def to_ui_label(x: str) -> str:
        xl = str(x).strip().lower()
        if xl in {"yes", "y", "true", "1"}:
            return "Yes"
        if xl in {"no", "n", "false", "0"}:
            return "No"
        # fallback (treat unknown as No, will be corrected by prob if needed)
        return "Yes" if xl == "yes" else "No"

    # If 'predict' is missing, derive it from the max prob column
    if "predict" not in raw.columns:
        if num_cols:
            # pick the class name of the argmax if columns are class-named, else just threshold p1/any
            max_idx = raw[num_cols].values.argmax(axis=1)
            if classes:
                preds_series = [classes[i] if i < len(classes) else "yes" for i in max_idx]
            else:
                preds_series = ["yes" if i == 1 else "no" for i in max_idx]
            raw["predict"] = preds_series
        else:
            # no numeric probs? highly unlikely—fallback to 'no'
            raw["predict"] = "no"

    # Build confidence vector matching the predicted class
    pred_str = raw["predict"].astype(str)
    pred_lower = pred_str.str.lower()

    conf = None

    # Case A: probability columns are named by class, e.g. 'yes','no'
    # Try exact/ci match
    class_named_cols = {c.lower(): c for c in raw.columns}
    if any(k in class_named_cols for k in {"yes", "no"}):
        # pull the prob for the predicted label if available
        def get_row_conf(row):
            lab = str(row["predict"]).lower()
            col = class_named_cols.get(lab)
            if col in num_cols:
                return row[col]
            return np.nan
        conf = raw.apply(get_row_conf, axis=1)

    # Case B: classic H2O naming p0/p1 and we know the class order
    if conf is None or conf.isna().all():
        if classes_lower and {"p0", "p1"}.issubset(set(raw.columns)):
            # map predicted label -> its class index -> p{idx}
            def get_row_conf_p(row):
                lab = str(row["predict"]).lower()
                if lab in classes_lower:
                    idx = classes_lower.index(lab)
                    col = f"p{idx}"
                    if col in num_cols:
                        return row[col]
                return np.nan
            conf = raw.apply(get_row_conf_p, axis=1)

    # Case C: fall back to the max numeric probability per row
    if conf is None or conf.isna().all():
        if num_cols:
            conf = raw[num_cols].max(axis=1)
        else:
            conf = pd.Series(0.5, index=raw.index)

    # Normalize to numeric and bound to [0,1]
    conf = pd.to_numeric(conf, errors="coerce").fillna(0.5).clip(0, 1)

    # Final UI fields
    out = df.copy()
    out["prediction"] = pred_lower.map(to_ui_label)
    out["confidence"] = (conf * 100).round(2)
    return out




# Configure Streamlit page
st.set_page_config(
    page_title="ML Term Deposit Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
def load_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styles */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        font-weight: 400;
        opacity: 0.9;
    }
    
    /* Card Styles */
    .card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
        margin-bottom: 2rem;
    }
    
    .card-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Success/Error Cards */
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .error-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    /* Form Styles */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        transition: all 0.2s;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .stNumberInput > div > div {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        transition: all 0.2s;
    }
    
    .stNumberInput > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* File Uploader */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8faff;
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .css-1d391kg .css-1v0mbdj {
        color: white;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Header Component
def render_header():
    st.markdown("""
    <div class="header-container">
        <div class="header-title">🎯 Term Deposit Prediction System</div>
        <div class="header-subtitle">Advanced ML-powered customer analysis for banking decisions</div>
    </div>
    """, unsafe_allow_html=True)

# Metric Cards Component
def render_metrics(total: int, will_subscribe: int, wont_subscribe: int):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total}</div>
            <div class="metric-label">Total Customers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="metric-value">{will_subscribe}</div>
            <div class="metric-label">Will Subscribe</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
            <div class="metric-value">{wont_subscribe}</div>
            <div class="metric-label">Won't Subscribe</div>
        </div>
        """, unsafe_allow_html=True)

# Prediction Result Component
def render_prediction_result(prediction: str, confidence: float):
    if prediction.lower() == 'yes':
        st.markdown(f"""
        <div class="success-card">
            <h2>✅ Will Subscribe</h2>
            <p style="font-size: 1.2rem; margin: 1rem 0;">
                This customer is likely to subscribe to a term deposit
            </p>
            <p style="font-size: 1.1rem; font-weight: 600;">
                Confidence: {confidence:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="error-card">
            <h2>❌ Won't Subscribe</h2>
            <p style="font-size: 1.2rem; margin: 1rem 0;">
                This customer is unlikely to subscribe to a term deposit
            </p>
            <p style="font-size: 1.1rem; font-weight: 600;">
                Confidence: {confidence:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)

# Create Interactive Pie Chart
def create_pie_chart(predictions_df: pd.DataFrame) -> go.Figure:
    # Count predictions
    prediction_counts = predictions_df['prediction'].value_counts()
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=['Will Subscribe', 'Won\'t Subscribe'],
        values=[
            prediction_counts.get('Yes', 0),
            prediction_counts.get('No', 0)
        ],
        hole=0.4,
        marker_colors=['#4facfe', '#fa709a'],
        textinfo='label+percent+value',
        textfont_size=14,
        marker=dict(
            line=dict(color='white', width=3)
        )
    )])
    
    fig.update_layout(
        title={
            'text': 'Prediction Distribution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'family': 'Inter'}
        },
        font=dict(family="Inter", size=14),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        height=500,
        margin=dict(t=80, b=80, l=50, r=50)
    )
    
    return fig

# Download CSV Function
def get_csv_download_link(df: pd.DataFrame, filename: str) -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-link">📥 Download Results CSV</a>'
    return href

# TODO: Replace these functions with your actual ML model
def predict_single(customer_data: Dict) -> Tuple[str, float]:
    """
    Uses the cached H2O AutoML model to predict a single record.
    Returns ('Yes' or 'No', confidence in %).
    """
    model = _init_h2o_and_load_model(MODEL_PATH)
    df = pd.DataFrame([customer_data])
    results = _predict_h2o(df, model)
    return results.loc[0, "prediction"], float(results.loc[0, "confidence"])

def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses the cached H2O AutoML model to predict a batch of records.
    Returns the original df with 'prediction' and 'confidence' columns.
    """
    model = _init_h2o_and_load_model(MODEL_PATH)
    results = _predict_h2o(df, model)
    return results

# Main Application
def main():
    load_css()
    render_header()
    
    # Sidebar Navigation
    st.sidebar.markdown("## 🎛️ Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["📊 CSV Batch Analysis", "👤 Individual Prediction"],
        index=0
    )
    
    if page == "📊 CSV Batch Analysis":
        csv_analysis_page()
    else:
        individual_prediction_page()

def csv_analysis_page():
    st.markdown('<div class="card-header">📊 CSV Batch Analysis</div>', unsafe_allow_html=True)
    
    # File Upload Section
    st.markdown("### Upload Customer Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file containing customer data",
        type=['csv'],
        help="Upload a CSV file with customer information for batch prediction analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Display uploaded data info
            st.success(f"✅ Successfully loaded {len(df)} customer records")
            
            # Show data preview
            with st.expander("📋 Data Preview", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Predict button
            if st.button("🎯 Generate Predictions", type="primary"):
                with st.spinner("🔄 Analyzing customer data..."):
                    # TODO: Replace with your actual batch prediction
                    predictions_df = predict_batch(df)
                
                # Calculate metrics
                total_customers = len(predictions_df)
                will_subscribe = len(predictions_df[predictions_df['prediction'] == 'Yes'])
                wont_subscribe = total_customers - will_subscribe
                
                # Display metrics
                st.markdown("### 📈 Prediction Summary")
                render_metrics(total_customers, will_subscribe, wont_subscribe)
                
                # Display pie chart
                st.markdown("### 📊 Results Visualization")
                fig = create_pie_chart(predictions_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display results table
                st.markdown("### 📋 Detailed Results")
                
                # Add download button
                csv_download = get_csv_download_link(predictions_df, "term_deposit_predictions.csv")
                st.markdown(csv_download, unsafe_allow_html=True)
                
                # Show results table
                st.dataframe(
                    predictions_df[['age', 'job', 'marital', 'education', 'prediction', 'confidence']].head(20),
                    use_container_width=True
                )
                
                if len(predictions_df) > 20:
                    st.info(f"Showing first 20 of {len(predictions_df)} predictions. Download CSV for complete results.")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV file contains the required customer data columns.")

def individual_prediction_page():
    st.markdown('<div class="card-header">👤 Individual Customer Prediction</div>', unsafe_allow_html=True)
    
    # Create form
    with st.form("customer_form"):
        st.markdown("### 📝 Customer Information")
        
        # Personal Information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Personal Details")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            job = st.selectbox("Job", [
                "admin.", "blue-collar", "entrepreneur", "housemaid",
                "management", "retired", "self-employed", "services",
                "student", "technician", "unemployed"
            ])
            marital = st.selectbox("Marital Status", ["divorced", "married", "single"])
            education = st.selectbox("Education", [
                "basic.4y", "basic.6y", "basic.9y", "high.school",
                "illiterate", "professional.course", "university.degree"
            ])
        
        with col2:
            st.markdown("#### Financial Information")
            default = st.selectbox("Has Credit Default?", ["no", "yes", "unknown"])
            housing = st.selectbox("Has Housing Loan?", ["no", "yes", "unknown"])
            loan = st.selectbox("Has Personal Loan?", ["no", "yes", "unknown"])
            duration = st.number_input("Last Contact Duration (seconds)", min_value=0, value=200)
        
        # Submit button
        submitted = st.form_submit_button("🎯 Get Prediction", type="primary")
        
        if submitted:
            # Prepare customer data
            customer_data = {
                'age': age,
                'job': job,
                'marital': marital,
                'education': education,
                'default': default,
                'housing': housing,
                'loan': loan,
                'duration': duration
            }
            
            with st.spinner("🔄 Analyzing customer profile..."):
                # TODO: Replace with your actual single prediction
                prediction, confidence = predict_single(customer_data)
            
            # Display result
            st.markdown("### 🎯 Prediction Result")
            render_prediction_result(prediction, confidence)
            
            # Display customer summary
            with st.expander("📊 Customer Profile Summary"):
                summary_df = pd.DataFrame([customer_data])
                st.dataframe(summary_df, use_container_width=True)

if __name__ == "__main__":
    main()