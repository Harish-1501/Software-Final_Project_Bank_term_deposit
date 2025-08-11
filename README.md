**StreamLit Deployment Link :**
https://harish-1501-software-final-project-bank-term-deposit-app-0kfkov.streamlit.app/

Note : If Deployment in local machine , install the Java JDK in local environment.
PS : Testing is done using 1% of the actual Test data, code for data split is available in resource file.

# Dataset Details
* *Dataset Size:* Approximately 45,000 customer records, providing a robust sample size for training machine learning models.
* *Features:* Mix of categorical and numerical variables including age, job type, marital status, account balance, contact communication type, campaign duration, and others.
* *Target Variable:* Binary classification — whether the customer subscribed to a term deposit (yes or no).
* *Source:* Publicly available from the UCI Machine Learning Repository and Kaggle.
Dataset Link : https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset

# ML Term Deposit Prediction App

A beautiful Streamlit application for predicting term deposit subscriptions using machine learning.

## Features

- 📊 **CSV Batch Analysis**: Upload CSV files for batch prediction analysis
- 👤 **Individual Prediction**: Fill out a form for single customer predictions
- 📈 **Interactive Visualizations**: Beautiful pie charts and metrics
- 💾 **Export Results**: Download prediction results as CSV
- 🎨 **Modern UI**: Professional design with gradients and animations

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

## Integration with Your ML Model

Replace the placeholder functions in `app.py`:

### Single Prediction
```python
def predict_single(customer_data: Dict) -> Tuple[str, float]:
    # Replace with your ML model logic
    prediction = your_model.predict(customer_data)
    confidence = your_model.predict_proba(customer_data)
    return prediction, confidence
```

### Batch Prediction
```python
def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    # Replace with your batch ML model logic
    predictions = your_model.predict(df)
    confidences = your_model.predict_proba(df)
    
    df_copy = df.copy()
    df_copy['prediction'] = predictions
    df_copy['confidence'] = confidences
    return df_copy
```

## Expected Data Format

Your CSV should contain columns like:
- `age`: Customer age
- `job`: Job category
- `marital`: Marital status
- `education`: Education level
- `default`: Has credit default
- `housing`: Has housing loan
- `loan`: Has personal loan
- `duration`: Last contact duration

## Customization

- Modify the form fields in `individual_prediction_page()`
- Update the color scheme in the CSS section
- Add more visualization types using Plotly
- Customize the metrics and summary cards
