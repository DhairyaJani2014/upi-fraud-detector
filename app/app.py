import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Neural Network Architecture ──
class FraudDetectorNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

# ── Page Config ──
st.set_page_config(
    page_title="UPI Fraud Detector",
    page_icon="🛡️",
    layout="wide"
)

# ── Custom Font ──
st.markdown("""
    <style>
        * { font-family: 'Times New Roman', Times, serif !important; }
        .big-score { font-size: 60px; font-weight: bold; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# ── Load Models ──
@st.cache_resource
def load_models():
    model_info = joblib.load(r'C:\Users\Dhairy Jani\upi-fraud-detector\models\model_info.pkl')
    model = FraudDetectorNN(model_info['input_size'])
    model.load_state_dict(torch.load(
        r'C:\Users\Dhairy Jani\upi-fraud-detector\models\pytorch_nn.pth',
        weights_only=True
    ))
    model.eval()
    rf = joblib.load(r'C:\Users\Dhairy Jani\upi-fraud-detector\models\random_forest.pkl')
    return model, rf

# ── Load Dataset ──
@st.cache_data
def load_data():
    df = pd.read_csv(r'C:\Users\Dhairy Jani\upi-fraud-detector\data\raw\creditcard.csv')
    return df

# ── Header ──
st.title("🛡️ UPI Fraud Detector")
st.markdown("Real-time fraud detection using Machine Learning + Deep Learning")
st.divider()

# ── Load everything ──
model, rf = load_models()
df = load_data()

# ── Sidebar ──
st.sidebar.title("⚙️ Settings")
test_mode = st.sidebar.radio(
    "Select Test Mode:",
    ["🎲 Random Transaction", "🔴 Known Fraud Transaction", "🟢 Known Legitimate Transaction"]
)

st.sidebar.divider()
st.sidebar.markdown("**About this App:**")
st.sidebar.markdown("This app uses 3 ML models to detect fraud:")
st.sidebar.markdown("- K-Means Clustering")
st.sidebar.markdown("- Random Forest Classifier")
st.sidebar.markdown("- PyTorch Neural Network")

# ── Main Area ──
st.subheader("Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input(
        "💰 Transaction Amount (€)",
        min_value=0.01,
        max_value=30000.0,
        value=100.0,
        step=0.01,
        help="Enter the transaction amount"
    )

with col2:
    st.info("💡 Other features are automatically loaded based on selected test mode")

st.divider()

# ── Detect Button ──
if st.button("🔍 Detect Fraud", type="primary", use_container_width=True):

    # Get transaction based on mode
    if test_mode == "🔴 Known Fraud Transaction":
        sample = df[df['Class'] == 1].sample(1, random_state=42)
        st.warning("⚠️ Testing with a KNOWN FRAUD transaction from dataset")

    elif test_mode == "🟢 Known Legitimate Transaction":
        sample = df[df['Class'] == 0].sample(1, random_state=42)
        st.info("ℹ️ Testing with a KNOWN LEGITIMATE transaction from dataset")

    else:
        sample = df.sample(1, random_state=np.random.randint(0, 1000))
        st.info("ℹ️ Testing with a RANDOM transaction from dataset")

    # Override amount with user input
    sample = sample.copy()
    sample['Amount'] = amount

    # Scale Amount and Time
    scaler = StandardScaler()
    scaled = scaler.fit_transform(sample[['Amount', 'Time']])
    sample['Amount_scaled'] = scaled[:, 0]
    sample['Time_scaled']   = scaled[:, 1]

    # Prepare features
    feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount_scaled', 'Time_scaled']
    features = sample[feature_cols].values.astype(np.float32)

    # ── PyTorch Prediction ──
    input_tensor = torch.tensor(features)
    with torch.no_grad():
        risk_score = model(input_tensor).item() * 100

    # ── Random Forest Prediction ──
    rf_pred = rf.predict(features)[0]
    rf_prob  = rf.predict_proba(features)[0][1] * 100

    # ── Actual Label ──
    actual = "🔴 FRAUD" if sample['Class'].values[0] == 1 else "🟢 LEGITIMATE"

    st.divider()

    # ── Results Section ──
    st.subheader("🎯 Detection Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("💰 Amount", f"€{amount:.2f}")

    with col2:
        st.metric("🤖 PyTorch Risk Score", f"{risk_score:.1f}/100")

    with col3:
        st.metric("🌲 Random Forest", f"{rf_prob:.1f}%")

    with col4:
        st.metric("✅ Actual Label", actual)

    st.divider()

    # ── Risk Display ──
    if risk_score > 70:
        st.error(f"""
        ## ⚠️ HIGH RISK — FRAUDULENT TRANSACTION DETECTED!
        **PyTorch Risk Score: {risk_score:.1f}/100**
        **Random Forest Probability: {rf_prob:.1f}%**
        This transaction shows strong fraud signals. Block immediately!
        """)
    elif risk_score > 30:
        st.warning(f"""
        ## 🟡 MEDIUM RISK — REVIEW REQUIRED
        **PyTorch Risk Score: {risk_score:.1f}/100**
        **Random Forest Probability: {rf_prob:.1f}%**
        This transaction requires manual review.
        """)
    else:
        st.success(f"""
        ## ✅ LOW RISK — LEGITIMATE TRANSACTION
        **PyTorch Risk Score: {risk_score:.1f}/100**
        **Random Forest Probability: {rf_prob:.1f}%**
        This transaction appears safe to process.
        """)

    # ── Transaction Details ──
    with st.expander("📊 View Full Transaction Details"):
        st.dataframe(sample[['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)] + ['Class']])