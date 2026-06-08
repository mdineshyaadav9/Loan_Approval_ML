# ═══════════════════════════════════════════════════════════════
# PHASE 6 — Frontend: Loan Approval Prediction App
# ═══════════════════════════════════════════════════════════════
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.graph_objects as go
import plotly.express as px

import sys
import os

# ── Auto-train if model doesn't exist (for cloud deployment) ──
# On Streamlit Cloud, no .pkl files exist when app first starts
# This runs setup.py which trains everything automatically
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_model_path = os.path.join(_base, 'models', 'best_model.pkl')

if not os.path.exists(_model_path):
    sys.path.insert(0, _base)
    from setup import setup
    setup()

st.set_page_config(
    page_title="LoanIQ — AI Credit Decision",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model  = joblib.load(os.path.join(base, 'models/best_model.pkl'))
    scaler = joblib.load(os.path.join(base, 'models/scaler.pkl'))
    with open(os.path.join(base, 'models/model_config.json')) as f:
        config = json.load(f)
    return model, scaler, config

model, scaler, config = load_model()
THRESHOLD     = config.get('best_threshold', 0.5)
FEATURE_NAMES = config['feature_names']

# ── CSS: fix all dark-theme visibility issues ──────────────────
st.markdown("""
<style>
  /* Global background */
  .stApp, [data-testid="stAppViewContainer"], .main { background-color: #F0F2F6 !important; }

  /* Headings */
  h1, h2, h3, h4, h5, h6 { color: #0D1117 !important; }

  /* Body text and labels */
  p, label, li, span { color: #0D1117 !important; }

  /* Widget labels */
  [data-testid="stWidgetLabel"] p,
  .stSelectbox label, .stNumberInput label, .stRadio label { color: #0D1117 !important; font-weight: 500 !important; }

  /* Selectbox container */
  [data-baseweb="select"],
  [data-baseweb="select"] > div,
  [data-baseweb="select"] > div > div {
    background-color: #FFFFFF !important;
    border: 1.5px solid #CED4DA !important;
    border-radius: 8px !important;
  }

  /* Selected value text - the key fix for disappearing text */
  [data-baseweb="select"] div,
  [data-baseweb="select"] span,
  [data-baseweb="select"] input,
  [data-baseweb="select"] p {
    color: #0D1117 !important;
    -webkit-text-fill-color: #0D1117 !important;
  }

  /* Dropdown arrow icon */
  [data-baseweb="select"] svg { fill: #0D1117 !important; }

  /* Dropdown open list */
  [data-baseweb="popover"],
  [data-baseweb="popover"] > div,
  [data-baseweb="menu"] ul {
    background-color: #FFFFFF !important;
  }
  [data-baseweb="popover"] li,
  [data-baseweb="menu"] li,
  [role="option"] {
    background-color: #FFFFFF !important;
    color: #0D1117 !important;
  }
  [role="option"]:hover,
  [data-baseweb="popover"] li:hover { background-color: #E8F0FE !important; }

  /* Number inputs */
  input[type="number"], .stNumberInput input { background-color: #FFFFFF !important; color: #0D1117 !important; border: 1.5px solid #CED4DA !important; border-radius: 8px !important; }

  /* Sidebar */
  section[data-testid="stSidebar"] > div { background-color: #E8EAED !important; }
  section[data-testid="stSidebar"] * { color: #0D1117 !important; }

  /* Radio buttons */
  .stRadio [role="radiogroup"] label p { color: #0D1117 !important; }

  /* Metrics */
  [data-testid="stMetricLabel"] p,
  [data-testid="stMetricValue"],
  [data-testid="stMetricValue"] div { color: #0D1117 !important; }

  /* Fix code backtick black boxes in sidebar */
  code {
    background-color: #E0E4EA !important;
    color: #0D1117 !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
  }

  /* Caption */
  [data-testid="stCaptionContainer"] p { color: #444 !important; }

  /* Cards */
  .approve-card {
    background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
    border-left: 5px solid #2E7D32;
    padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
  }
  .reject-card {
    background: linear-gradient(135deg, #FFEBEE, #FFCDD2);
    border-left: 5px solid #C62828;
    padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
  }
  /* Submit button */
  .stButton > button {
    background: #1565C0 !important; color: #FFFFFF !important;
    border: none; padding: 0.75rem 2rem; border-radius: 8px;
    font-size: 1rem; font-weight: 600; width: 100%;
  }
  .stButton > button:hover { background: #0D47A1 !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 LoanIQ")
    st.markdown("### Navigation")
    page = st.radio("", ["🎯 New Application", "📊 Model Dashboard", "📖 How It Works"])
    st.divider()
    acc = config.get('accuracy', 0)*100
    auc = config.get('roc_auc', 0)
    st.markdown(f"**Model Accuracy:** <span style='color:#2E7D32;font-weight:700'>{acc:.1f}%</span>", unsafe_allow_html=True)
    st.markdown(f"**ROC-AUC Score:** <span style='color:#1565C0;font-weight:700'>{auc:.3f}</span>", unsafe_allow_html=True)
    st.markdown(f"**Decision Threshold:** <span style='color:#E65100;font-weight:700'>{THRESHOLD:.2f}</span>", unsafe_allow_html=True)
    st.markdown("**Algorithm:** Random Forest")


# ── Header ─────────────────────────────────────────────────────
st.title("🏦 LoanIQ — AI-Powered Credit Decision System")
st.caption("Instant loan approval decisions powered by Machine Learning")
st.divider()


# ══════════════════════════════════════════════════════════════
# PAGE 1 — NEW APPLICATION
# ══════════════════════════════════════════════════════════════
if "New Application" in page:

    st.subheader("📋 Loan Application Form")
    st.caption("Fill in the applicant's details below. The AI model will assess credit risk instantly.")

    with st.form("loan_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("👤 Personal Details")
            gender        = st.selectbox("Gender",            ["Male", "Female"])
            married       = st.selectbox("Marital Status",    ["Yes", "No"])
            dependents    = st.selectbox("No. of Dependents", ["0", "1", "2", "3+"])
            education     = st.selectbox("Education",         ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed",     ["No", "Yes"])

        with col2:
            st.subheader("💰 Financial Details")
            applicant_income = st.number_input(
                "Applicant Monthly Income (₹)",
                min_value=0, max_value=200000, value=5000, step=500,
                help="Monthly take-home income of primary applicant"
            )
            coapplicant_income = st.number_input(
                "Co-Applicant Income (₹)",
                min_value=0, max_value=100000, value=0, step=500,
                help="Income of spouse or co-borrower. Enter 0 if none."
            )
            loan_amount = st.number_input(
                "Loan Amount Requested (₹ thousands)",
                min_value=9, max_value=700, value=120, step=10,
                help="Amount in thousands. e.g., 150 = ₹1,50,000"
            )

        with col3:
            st.subheader("🏠 Loan Details")
            loan_term = st.selectbox(
                "Loan Term (days)",
                [360, 180, 480, 300, 120, 84, 60, 36],
                help="360 days = 1 year. Most loans are 360 days."
            )
            credit_history = st.selectbox(
                "Credit History",
                [1.0, 0.0],
                format_func=lambda x: "✅ Good (met all obligations)" if x == 1.0 else "❌ Bad (past defaults)",
                help="Most important factor! Has applicant repaid previous loans?"
            )
            property_area = st.selectbox(
                "Property Area",
                ["Urban", "Semiurban", "Rural"],
                help="Location of property being purchased"
            )

        submitted = st.form_submit_button("🔍 Assess Credit Risk")

    # ── Prediction ────────────────────────────────────────────
    if submitted:
        raw_input = {
            'Gender':                  1 if gender == 'Male' else 0,
            'Married':                 1 if married == 'Yes' else 0,
            'Dependents':              3 if dependents == '3+' else int(dependents),
            'Education':               1 if education == 'Graduate' else 0,
            'Self_Employed':           1 if self_employed == 'Yes' else 0,
            'Loan_Amount_Term':        float(loan_term),
            'Credit_History':          float(credit_history),
            'ApplicantIncome_log':     np.log1p(applicant_income),
            'CoapplicantIncome_log':   np.log1p(coapplicant_income),
            'LoanAmount_log':          np.log1p(loan_amount),
            'Total_Income_log':        np.log1p(applicant_income + coapplicant_income),
            'Property_Area_Semiurban': 1 if property_area == 'Semiurban' else 0,
            'Property_Area_Urban':     1 if property_area == 'Urban' else 0,
        }

        input_df     = pd.DataFrame([raw_input])[FEATURE_NAMES]
        input_scaled = scaler.transform(input_df)
        probability  = model.predict_proba(input_scaled)[0][1]
        # ── Hard business rules — override ML if basic criteria fail ──
        # ML learned from historical data which may have had biases.
        # These rules represent non-negotiable real-world lending limits.
        total_inc     = applicant_income + coapplicant_income
        principal_amt = loan_amount * 1000
        months_n      = max(loan_term / 30, 1)
        r_rate        = 0.085 / 12
        emi_amt       = principal_amt * r_rate * (1+r_rate)**months_n / ((1+r_rate)**months_n - 1)
        emi_pct       = (emi_amt / total_inc * 100) if total_inc >= 1000 else 9999

        rejection_reason = None
        if total_inc < 1000:
            rejection_reason = "⚠️ Override: Total income ₹{:,} is below minimum lending threshold of ₹1,000/month".format(total_inc)
            probability = 0.02
        elif emi_pct > 70:
            rejection_reason = "⚠️ Override: EMI would consume {:.0f}% of income — bank maximum is 70%".format(emi_pct)
            probability = min(probability, 0.25)
        elif loan_amount > (total_inc * 60 / 1000):
            rejection_reason = "⚠️ Override: Loan is {:.0f}× monthly income — exceeds 60× ceiling".format(loan_amount*1000/max(total_inc,1))
            probability = min(probability, 0.25)

        if rejection_reason:
            decision = "REJECTED"
        else:
            decision = "APPROVED" if probability >= THRESHOLD else "REJECTED"

        st.divider()
        st.subheader("🤖 AI Decision Result")

        res_col1, res_col2, res_col3 = st.columns([2, 1, 1])

        with res_col1:
            if decision == "APPROVED":
                st.markdown(f"""
                <div class="approve-card">
                    <h2 style="color:#2E7D32;margin:0">✅ LOAN APPROVED</h2>
                    <p style="font-size:1.1rem;margin:0.5rem 0 0;color:#1B5E20">
                    Approval Probability: <strong>{probability*100:.1f}%</strong>
                    </p>
                </div>""", unsafe_allow_html=True)
            else:
                override_note = f"<p style=\"font-size:0.9rem;margin:0.5rem 0 0;color:#7F0000\"><strong>Reason:</strong> {rejection_reason}</p>" if rejection_reason else ""
                st.markdown(f"""
                <div class="reject-card">
                    <h2 style="color:#C62828;margin:0">❌ LOAN REJECTED</h2>
                    <p style="font-size:1.1rem;margin:0.5rem 0 0;color:#7F0000">
                    Approval Probability: <strong>{probability*100:.1f}%</strong>
                    </p>
                    {override_note}
                </div>""", unsafe_allow_html=True)

        with res_col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={"text": "Approval Score", "font": {"color": "#0D1117"}},
                number={"font": {"color": "#0D1117"}},
                gauge={
                    "axis":  {"range": [0, 100], "tickfont": {"color": "#0D1117"}},
                    "bar":   {"color": "#2E7D32" if decision == "APPROVED" else "#C62828"},
                    "steps": [
                        {"range": [0,  40],  "color": "#FFCDD2"},
                        {"range": [40, 60],  "color": "#FFF9C4"},
                        {"range": [60, 100], "color": "#C8E6C9"}
                    ],
                    "threshold": {"line": {"color": "black", "width": 2}, "value": THRESHOLD * 100}
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(t=30, b=0, l=20, r=20),
                                    paper_bgcolor="#F0F2F6", plot_bgcolor="#F0F2F6",
                                    font=dict(color="#0D1117"))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col3:
            # Reuse values already calculated in business rules above
            st.metric("Approval Probability", f"{probability*100:.1f}%")
            if total_inc < 1000:
                st.metric("EMI-to-Income Ratio", "N/A", delta="Income too low", delta_color="inverse")
            else:
                st.metric("EMI-to-Income Ratio", f"{emi_pct:.1f}%",
                          delta="Low Risk" if emi_pct < 40 else "High Risk",
                          delta_color="normal" if emi_pct < 40 else "inverse")
            st.metric("Total Monthly Income", f"₹{total_inc:,}")

        # ── Key Factors ───────────────────────────────────────
        st.subheader("🔍 Key Factors in This Decision")
        factors = []
        if credit_history == 1.0:
            factors.append(("✅ Good credit history",  "Strong positive — has repaid loans before",           "#E8F5E9", "#4CAF50"))
        else:
            factors.append(("❌ Bad credit history",   "Strongest negative — past defaults recorded",         "#FFEBEE", "#F44336"))

        # Reuse business-rule calculations (total_inc, emi_amt, emi_pct already computed)
        if rejection_reason:
            factors.append(("❌ Business Rule Override",
                            rejection_reason,
                            "#FFEBEE", "#F44336"))
        elif emi_pct < 40:
            factors.append(("✅ Affordable EMI",
                            f"EMI ≈ ₹{emi_amt:,.0f}/month ({emi_pct:.0f}% of income) — within safe limit",
                            "#E8F5E9", "#4CAF50"))
        elif emi_pct < 60:
            factors.append(("⚠️ Moderate EMI burden",
                            f"EMI ≈ ₹{emi_amt:,.0f}/month ({emi_pct:.0f}% of income) — borderline",
                            "#FFF8E1", "#FF9800"))
        else:
            factors.append(("❌ High EMI burden",
                            f"EMI ≈ ₹{emi_amt:,.0f}/month ({emi_pct:.0f}% of income) — exceeds safe limit",
                            "#FFEBEE", "#F44336"))

        if education == "Graduate":
            factors.append(("✅ Graduate education",   "Higher earning stability and career prospects",       "#E8F5E9", "#4CAF50"))
        if property_area == "Semiurban":
            factors.append(("✅ Semiurban property",   "Best loan performance area historically",              "#E8F5E9", "#4CAF50"))
        elif property_area == "Rural":
            factors.append(("⚠️ Rural property",      "Rural collateral harder to liquidate on default",     "#FFF8E1", "#FF9800"))
        if coapplicant_income > 0:
            factors.append(("✅ Co-applicant income",  f"Extra ₹{coapplicant_income:,}/month reduces risk",   "#E8F5E9", "#4CAF50"))

        for title, reason, bg, border in factors:
            st.markdown(f"""
            <div style="background:{bg};border-left:4px solid {border};
                        padding:0.6rem 1rem;border-radius:6px;margin:0.4rem 0">
                <strong style="color:#0D1117">{title}</strong><br>
                <span style="font-size:0.9rem;color:#333">{reason}</span>
            </div>""", unsafe_allow_html=True)

        # ── Feature Importance ────────────────────────────────
        st.subheader("📊 What drove this decision?")
        feat_imp = pd.Series(model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False).head(8)
        fig_imp  = px.bar(
            x=feat_imp.values * 100, y=feat_imp.index, orientation='h',
            color=feat_imp.values, color_continuous_scale='RdYlGn',
            labels={'x': 'Importance (%)', 'y': 'Feature'},
            title='Top 8 Features Influencing This Decision'
        )
        fig_imp.update_layout(
            height=400, showlegend=False, coloraxis_showscale=False,
            paper_bgcolor="#F0F2F6", plot_bgcolor="#F0F2F6",
            font=dict(color="#0D1117", size=13),
            title_font=dict(color="#0D1117", size=14),
            xaxis=dict(color="#0D1117", tickfont=dict(color="#0D1117"), title_font=dict(color="#0D1117"), gridcolor="#DDDDDD"),
            yaxis=dict(color="#0D1117", tickfont=dict(color="#0D1117", size=12), title_font=dict(color="#0D1117")),
            margin=dict(l=200, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_imp, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — MODEL DASHBOARD
# ══════════════════════════════════════════════════════════════
elif "Dashboard" in page:
    st.subheader("📊 Model Performance Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Accuracy",      f"{config.get('accuracy',0)*100:.1f}%", "Random Forest")
    with col2:
        st.metric("ROC-AUC Score",       f"{config.get('roc_auc',0):.3f}",       "+vs baseline")
    with col3:
        st.metric("Decision Threshold",  f"{THRESHOLD:.2f}",                      "Optimised")
    with col4:
        st.metric("Training Algorithm",  "Random Forest",                         "100 trees")

    st.divider()
    st.subheader("🌳 Why Random Forest?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""**Logistic Regression**
- Simple, fast, explainable
- Assumes linear relationships
- Good baseline: ~78% accuracy""")
    with col2:
        st.markdown("""**Decision Tree**
- Visual, rule-based
- Can overfit easily
- ~80% accuracy""")
    with col3:
        st.markdown("""**✅ Random Forest (chosen)**
- 100 trees → stable predictions
- Handles non-linearity
- Best accuracy & AUC
- Industry standard for credit""")

    for plot in ['plots/phase4_models.png', 'plots/phase5_optimization.png']:
        if os.path.exists(plot):
            st.image(plot)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — HOW IT WORKS
# ══════════════════════════════════════════════════════════════
elif "How It Works" in page:
    st.subheader("📖 How the ML System Works")
    st.markdown(f"""
### The Pipeline

**Step 1 — Raw Application**  
Bank officer fills the form with applicant details.

**Step 2 — Preprocessing**  
Inputs transformed the same way training data was prepared:
income → log-transformed · text → numbers · values → scaled (StandardScaler)

**Step 3 — Feature Vector**  
All inputs arranged in the exact same column order the model expects.

**Step 4 — Random Forest Predicts**  
100 decision trees each vote: Approve or Reject. Majority vote + probability returned.

**Step 5 — Threshold Applied**  
If approval probability ≥ {THRESHOLD:.2f} → APPROVED, else REJECTED.

**Step 6 — Explanation**  
Feature importances reveal which factors drove the decision.

---

""")

    st.markdown("---")
    st.subheader("🏦 Real-World Impact")
    st.markdown('''
<table style="width:100%;border-collapse:collapse;margin-top:12px">
  <thead>
    <tr>
      <th style="background:#1565C0;color:#FFFFFF;padding:12px 16px;text-align:left;font-size:15px">Traditional Process</th>
      <th style="background:#2E7D32;color:#FFFFFF;padding:12px 16px;text-align:left;font-size:15px">✅ ML-Powered Process</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:#F8F9FA">
      <td style="padding:10px 16px;color:#0D1117;border-bottom:1px solid #DEE2E6">⏱ 2–5 days processing</td>
      <td style="padding:10px 16px;color:#0D1117;border-bottom:1px solid #DEE2E6">⚡ Under 1 second</td>
    </tr>
    <tr style="background:#FFFFFF">
      <td style="padding:10px 16px;color:#0D1117;border-bottom:1px solid #DEE2E6">⚠️ Human bias possible</td>
      <td style="padding:10px 16px;color:#0D1117;border-bottom:1px solid #DEE2E6">✅ Consistent and objective</td>
    </tr>
    <tr style="background:#F8F9FA">
      <td style="padding:10px 16px;color:#0D1117;border-bottom:1px solid #DEE2E6">👤 1 officer at a time</td>
      <td style="padding:10px 16px;color:#0D1117;border-bottom:1px solid #DEE2E6">🌐 Scales to millions</td>
    </tr>
    <tr style="background:#FFFFFF">
      <td style="padding:10px 16px;color:#0D1117">❓ Hard to explain decisions</td>
      <td style="padding:10px 16px;color:#0D1117">📊 Feature importance per case</td>
    </tr>
  </tbody>
</table>
''', unsafe_allow_html=True)
