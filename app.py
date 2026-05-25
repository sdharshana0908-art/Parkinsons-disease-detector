import streamlit as tf # Using st as standard convention
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Parkinson's Disease Detection", layout="wide", initial_sidebar_state="expanded")

# --- DATA & MODEL INITIALIZATION ---
@st.cache_data
def load_data():
    df = pd.read_csv("Data - Parkinsons.csv")
    return df

@st.cache_resource
def train_model(df):
    # Features and target split based on your dataset structure
    X = df.drop(columns=['name', 'status'])
    y = df['status']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train the optimal XGBoost model from your notebook
    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, X.columns.tolist()

# Load data and train model
try:
    df = load_data()
    model, scaler, feature_cols = train_model(df)
except Exception as e:
    st.error(f"Error loading data or training model: {e}")
    st.stop()

# --- SESSION STATE MANAGEMENT (Mock Database) ---
if 'users' not in st.session_state:
    # Pre-populating with an admin account and one sample user
    st.session_state['users'] = {
        "admin@project.com": {"password": "admin123", "name": "Admin User", "role": "Admin", "history": []},
        "user@test.com": {"password": "user123", "name": "John Doe", "role": "User", "history": [1, 0]}
    }

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['user_email'] = None
    st.session_state['user_role'] = None

# --- AUTHENTICATION FUNCTIONS ---
def login_user(email, password):
    if email in st.session_state['users'] and st.session_state['users'][email]['password'] == password:
        st.session_state['logged_in'] = True
        st.session_state['user_email'] = email
        st.session_state['user_role'] = st.session_state['users'][email]['role']
        return True
    return False

def register_user(name, email, password):
    if email in st.session_state['users']:
        return False
    st.session_state['users'][email] = {"password": password, "name": name, "role": "User", "history": []}
    return True

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")

if st.session_state['logged_in']:
    st.sidebar.write(f"**Welcome, {st.session_state['users'][st.session_state['user_email']]['name']}**")
    st.sidebar.write(f"Role: `{st.session_state['user_role']}`")
    
    # Navigation options based on roles
    pages = ["Home / Intro", "Disease Checking Portal", "Statistical Dashboard", "About Us"]
    if st.session_state['user_role'] == 'Admin':
        pages.append("Admin Control Panel")
        
    page = st.sidebar.radio("Go to", pages)
    
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['user_email'] = None
        st.session_state['user_role'] = None
        st.rerun()
else:
    page = st.sidebar.radio("Authentication", ["Login", "Sign Up"])

# --- PAGES IMPLEMENTATION ---

# 1. LOGIN PAGE
if page == "Login":
    st.title("🔐 User & Admin Login")
    with st.form("login_form"):
        email = st.text_input("Email Address")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_submit_button = st.form_submit_button("Login")
        
        if submitted:
            if login_user(email, password):
                st.success("Login Successful!")
                st.rerun()
            else:
                st.error("Invalid email or password.")

# 2. SIGN UP PAGE
elif page == "Sign Up":
    st.title("📝 Create an Account")
    with st.form("signup_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email Address")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Register")
        
        if submitted:
            if name and email and password:
                if register_user(name, email, password):
                    st.success("Registration successful! Please head over to the Login page.")
                else:
                    st.error("Email already registered.")
            else:
                st.warning("Please fill out all fields.")

# 3. HOME / INTRO PAGE
elif page == "Home / Intro":
    st.title("🧠 Parkinson's Disease Detection & Analysis Portal")
    st.subheader("Presented by Dharshana")
    st.markdown("### This is my machine learning project using Python.")
    
    st.markdown("""
    Welcome to the comprehensive Parkinson's Disease detection application. 
    This system leverages a state-of-the-art Machine Learning model (**XGBoost Classifier**) trained on vocal audio features extracted from biomedical voice recordings.
    
    #### How it works:
    1. **Check Status:** Navigate to the **Disease Checking Portal**, provide the clinical audio metrics from a voice assessment report, and evaluate potential risk factors.
    2. **View Statistics:** Browse through the **Statistical Dashboard** to analyze metrics, execute functions, and visualize graphs across historic diagnostic datasets.
    """)
    
    st.info("💡 **Disclaimer:** This software tool functions as an analytical assessment utility leveraging machine learning patterns. It does not replace professional clinical diagnostic procedures.")

# 4. DISEASE CHECKING PAGE
elif page == "Disease Checking Portal":
    st.title("🩺 Medical Diagnosis & Prediction Engine")
    st.write("Input the vocal frequency metrics derived from the clinical voice diagnostics to perform the predictive check.")
    
    st.markdown("### Input Biomedical Parameters")
    
    # Arranging inputs cleanly into a 3-column structural layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fo = st.number_input("MDVP:Fo(Hz) - Average vocal fundamental frequency", value=119.99)
        fhi = st.number_input("MDVP:Fhi(Hz) - Maximum vocal fundamental frequency", value=157.30)
        flo = st.number_input("MDVP:Flo(Hz) - Minimum vocal fundamental frequency", value=74.99)
        jitter_pct = st.number_input("MDVP:Jitter(%)", value=0.0078)
        jitter_abs = st.number_input("MDVP:Jitter(Abs)", value=0.00007)
        rap = st.number_input("MDVP:RAP", value=0.0037)
        ppq = st.number_input("MDVP:PPQ", value=0.0055)
        ddp = st.number_input("Jitter:DDP", value=0.011)

    with col2:
        shimmer = st.number_input("MDVP:Shimmer", value=0.0437)
        shimmer_db = st.number_input("MDVP:Shimmer(dB)", value=0.426)
        apq3 = st.number_input("Shimmer:APQ3", value=0.0218)
        apq5 = st.number_input("Shimmer:APQ5", value=0.0313)
        apq = st.number_input("MDVP:APQ", value=0.0297)
        dda = st.number_input("Shimmer:DDA", value=0.0654)
        nhr = st.number_input("NHR (Noise-to-Harmonics Ratio)", value=0.0221)

    with col3:
        hnr = st.number_input("HNR (Harmonics-to-Noise Ratio)", value=21.03)
        rpde = st.number_input("RPDE (Recurrence period density entropy)", value=0.4147)
        dfa = st.number_input("DFA (Signal fractal scaling exponent)", value=0.8152)
        spread1 = st.number_input("spread1 (Nonlinear measure of fundamental freq.)", value=-4.813)
        spread2 = st.number_input("spread2 (Nonlinear measure of fundamental freq.)", value=0.266)
        d2 = st.number_input("D2 (Correlation dimension)", value=2.301)
        ppe = st.number_input("PPE (Pitch period entropy)", value=0.284)

    if st.button("Run Diagnostic Analysis", type="primary"):
        # Map inputs array safely matching training layout feature sequence orders
        user_features = np.array([[fo, fhi, flo, jitter_pct, jitter_abs, rap, ppq, ddp, 
                                   shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, 
                                   rpde, dfa, spread1, spread2, d2, ppe]])
        
        # Scale inputs using cached scalar weights 
        scaled_user_features = scaler.transform(user_features)
        
        # Run prediction and fetch probabilities
        prediction = model.predict(scaled_user_features)[0]
        prediction_proba = model.predict_proba(scaled_user_features)[0][1]
        
        # Save results history log back into logged user account instance profile states
        st.session_state['users'][st.session_state['user_email']]['history'].append(int(prediction))
        
        st.subheader("Diagnostic Results Evaluation")
        
        # Display condition classifications based on prediction metrics
        if prediction == 1 or prediction_proba >= 0.5:
            st.error(f"⚠️ **Result Alert:** High indications matching Parkinson's disease metrics were observed (Confidence Score: {prediction_proba*100:.2f}%).")
            st.markdown("""> **Important Notice:** Please visit a qualified neurologist or clinical health practitioner for a thorough medical consultation as soon as possible.""")
        else:
            st.success(f"✅ **Result Alert:** Low indications matching Parkinson's patterns observed (Confidence Score: {(1 - prediction_proba)*100:.2f}% clear).")
            if prediction_proba > 0.35:
                st.warning("⚠️ **Note:** Minimal borderline variances were recorded. If you are experiencing symptoms, please arrange an exploratory check with your physician.")

# 5. STATISTICAL DASHBOARD PAGE
elif page == "Statistical Dashboard":
    st.title("📊 Dataset Exploratory Statistics & Visualization Engine")
    
    st.write("Perform aggregations and interact with plots using the baseline data.")
    
    if st.checkbox("Show Raw Diagnostic Dataset"):
        st.dataframe(df)
        
    st.subheader("Basic Statistical Operations")
    metric_choice = st.selectbox("Select Target Evaluation Column for Statistical Summary", df.select_dtypes(include=[np.number]).columns)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Value", f"{df[metric_choice].mean():.4f}")
    col2.metric("Median Value", f"{df[metric_choice].median():.4f}")
    col3.metric("Standard Deviation", f"{df[metric_choice].std():.4f}")
    col4.metric("Maximum Entry Value", f"{df[metric_choice].max():.4f}")

    st.markdown("---")
    st.subheader("Data Visualizations")
    
    viz_type = st.selectbox("Select Graph Style Component", ["Bar Graph", "Pie Chart", "Line Chart", "Correlation Heatmap"])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if viz_type == "Bar Graph":
        # Comparing average feature groupings split by status condition values
        feature_to_plot = st.selectbox("Select feature to check by Status group", ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Shimmer', 'NHR', 'HNR'])
        sns.barplot(x='status', y=feature_to_plot, data=df, ax=ax, palette="Set2")
        ax.set_title(f"Average {feature_to_plot} grouped by Diagnostic Status (0: Healthy, 1: Parkinson's)")
        st.pyplot(fig)
        
    elif viz_type == "Pie Chart":
        # Split ratios between target labels inside original CSV collections
        counts = df['status'].value_counts()
        ax.pie(counts, labels=['Parkinson Disease (1)', 'Healthy (0)'], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
        ax.set_title("Proportion of Target Status Cases across Baseline Dataset")
        st.pyplot(fig)
        
    elif viz_type == "Line Chart":
        # Trends chart representing sequentially ordered sampling runs across selections
        line_feature = st.selectbox("Select trend vector timeline column value", ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)'])
        ax.plot(df.index[:50], df[line_feature].head(50), marker='o', color='purple', linestyle='-')
        ax.set_title(f"Sequential Line Trend for First 50 Dataset Samples ({line_feature})")
        ax.set_xlabel("Sample Record Index Reference")
        ax.set_ylabel(line_feature)
        st.pyplot(fig)
        
    elif viz_type == "Correlation Heatmap":
        # Correlating targeted metric pairings subset to preserve scaling clarity properties
        subset_cols = ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Shimmer', 'NHR', 'HNR', 'spread1', 'status']
        sns.heatmap(df[subset_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap Matrix Inter-relationships")
        st.pyplot(fig)

# 6. ABOUT US & README PAGE
elif page == "About Us":
    st.title("ℹ️ About This Project")
    st.subheader("Developer Profile")
    st.markdown("""
    * **Developer Name:** Dharshana  
    * **Project Domain:** Machine Learning Application (Healthcare Systems Analytics)  
    * **Core Implementation Language:** Python  
    """)
    
    st.markdown("---")
    st.subheader("📚 Project README.md Documentation")
    
    readme_text = """
    # Parkinson's Disease Prediction Using High-Performance Machine Learning
    
    ## 📌 Project Overview
    This platform acts as an automated interactive diagnostic hub classifying biometric vocal anomalies associated with Parkinson's Disease. Utilizing voice attributes (frequencies, frequency jitters, and amplitude shimmers), the underlying classification system maps correlations to track neurodegenerative patterns.
    
    ## 🛠️ Technological Architecture Components
    - **UI Engine Framework:** Streamlit Dashboard Architecture
    - **Predictive Core Algorithm Model:** XGBoost Classifier Model
    - **Statistical Core Parsers:** Pandas DataFrames, NumPy Arrays
    - **Visual Plot Renderers:** Matplotlib Studio, Seaborn Graphics Pipelines
    
    ## 📊 Dataset Features & Context
    The model parses voice recordings from 195 instances containing 22 specific metric features. Key attributes include:
    - **MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz):** Fundamental vocal tone frequencies.
    - **Jitter & Shimmer Variations:** Fundamental pitch cycle variations and voice amplitude perturbations.
    - **HNR, NHR:** Noise ratios tracking tone degradation changes.
    - **Status (Target label):** 0 indicates a Healthy baseline, while 1 flags active Parkinson's constraints.
    
    ## 🔒 Role Access Control Rules
    - **Standard User Roles:** Run personal health diagnostics assessments, view predictive insights, and use exploratory statistics utilities.
    - **Administrative Roles:** Access complete back-end structural logs, evaluate usage trends, and manage user details.
    """
    st.markdown(readme_text)

# 7. ADMIN CONTROL PANEL
elif page == "Admin Control Panel" and st.session_state['user_role'] == 'Admin':
    st.title("🛠️ Administrative Management Console")
    st.write("Secure administrative area monitoring active system states and registration histories.")
    
    # Structural presentation detailing active platform account database frames
    user_records = []
    for email, details in st.session_state['users'].items():
        user_records.append({
            "Full Name": details['name'],
            "Email Address": email,
            "Access Authorization Role": details['role'],
            "Total Diagnosis Runs Generated": len(details['history']),
            "Last Diagnosis Status Code Result": details['history'][-1] if len(details['history']) > 0 else "No Tests Run"
        })
        
    user_df = pd.DataFrame(user_records)
    
    st.subheader("Registered Active System Accounts Database")
    st.dataframe(user_df, use_container_width=True)
    
    st.subheader("⭐ Administrative System Privileges Functions")
    col_adm1, col_adm2 = st.columns(2)
    
    with col_adm1:
        st.markdown("#### System Actions")
        if st.button("Download System User Log CSV"):
            csv = user_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Confirm & Download File",
                data=csv,
                file_name="System_User_Diagnostic_Logs.csv",
                mime="text/csv",
            )
            
    with col_adm2:
        st.markdown("#### Database Operations")
        target_del = st.selectbox("Select user profile to manage or prune", [u for u in st.session_state['users'].keys() if u != st.session_state['user_email']])
        if st.button("Delete User Account Profile", type="secondary"):
            del st.session_state['users'][target_del]
            st.success(f"Profile context `{target_del}` pruned successfully!")
            st.rerun()
