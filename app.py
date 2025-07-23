import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="DGF Prediction Model",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c5f8a;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .author-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f4e79;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Variable labels mapping
VARIABLE_LABELS = {
    'rage': 'Recipient Age (years)',
    'rsex': 'Recipient Sex',
    'rwhite': 'Recipient Race',
    'rbmi': 'Recipient BMI (kg/m²)',
    'tdialyr': 'Time on Dialysis (years)',
    'pkpracat0': 'Peak PRA',
    'rhist_dm': 'History of Diabetes Mellitus',
    'rhist_chd': 'History of Coronary Heart Disease',
    'rhist_chf': 'History of Congestive Heart Failure',
    'rhist_stroke': 'History of Stroke',
    'rhist_pvd': 'History of Peripheral Vascular Disease',
    'rhist_cld': 'History of Chronic Lung Disease',
    'rhist_skin_cancer': 'History of Skin Cancer',
    'rhist_non_skin_cancer': 'History of Non-Skin Cancer',
    'dage': 'Donor Age (years)',
    'dsex': 'Donor Sex',
    'dbmi': 'Donor BMI (kg/m²)',
    'ddeathcause_cva': 'Stroke as Donor Cause of Death',
    'cit': 'Cold Ischemic Time (hours)',
    'ecd': 'Expanded Criteria Donor (ECD)',
    'dcd': 'Death by Circulatory Criteria Donor (DCCD)',
    'regraft': 'Re-graft',
    'inductype': 'Induction Type'
}

# Load the trained model and scaler
@st.cache_resource
def load_model():
    try:
        with open('best_rf_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'best_rf_model.pkl' is in the directory.")
        return None

@st.cache_data
def load_sample_data():
    """Load sample data to fit the scaler"""
    try:
        df = pd.read_csv('DGF_Cleaned.csv')
        return df
    except FileNotFoundError:
        st.error("Sample data file not found.")
        return None

def create_csv_template():
    """Create a CSV template for users to download with multiple sample patients"""
    template_data = {
        'rage': [45.0, 62.0, 38.0],
        'rsex': [1, 0, 1],  # 1=Male, 0=Female
        'rwhite': [1.0, 1.0, 0.0],  # 1=White, 0=Non-White
        'rbmi': [25.0, 28.5, 22.3],
        'tdialyr': [2.0, 4.5, 1.2],
        'pkpracat0': [0.0, 1.0, 0.0],  # 0=0%, 1=>0%
        'rhist_dm': [0.0, 1.0, 0.0],
        'rhist_chd': [0.0, 0.0, 0.0],
        'rhist_chf': [0.0, 1.0, 0.0],
        'rhist_stroke': [0.0, 0.0, 0.0],
        'rhist_pvd': [0.0, 1.0, 0.0],
        'rhist_cld': [0.0, 0.0, 0.0],
        'rhist_skin_cancer': [0.0, 0.0, 0.0],
        'rhist_non_skin_cancer': [0.0, 1.0, 0.0],
        'dage': [50.0, 65.0, 35.0],
        'dsex': [1, 0, 1],  # 1=Male, 0=Female
        'dbmi': [25.0, 30.2, 23.8],
        'ddeathcause_cva': [0.0, 1.0, 0.0],
        'cit': [12.0, 18.5, 8.0],
        'ecd': [0, 1, 0],
        'dcd': [0, 1, 0],
        'regraft': [0, 0, 0],
        'inductype': [0.0, 1.0, 0.0]  # 0=Non-depleting, 1=Depleting
    }
    return pd.DataFrame(template_data)

def preprocess_data(df):
    """Preprocess the input data using the same transformations as the training data"""
    # Load sample data to fit scaler
    sample_df = load_sample_data()
    if sample_df is None:
        return None
    
    # Variables that need normalization
    normalized_vars = ['rage', 'rbmi', 'tdialyr', 'dage', 'dbmi', 'cit']
    
    # Initialize and fit scaler on sample data
    scaler = StandardScaler()
    sample_X = sample_df[normalized_vars]
    scaler.fit(sample_X)
    
    # Apply normalization to input data
    df_processed = df.copy()
    df_processed[normalized_vars] = scaler.transform(df[normalized_vars])
    
    return df_processed

def predict_dgf(data):
    """Make prediction using the trained model for one or multiple patients"""
    model = load_model()
    if model is None:
        return None, None
    
    processed_data = preprocess_data(data)
    if processed_data is None:
        return None, None
    
    # Make predictions for all patients
    predictions = model.predict(processed_data)
    probabilities = model.predict_proba(processed_data)
    
    return predictions, probabilities

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">Predicting Delayed Graft Function after Kidney Transplantation</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666; font-size: 1.2rem;">A New Look at an Old Problem</h2>', unsafe_allow_html=True)
    
    # Author information
    st.markdown("""
    <div class="author-info">
        <h3 style="color: #1f4e79; margin-bottom: 0.5rem;">Research Team</h3>
        <p style="margin-bottom: 0.5rem;"><strong>Michelle Minkovich¹, Sarah De Buono¹, Ghazal Azarfar¹, Muhammad Enrizky Brilian¹, 
        Yanhong Li¹, Jasleen Panesar¹, Olusegun Famure¹, Mamatha Bhat¹, S. Joseph Kim¹</strong></p>
        <p style="margin: 0; font-size: 0.9rem; color: #666;">
        ¹Kidney Transplant Program, Ajmera Transplant Centre, University Health Network<br>
        *These authors contributed equally to this work
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction Tool", "📊 Study Overview", "🧠 Model Performance", "📈 Feature Analysis"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">DGF Prediction Tool</h2>', unsafe_allow_html=True)
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Manual Input", "Upload CSV File"],
            horizontal=True
        )
        
        if input_method == "Manual Input":
            st.markdown("### Patient and Donor Information")
            
            # Option to add multiple patients
            if 'num_patients' not in st.session_state:
                st.session_state.num_patients = 1
            
            col_control1, col_control2, col_control3 = st.columns([1, 1, 2])
            with col_control1:
                if st.button("➕ Add Patient"):
                    st.session_state.num_patients += 1
            with col_control2:
                if st.button("➖ Remove Patient") and st.session_state.num_patients > 1:
                    st.session_state.num_patients -= 1
            with col_control3:
                st.write(f"**Total Patients: {st.session_state.num_patients}**")
            
            # Store all patient data
            all_patients_data = []
            
            # Create input forms for each patient
            for patient_num in range(st.session_state.num_patients):
                if st.session_state.num_patients > 1:
                    st.markdown(f"#### 👤 Patient {patient_num + 1}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Recipient Information")
                    rage = st.number_input(
                        "Age (years)", 
                        min_value=0, max_value=100, value=45, 
                        key=f"rage_{patient_num}"
                    )
                    rsex = st.selectbox(
                        "Sex", 
                        options=[1, 0], 
                        format_func=lambda x: "Male" if x == 1 else "Female",
                        key=f"rsex_{patient_num}"
                    )
                    rwhite = st.selectbox(
                        "Race", 
                        options=[1.0, 0.0], 
                        format_func=lambda x: "White" if x == 1.0 else "Non-White",
                        key=f"rwhite_{patient_num}"
                    )
                    rbmi = st.number_input(
                        "BMI (kg/m²)", 
                        min_value=10.0, max_value=60.0, value=25.0, step=0.1,
                        key=f"rbmi_{patient_num}"
                    )
                    tdialyr = st.number_input(
                        "Time on Dialysis (years)", 
                        min_value=0.0, max_value=20.0, value=2.0, step=0.1,
                        key=f"tdialyr_{patient_num}"
                    )
                    pkpracat0 = st.selectbox(
                        "Peak PRA", 
                        options=[0.0, 1.0], 
                        format_func=lambda x: "=0%" if x == 0.0 else ">0%",
                        key=f"pkpracat0_{patient_num}"
                    )
                    
                    st.markdown("##### Recipient Medical History")
                    rhist_dm = st.selectbox(
                        "Diabetes Mellitus", 
                        options=[0.0, 1.0], 
                        format_func=lambda x: "No" if x == 0.0 else "Yes",
                        key=f"rhist_dm_{patient_num}"
                    )
                    rhist_chd = st.selectbox(
                        "Coronary Heart Disease", 
                        options=[0.0, 1.0], 
                        format_func=lambda x: "No" if x == 0.0 else "Yes",
                        key=f"rhist_chd_{patient_num}"
                    )
                    rhist_chf = st.selectbox(
                        "Congestive Heart Failure", 
                        options=[0.0, 1.0], 
                        format_func=lambda x: "No" if x == 0.0 else "Yes",
                        key=f"rhist_chf_{patient_num}"
                    )
                    rhist_stroke = st.selectbox(
                        "Stroke", 
                        options=[0.0, 1.0], 
                        format_func=lambda x: "No" if x == 0.0 else "Yes",
                        key=f"rhist_stroke_{patient_num}"
                    )
                    rhist_pvd = st.selectbox(
                        "Peripheral Vascular Disease", 
                        options=[0.0, 1.0], 
                        format_func=lambda x: "No" if x == 0.0 else "Yes",
                        key=f"rhist_pvd_{patient_num}"
                    )
                    rhist_cld = st.selectbox(
                        "Chronic Lung Disease", 
                        options=[0.0, 1.0], 
                        format_func=lambda x: "No" if x == 0.0 else "Yes",
                        key=f"rhist_cld_{patient_num}"
                    )
                    rhist_skin_cancer = st.selectbox(
                        "Skin Cancer", 
                        options=[0.0, 1.0], 
                        format_func=lambda x: "No" if x == 0.0 else "Yes",
                        key=f"rhist_skin_cancer_{patient_num}"
                    )
                    rhist_non_skin_cancer = st.selectbox(
                        "Non-Skin Cancer", 
                        options=[0.0, 1.0], 
                        format_func=lambda x: "No" if x == 0.0 else "Yes",
                        key=f"rhist_non_skin_cancer_{patient_num}"
                    )
                
                with col2:
                    st.markdown("##### Donor Information")
                    dage = st.number_input(
                        "Donor Age (years)", 
                        min_value=0, max_value=100, value=50,
                        key=f"dage_{patient_num}"
                    )
                    dsex = st.selectbox(
                        "Donor Sex", 
                        options=[1, 0], 
                        format_func=lambda x: "Male" if x == 1 else "Female",
                        key=f"dsex_{patient_num}"
                    )
                    dbmi = st.number_input(
                        "Donor BMI (kg/m²)", 
                        min_value=10.0, max_value=60.0, value=25.0, step=0.1,
                        key=f"dbmi_{patient_num}"
                    )
                    ddeathcause_cva = st.selectbox(
                        "Stroke as Cause of Death", 
                        options=[0.0, 1.0], 
                        format_func=lambda x: "No" if x == 0.0 else "Yes",
                        key=f"ddeathcause_cva_{patient_num}"
                    )
                    
                    st.markdown("##### Transplant Information")
                    cit = st.number_input(
                        "Cold Ischemic Time (hours)", 
                        min_value=0.0, max_value=48.0, value=12.0, step=0.1,
                        key=f"cit_{patient_num}"
                    )
                    ecd = st.selectbox(
                        "Expanded Criteria Donor", 
                        options=[0, 1], 
                        format_func=lambda x: "No" if x == 0 else "Yes",
                        key=f"ecd_{patient_num}"
                    )
                    dcd = st.selectbox(
                        "Death by Circulatory Criteria", 
                        options=[0, 1], 
                        format_func=lambda x: "No" if x == 0 else "Yes",
                        key=f"dcd_{patient_num}"
                    )
                    regraft = st.selectbox(
                        "Re-graft", 
                        options=[0, 1], 
                        format_func=lambda x: "No" if x == 0 else "Yes",
                        key=f"regraft_{patient_num}"
                    )
                    inductype = st.selectbox(
                        "Induction Type", 
                        options=[0.0, 1.0], 
                        format_func=lambda x: "Non-depleting agent" if x == 0.0 else "Depleting agent",
                        key=f"inductype_{patient_num}"
                    )
                
                # Store this patient's data
                patient_data = {
                    'rage': rage, 'rsex': rsex, 'rwhite': rwhite, 'rbmi': rbmi,
                    'tdialyr': tdialyr, 'pkpracat0': pkpracat0, 'rhist_dm': rhist_dm,
                    'rhist_chd': rhist_chd, 'rhist_chf': rhist_chf, 'rhist_stroke': rhist_stroke,
                    'rhist_pvd': rhist_pvd, 'rhist_cld': rhist_cld, 'rhist_skin_cancer': rhist_skin_cancer,
                    'rhist_non_skin_cancer': rhist_non_skin_cancer, 'dage': dage, 'dsex': dsex,
                    'dbmi': dbmi, 'ddeathcause_cva': ddeathcause_cva, 'cit': cit,
                    'ecd': ecd, 'dcd': dcd, 'regraft': regraft, 'inductype': inductype
                }
                all_patients_data.append(patient_data)
                
                # Add separator between patients
                if patient_num < st.session_state.num_patients - 1:
                    st.markdown("---")
            
            # Create dataframe from all patients' inputs
            input_data = pd.DataFrame(all_patients_data)
        
        else:  # CSV Upload
            st.markdown("### Upload CSV File")
            st.info("💡 **Tip**: You can upload data for multiple patients at once. Each row in your CSV file represents one patient.")
            
            # Download template
            template_df = create_csv_template()
            csv_buffer = io.StringIO()
            template_df.to_csv(csv_buffer, index=False)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.download_button(
                    label="📥 Download CSV Template",
                    data=csv_buffer.getvalue(),
                    file_name="dgf_prediction_template.csv",
                    mime="text/csv"
                )
            with col2:
                st.markdown("**Template includes 3 sample patients. You can modify, add, or remove rows as needed.**")
            
            uploaded_file = st.file_uploader("Choose CSV file", type="csv")
            
            if uploaded_file is not None:
                input_data = pd.read_csv(uploaded_file)
                num_patients = len(input_data)
                st.success(f"📊 Successfully loaded data for {num_patients} patient{'s' if num_patients > 1 else ''}!")
                
                # Show data preview
                with st.expander(f"📋 Preview uploaded data ({num_patients} patient{'s' if num_patients > 1 else ''})", expanded=True):
                    st.dataframe(input_data, use_container_width=True)
                
                # Data validation
                required_columns = list(template_df.columns)
                missing_columns = [col for col in required_columns if col not in input_data.columns]
                
                if missing_columns:
                    st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                    input_data = None
                else:
                    st.success("✅ All required columns are present!")
            else:
                input_data = None
        
        # Make prediction
        if st.button("🔮 Predict DGF", type="primary"):
            if input_data is not None:
                with st.spinner("Making predictions..."):
                    predictions, probabilities = predict_dgf(input_data)
                
                if predictions is not None:
                    num_patients = len(predictions)
                    
                    # Success message
                    st.success(f"✅ Successfully predicted DGF for {num_patients} patient{'s' if num_patients > 1 else ''}!")
                    
                    # Display results for each patient
                    for i in range(num_patients):
                        prediction = predictions[i]
                        probability = probabilities[i]
                        
                        st.markdown(f'<div class="prediction-box">', unsafe_allow_html=True)
                        
                        if num_patients > 1:
                            st.markdown(f"### Patient {i+1}")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric(
                                label="DGF Prediction",
                                value="DGF" if prediction == 1 else "No DGF",
                                delta=None
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric(
                                label="DGF Probability",
                                value=f"{probability[1]:.1%}",
                                delta=None
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric(
                                label="No DGF Probability",
                                value=f"{probability[0]:.1%}",
                                delta=None
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Probability visualization for this patient
                        fig = go.Figure(data=[
                            go.Bar(x=['No DGF', 'DGF'], 
                                   y=[probability[0], probability[1]],
                                   marker_color=['#2E8B57', '#DC143C'])
                        ])
                        fig.update_layout(
                            title=f"Prediction Probabilities{' - Patient ' + str(i+1) if num_patients > 1 else ''}",
                            xaxis_title="Outcome",
                            yaxis_title="Probability",
                            yaxis=dict(range=[0, 1]),
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Summary results table for multiple patients
                    if num_patients > 1:
                        st.markdown("### Summary Results")
                        summary_data = []
                        for i in range(num_patients):
                            summary_data.append({
                                'Patient': i+1,
                                'DGF Prediction': 'DGF' if predictions[i] == 1 else 'No DGF',
                                'DGF Probability': f"{probabilities[i][1]:.1%}",
                                'No DGF Probability': f"{probabilities[i][0]:.1%}",
                                'Risk Level': 'High' if probabilities[i][1] > 0.7 else 'Medium' if probabilities[i][1] > 0.3 else 'Low'
                            })
                        
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True)
                        
                        # Download results
                        csv_buffer = io.StringIO()
                        summary_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv_buffer.getvalue(),
                            file_name=f"dgf_predictions_{num_patients}_patients.csv",
                            mime="text/csv"
                        )
                
                else:
                    st.error("❌ Error making predictions. Please check your input data.")
            else:
                st.warning("⚠️ Please provide input data for prediction.")
    
    with tab2:
        st.markdown('<h2 class="sub-header">Study Overview</h2>', unsafe_allow_html=True)
        
        # Abstract
        st.markdown("### Abstract")
        st.markdown("""
        **BACKGROUND:** Delayed graft function (DGF) is a form of acute kidney injury that occurs in one-third of patients at kidney transplant. 
        DGF adversely affects graft and patient outcomes. The reliable prediction of DGF could help clinicians plan interventions to reduce its occurrence, 
        shorten its duration, and/or improve its prognosis.

        **METHODS:** We performed a cohort study of deceased donor kidney transplants at the University Health Network from 1-Jan-2000 to 31-May-2022. 
        DGF was defined as: (i) the need for dialysis within the first week after transplant (dialysis-based definition) and (ii) a proportional reduction 
        in serum creatinine between post-operative days 1 and 2 of less than 30% (creatinine-based definition). A random forest model was used to generate 
        estimates of discrimination, accuracy, and F1 scores. Shapley values were used to determine the relative contribution of the included features.

        **RESULTS:** A total of 1,840 patients were included in the analytic cohort. DGF occurred in 35% and 55% of patients using the dialysis- and 
        creatinine-based definitions, respectively. The random forest model yielded AUC estimates of 0.71 (sensitivity: 28%, specificity: 91%) for 
        predicting the dialysis-based definition of DGF, and 0.64 (sensitivity: 70%, specificity: 49%) for the creatinine-based definition of DGF. 
        Donation after circulatory death, recipient BMI, and donor age were the features that displayed the highest Shapley values.

        **CONCLUSIONS:** A machine learning model can reasonably predict DGF, with better performance for dialysis-based rather than creatinine-based definitions. 
        Random forest models have superior predictive ability than other algorithms for both definitions, with better specificity for dialysis-based DGF and 
        better sensitivity for creatinine-based DGF. Further research is needed to improve predictability of creatinine-based DGF.
        """)
        
        # Study Flow Diagram
        st.markdown("### Study Flow Diagram")
        try:
            study_flow_img = Image.open("Figure 1 Study flow diagram.jpg")
            st.image(study_flow_img, caption="Study Flow Diagram", use_container_width=True)
        except FileNotFoundError:
            st.error("Study flow diagram not found.")
    
    with tab3:
        st.markdown('<h2 class="sub-header">Model Performance</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### DGF Dialysis-based Definition")
            try:
                dgf_img = Image.open("dgf_rf.png")
                st.image(dgf_img, caption="ROC Curve and Confusion Matrix - DGF Dialysis", use_container_width=True)
            except FileNotFoundError:
                st.error("DGF dialysis image not found.")
        
        with col2:
            st.markdown("#### DGF Creatinine-based Definition")
            try:
                crr_img = Image.open("crr_rf.png")
                st.image(crr_img, caption="ROC Curve and Confusion Matrix - DGF Creatinine", use_container_width=True)
            except FileNotFoundError:
                st.error("DGF creatinine image not found.")
        
        # Performance Summary
        st.markdown("### Model Performance Summary")
        
        performance_data = {
            'Definition': ['Dialysis-based', 'Creatinine-based'],
            'AUC': [0.71, 0.64],
            'Sensitivity': ['28%', '70%'],
            'Specificity': ['91%', '49%'],
            'Interpretation': [
                'High specificity - good at ruling in DGF when predicted positive',
                'High sensitivity - good at detecting DGF cases'
            ]
        }
        
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True)
    
    with tab4:
        st.markdown('<h2 class="sub-header">Feature Importance Analysis</h2>', unsafe_allow_html=True)
        
        st.markdown("### SHAP Feature Importance")
        st.markdown("SHAP (SHapley Additive exPlanations) values show the contribution of each feature to the model's predictions.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### DGF Dialysis-based Definition")
            try:
                shap_dialysis_img = Image.open("shap_dgf_dialysis.png")
                st.image(shap_dialysis_img, caption="SHAP Feature Importance - DGF Dialysis", use_container_width=True)
            except FileNotFoundError:
                st.error("SHAP dialysis image not found.")
        
        with col2:
            st.markdown("#### DGF Creatinine-based Definition")
            try:
                shap_creatinine_img = Image.open("shap_dgf_creatinine.png")
                st.image(shap_creatinine_img, caption="SHAP Feature Importance - DGF Creatinine", use_container_width=True)
            except FileNotFoundError:
                st.error("SHAP creatinine image not found.")
        
        # Key Findings
        st.markdown("### Key Findings from Feature Analysis")
        
        findings_data = {
            'Feature': [
                'Death by circulatory criteria donor (DCD)',
                'Recipient BMI',
                'Donor age',
                'Cold ischemic time',
                'Time on dialysis'
            ],
            'Importance': ['Very High', 'High', 'High', 'Moderate', 'Moderate'],
            'Impact on DGF Risk': [
                'Increases risk significantly',
                'Higher BMI increases risk',
                'Older donors increase risk',
                'Longer time increases risk',
                'Longer dialysis increases risk'
            ]
        }
        
        findings_df = pd.DataFrame(findings_data)
        st.dataframe(findings_df, use_container_width=True)

if __name__ == "__main__":
    main()
