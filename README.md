# DGF Prediction Streamlit Application

## Predicting Delayed Graft Function after Kidney Transplantation: A New Look at an Old Problem

This Streamlit application provides an interactive tool for predicting Delayed Graft Function (DGF) after kidney transplantation using a machine learning model trained on data from the University Health Network.

### Features

1. **Interactive Prediction Tool**
   - Manual input form for patient and donor characteristics
   - CSV file upload option with downloadable template
   - Real-time DGF prediction with probability estimates

2. **Study Overview**
   - Complete research abstract
   - Study flow diagram
   - Patient cohort information

3. **Model Performance**
   - ROC curves and confusion matrices
   - Performance metrics for both DGF definitions
   - Comparative analysis

4. **Feature Importance Analysis**
   - SHAP (SHapley Additive exPlanations) plots
   - Feature importance rankings
   - Clinical insights

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

### Input Variables

The model requires the following variables:

**Recipient Variables:**
- Age (years)
- Sex (Male/Female)
- Race (White/Non-White)
- BMI (kg/m²)
- Time on dialysis (years)
- Peak PRA (=0% vs >0%)
- Medical history (diabetes, heart disease, etc.)

**Donor Variables:**
- Age (years)
- Sex (Male/Female)
- BMI (kg/m²)
- Cause of death (stroke/CVA)

**Transplant Variables:**
- Cold ischemic time (hours)
- Expanded criteria donor status
- Death by circulatory criteria donor status
- Re-graft status
- Induction type

### Model Information

- **Algorithm**: Random Forest Classifier
- **Training Data**: 1,840 kidney transplant patients (2000-2022)
- **Performance**: 
  - Dialysis-based DGF: AUC 0.71 (28% sensitivity, 91% specificity)
  - Creatinine-based DGF: AUC 0.64 (70% sensitivity, 49% specificity)

### Research Team

**Michelle Minkovich¹, Sarah De Buono¹, Ghazal Azarfar¹, Muhammad Enrizky Brilian¹, Yanhong Li¹, Jasleen Panesar¹, Olusegun Famure¹, Mamatha Bhat¹, S. Joseph Kim¹**

¹Kidney Transplant Program, Ajmera Transplant Centre, University Health Network

### License

This tool is for research and educational purposes only. Clinical decisions should not be made based solely on model predictions.
