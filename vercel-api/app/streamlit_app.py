"""
Ethiopian Road Traffic Accident Severity Prediction
Streamlit Frontend App
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.graph_objects as go
import plotly.express as px

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ethiopia RTA Severity Predictor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load Model and Metadata ────────────────────────────────────────────────────
MODEL_PATH    = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_rta_model.pkl')
METADATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_metadata.json')

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    with open(METADATA_PATH) as f:
        return json.load(f)

model    = load_model()
metadata = load_metadata()

CLASS_NAMES  = metadata['classes']
CAT_UNIQUE   = metadata['cat_unique']

# ── Severity Colors ────────────────────────────────────────────────────────────
SEVERITY_CONFIG = {
    'Slight Injury':  {'color': '#2ecc71', 'icon': '🟢', 'bg': '#eafaf1'},
    'Serious Injury': {'color': '#f39c12', 'icon': '🟡', 'bg': '#fef9e7'},
    'Fatal injury':   {'color': '#e74c3c', 'icon': '🔴', 'bg': '#fdedec'},
}

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 800; color: #1a1a2e; }
    .sub-title  { font-size: 1rem; color: #555; margin-top: -10px; }
    .metric-box {
        background: #f7f9fc; border-radius: 10px;
        padding: 16px; text-align: center; border-left: 4px solid #3498db;
    }
    .metric-val  { font-size: 1.6rem; font-weight: 700; color: #1a1a2e; }
    .metric-lab  { font-size: 0.8rem; color: #777; margin-top: 4px; }
    .result-card {
        border-radius: 12px; padding: 24px;
        text-align: center; margin-top: 10px;
    }
    .result-severity { font-size: 1.8rem; font-weight: 800; }
    .result-conf     { font-size: 1rem; margin-top: 8px; color: #555; }
    .section-header  { font-size: 1.1rem; font-weight: 700;
                       color: #1a1a2e; border-bottom: 2px solid #3498db;
                       padding-bottom: 4px; margin-top: 16px; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🚦 Ethiopian RTA Severity Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Addis Ababa Road Traffic Accident Severity Classification | Stacking Ensemble + Regularization</div>', unsafe_allow_html=True)
st.divider()

# ── Sidebar: Model Stats ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Model Performance")
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-val">{metadata['test_accuracy']*100:.1f}%</div>
        <div class="metric-lab">Test Accuracy</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")
    cols = st.columns(2)
    for label, key in [("F1 Score", "test_f1_weighted"), ("ROC-AUC", "test_roc_auc")]:
        cols[0 if label == "F1 Score" else 1].metric(label, f"{metadata[key]:.4f}")

    st.divider()
    st.markdown("### Model Info")
    st.info(f"""
**Model**: Stacking Ensemble  
**Regularization**: Bagging subsampling + LR L2 meta-learner  
**Class Imbalance**: SMOTE  
**CV**: Stratified 5-Fold  
**Dataset**: Mendeley + Figshare  
**Records**: 25,380
    """)
    st.divider()
    st.markdown("### Severity Guide")
    for sev, cfg in SEVERITY_CONFIG.items():
        st.markdown(f"{cfg['icon']} **{sev}**")


# ── Main Layout ────────────────────────────────────────────────────────────────
col_form, col_result = st.columns([1.2, 0.8], gap="large")

with col_form:
    st.markdown("## Input Accident Details")
    st.markdown("Fill in the accident scenario below to predict its severity.")

    # ── Section 1: Time & Location ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Time and Location</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        time_val = st.selectbox("Time of Day", CAT_UNIQUE['Time'])
        day_val  = st.selectbox("Day of Week", CAT_UNIQUE['Day_of_week'])
    with c2:
        area_val = st.selectbox("Area", CAT_UNIQUE['Area_accident_occured'])
        sub_val  = st.selectbox("Sub-district", CAT_UNIQUE['Sub_district'])

    # ── Section 2: Driver Info ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Driver Information</div>', unsafe_allow_html=True)
    c3, c4, c5 = st.columns(3)
    with c3:
        age_val = st.selectbox("Age Band", CAT_UNIQUE['Age_band_of_driver'])
        sex_val = st.selectbox("Sex", CAT_UNIQUE['Sex_of_driver'])
    with c4:
        edu_val  = st.selectbox("Education", CAT_UNIQUE['Educational_level'])
        exp_val  = st.selectbox("Driving Experience", CAT_UNIQUE['Driving_experience'])
    with c5:
        rel_val  = st.selectbox("Vehicle-Driver Relation", CAT_UNIQUE['Vehicle_driver_relation'])
        veh_val  = st.selectbox("Vehicle Type", CAT_UNIQUE['Vehicle_type'])

    # ── Section 3: Road Conditions ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Road and Environment Conditions</div>', unsafe_allow_html=True)
    c6, c7 = st.columns(2)
    with c6:
        road_surf_val  = st.selectbox("Road Surface Type",       CAT_UNIQUE['Road_surface_type'])
        road_cond_val  = st.selectbox("Road Surface Condition",  CAT_UNIQUE['Road_surface_conditions'])
        light_val      = st.selectbox("Light Conditions",        CAT_UNIQUE['Light_conditions'])
    with c7:
        weather_val    = st.selectbox("Weather Conditions",      CAT_UNIQUE['Weather_conditions'])
        lanes_val      = st.selectbox("Lanes / Medians",         CAT_UNIQUE['Lanes_or_Medians'])
        junction_val   = st.selectbox("Junction Type",           CAT_UNIQUE['Types_of_Junction'])

    st.markdown('<div class="section-header">Accident Details</div>', unsafe_allow_html=True)
    c8, c9, c10 = st.columns(3)
    with c8:
        collision_val   = st.selectbox("Type of Collision",  CAT_UNIQUE['Type_of_collision'])
        cause_val       = st.selectbox("Cause of Accident",  CAT_UNIQUE['Cause_of_accident'])
    with c9:
        ped_val         = st.selectbox("Pedestrian Movement",  CAT_UNIQUE['Pedestrian_movement'])
        veh_mov_val     = st.selectbox("Vehicle Movement",     CAT_UNIQUE['Vehicle_movement'])
    with c10:
        n_veh_val = st.slider("Number of Vehicles Involved", 1, 5, 2)
        n_cas_val = st.slider("Number of Casualties",        1, 8, 1)

    road_align_val  = st.selectbox("Road Alignment",  CAT_UNIQUE['Road_allignment'])
    type_veh_val    = st.selectbox("Vehicle Type (detail)", CAT_UNIQUE['Type_of_vehicle'])

    st.markdown("")
    predict_btn = st.button("Predict Accident Severity", type="primary", use_container_width=True)


# ── Result Panel ───────────────────────────────────────────────────────────────
with col_result:
    st.markdown("## Prediction Result")

    if predict_btn:
        input_data = {
            'Time': time_val,
            'Day_of_week': day_val,
            'Age_band_of_driver': age_val,
            'Sex_of_driver': sex_val,
            'Educational_level': edu_val,
            'Vehicle_driver_relation': rel_val,
            'Driving_experience': exp_val,
            'Lanes_or_Medians': lanes_val,
            'Types_of_Junction': junction_val,
            'Road_surface_type': road_surf_val,
            'Road_surface_conditions': road_cond_val,
            'Light_conditions': light_val,
            'Weather_conditions': weather_val,
            'Type_of_collision': collision_val,
            'Number_of_vehicles_involved': n_veh_val,
            'Number_of_casualties': n_cas_val,
            'Vehicle_type': veh_val,
            'Cause_of_accident': cause_val,
            'Pedestrian_movement': ped_val,
            'Vehicle_movement': veh_mov_val,
            'Type_of_vehicle': type_veh_val,
            'Road_allignment': road_align_val,
            'Area_accident_occured': area_val,
            'Sub_district': sub_val,
        }

        with st.spinner("Predicting..."):
            input_df    = pd.DataFrame([input_data])
            pred_class  = model.predict(input_df)[0]
            pred_proba  = model.predict_proba(input_df)[0]

        pred_label  = CLASS_NAMES[pred_class]
        confidence  = pred_proba[pred_class]
        cfg         = SEVERITY_CONFIG[pred_label]

        # Result card
        st.markdown(f"""
        <div class="result-card" style="background:{cfg['bg']}; border: 2px solid {cfg['color']};">
            <div style="font-size: 2.5rem;">{cfg['icon']}</div>
            <div class="result-severity" style="color:{cfg['color']};">{pred_label}</div>
            <div class="result-conf">Confidence: <b>{confidence*100:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # Probability gauge chart
        fig = go.Figure(go.Bar(
            x=[p * 100 for p in pred_proba],
            y=CLASS_NAMES,
            orientation='h',
            marker_color=[SEVERITY_CONFIG[c]['color'] for c in CLASS_NAMES],
            text=[f"{p*100:.1f}%" for p in pred_proba],
            textposition='outside',
        ))
        fig.update_layout(
            title="Class Probabilities",
            xaxis_title="Probability (%)",
            xaxis=dict(range=[0, 115]),
            height=250,
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
            plot_bgcolor='white',
        )
        st.plotly_chart(fig, use_container_width=True)

        # Risk interpretation
        st.markdown("### Risk Factors Detected")
        risks = []
        if time_val == 'Night (22-6)':
            risks.append("Night driving")
        if light_val in ['Darkness - no lighting', 'Darkness - lights unlit']:
            risks.append("Poor lighting")
        if cause_val in ['Drunk driving', 'Overspeed']:
            risks.append(f"High-risk cause: {cause_val}")
        if road_cond_val != 'Dry':
            risks.append(f"Unsafe road condition: {road_cond_val}")
        if collision_val == 'Rollover':
            risks.append("Rollover collision")
        if exp_val in ['Below 1yr', 'No Licence']:
            risks.append("Inexperienced driver")
        if n_cas_val >= 4:
            risks.append(f"{n_cas_val} casualties involved")
        if weather_val in ['Raining and Windy', 'Snow']:
            risks.append(f"Severe weather: {weather_val}")

        if risks:
            for r in risks:
                st.warning(f"⚠ {r}")
        else:
            st.success("No major risk factors detected for this scenario.")

    else:
        st.info("Fill in the accident details on the left and click Predict.")
        st.markdown("""
        **What this tool does:**
        - Predicts the severity of a road accident
        - Uses a trained Stacking Ensemble model with L1 + L2 regularization
        - Handles class imbalance with SMOTE
        - Trained on 25,380 Addis Ababa accident records

        **Classes:**
        - Slight Injury: Minor accidents
        - Serious Injury: Hospitalizations required
        - Fatal injury: Death involved
        """)
