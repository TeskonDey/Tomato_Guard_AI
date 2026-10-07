"""
TomatoGuard AI: Interactive Leaf Disease Diagnosis & Advisory Dashboard
"""

import os
import sys
import json
import joblib
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# Ensure local imports work cleanly
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.segmentation import segment_leaf, segment_lesions, calculate_severity, create_diagnostic_overlay
from src.features import extract_all_features
from src.advisory import get_disease_advice, DISEASE_KNOWLEDGE_BASE
from src.dataset import CLASS_NAMES, populate_sample_library

# --- Page Configuration ---
st.set_page_config(
    page_title="TomatoGuard - Crop Health & Advisory",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #2E7D32;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model():
    """Loads the trained GA-optimized model package."""
    model_path = PROJECT_ROOT / "models" / "tomato_rf_model.joblib"
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None


def run_pipeline(image_bgr: np.ndarray, model_pkg: dict):
    """
    Executes the complete diagnostic pipeline:
    1. Leaf segmentation
    2. Lesion segmentation & severity calculation
    3. 68-feature extraction
    4. Model prediction using GA-selected features
    5. Agronomic advisory lookup
    """
    segmented_leaf, leaf_mask = segment_leaf(image_bgr)
    lesion_mask = segment_lesions(image_bgr, leaf_mask)
    severity_info = calculate_severity(leaf_mask, lesion_mask)
    diagnostic_overlay = create_diagnostic_overlay(image_bgr, leaf_mask, lesion_mask)

    features, feature_names = extract_all_features(image_bgr, leaf_mask, lesion_mask)

    if model_pkg is not None and "rf_model" in model_pkg:
        selected_idx = model_pkg["selected_indices"]
        rf_model = model_pkg["rf_model"]
        class_names = model_pkg["class_names"]

        feats_sub = features[selected_idx].reshape(1, -1)
        pred_idx = int(rf_model.predict(feats_sub)[0])
        pred_class = class_names[pred_idx]

        if hasattr(rf_model, "predict_proba"):
            probs = rf_model.predict_proba(feats_sub)[0]
            prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
        else:
            prob_dict = {name: (1.0 if name == pred_class else 0.0) for name in class_names}
    else:
        if severity_info["severity_pct"] < 2.5:
            pred_class = "Healthy"
        elif severity_info["severity_pct"] < 12.0:
            pred_class = "Alternaria alternata"
        elif severity_info["severity_pct"] < 22.0:
            pred_class = "Alternaria solani"
        else:
            pred_class = "Botrytis cinerea"
        prob_dict = {name: (0.90 if name == pred_class else 0.025) for name in CLASS_NAMES}

    advice = get_disease_advice(pred_class, severity_info["stage"])

    return {
        "segmented_leaf": segmented_leaf,
        "leaf_mask": leaf_mask,
        "lesion_mask": lesion_mask,
        "diagnostic_overlay": diagnostic_overlay,
        "severity_info": severity_info,
        "features": features,
        "feature_names": feature_names,
        "predicted_class": pred_class,
        "probabilities": prob_dict,
        "advice": advice
    }


# --- Main Dashboard ---
def main():
    st.title("🍅 TomatoGuard AI")
    st.subheader("Tomato Leaf Fungal Disease Detection, Severity Rating & Treatment Advisory")

    model_pkg = load_trained_model()

    # --- Sidebar Controls ---
    st.sidebar.header("Leaf Image Source")
    input_source = st.sidebar.radio(
        "Choose image input method:",
        ["Sample Library", "Upload Image"],
        index=0
    )

    sample_image_path = None
    uploaded_file = None

    if input_source == "Sample Library":
        samples_dir = PROJECT_ROOT / "data" / "samples"
        if not samples_dir.exists():
            populate_sample_library(str(samples_dir), samples_per_class=4)

        selected_category = st.sidebar.selectbox("Disease Category:", CLASS_NAMES, index=1)
        cat_dir = samples_dir / selected_category
        sample_files = list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpg"))

        if sample_files:
            file_names = [f.name for f in sample_files]
            chosen_file = st.sidebar.selectbox("Select Sample Leaf:", file_names)
            sample_image_path = cat_dir / chosen_file
        else:
            st.sidebar.warning("No sample files found.")
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload a tomato leaf photo:",
            type=["jpg", "jpeg", "png"]
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    if model_pkg:
        st.sidebar.success("GA-Optimized Random Forest Ready")
        st.sidebar.caption(f"Active Features: **{len(model_pkg['selected_indices'])} / 68**")
        st.sidebar.caption(f"Test Accuracy: **{model_pkg['benchmarks']['rf_optimized']:.1f}%**")
    else:
        st.sidebar.info("Model package not loaded. Run `python train.py` to create.")

    # --- Load Image ---
    image_bgr = None
    if input_source == "Sample Library" and sample_image_path and sample_image_path.exists():
        image_bgr = cv2.imread(str(sample_image_path))
    elif input_source == "Upload Image" and uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.info("Select a sample from the sidebar or upload a leaf photo to begin diagnosis.")
        return

    # --- Execute Pipeline ---
    with st.spinner("Analyzing leaf, extracting multi-domain features, and evaluating severity..."):
        results = run_pipeline(image_bgr, model_pkg)

    orig_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    seg_rgb = cv2.cvtColor(results["segmented_leaf"], cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(results["diagnostic_overlay"], cv2.COLOR_BGR2RGB)

    # --- Visual Inspection Row ---
    st.markdown("### 🔍 Visual Inspection Stages")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(orig_rgb, caption="1. Input Leaf Image", use_container_width=True)
    with col2:
        st.image(seg_rgb, caption="2. Background Removed (Leaf Surface)", use_container_width=True)
    with col3:
        st.image(overlay_rgb, caption="3. Lesion Mask Overlay (Red) & Boundaries", use_container_width=True)

    st.markdown("---")

    # --- High Level Diagnosis & Severity Metrics ---
    sev = results["severity_info"]
    advice = results["advice"]
    pred_class = results["predicted_class"]

    m1, m2, m3, m4 = st.columns([1.3, 1, 1, 1.2])

    with m1:
        st.markdown(f"**Diagnosed Condition**")
        st.markdown(f"### {advice['common_name']}")
        st.caption(f"Pathogen: *{advice['scientific_name']}*")

    with m2:
        st.markdown(f"**Confidence**")
        top_prob = results["probabilities"].get(pred_class, 1.0) * 100
        st.markdown(f"### {top_prob:.1f}%")
        st.caption("Model Certainty")

    with m3:
        st.markdown(f"**Infection Severity**")
        st.markdown(f"### {sev['severity_pct']}%")
        st.caption(f"{sev['lesion_pixels']:,} lesion / {sev['leaf_pixels']:,} leaf px")

    with m4:
        st.markdown(f"**Agricultural Grade**")
        st.markdown(
            f"<div style='margin-top: 6px;'><span class='badge' style='background-color: {sev['badge_color']}; font-size: 1rem;'>"
            f"{sev['stage']}</span></div>",
            unsafe_allow_html=True
        )
        st.caption(sev["summary"])

    st.progress(min(1.0, sev["severity_pct"] / 100.0))

    st.markdown("---")

    # --- Detailed Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Treatment & Advisory",
        "🔬 Feature Analysis",
        "📊 Model Confidence & Benchmarks",
        "📄 Export Diagnostic Report"
    ])

    # TAB 1: Treatment & Advisory
    with tab1:
        st.subheader(f"Treatment Protocol for {advice['common_name']}")
        st.warning(f"**Urgent Immediate Action ({sev['stage']})**: {advice['urgency_action']}")

        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.markdown("#### 🌿 Biological & Organic Solutions")
            for bio in advice["biological_treatments"]:
                st.markdown(f"- {bio}")

            st.markdown("#### 🧪 Targeted Chemical Fungicides")
            for chem in advice["chemical_treatments"]:
                st.markdown(f"- {chem}")

        with t_col2:
            st.markdown("#### 🛡️ Cultural & Preventive Best Practices")
            for prev in advice["cultural_prevention"]:
                st.markdown(f"- {prev}")

            st.markdown("#### 🔍 Characteristic Symptoms")
            for sym in advice["symptoms"]:
                st.markdown(f"- {sym}")

    # TAB 2: Feature Analysis (Dyanmic Per-Image Metrics)
    with tab2:
        st.subheader("Multi-Domain Feature Extraction (68 Features)")
        st.info("💡 **Dynamic Per-Image View**: The graphs in this tab measure the exact physical, color, and textural properties of the specific leaf loaded above.")

        f_names = results["feature_names"]
        f_vals = results["features"]

        # 1. GLCM normalized metrics (values between 0.0 and 3.0)
        glcm_subset = [
            ("Entropy (Lesion Irregularity)", "texture_entropy"),
            ("Contrast (Sharpness of Spots)", "texture_contrast"),
            ("Homogeneity (Uniformity)", "texture_homogeneity"),
            ("Energy (Texture Order)", "texture_energy"),
            ("Correlation (Pattern Spread)", "texture_correlation"),
            ("Dissimilarity", "texture_dissimilarity")
        ]
        glcm_labels = [item[0] for item in glcm_subset]
        glcm_scores = [round(float(f_vals[f_names.index(item[1])]), 3) for item in glcm_subset]

        glcm_df = pd.DataFrame({
            "GLCM Metric": glcm_labels,
            "Value": glcm_scores
        })

        # 2. Key Color Deviations
        color_subset = [
            ("Red Component (RGB)", "RGB_R_mean"),
            ("Green Component (RGB)", "RGB_G_mean"),
            ("Blue Component (RGB)", "RGB_B_mean"),
            ("Yellow-Blue Chroma (L*a*b* b*)", "LAB_b_mean"),
            ("Red-Green Chroma (L*a*b* a*)", "LAB_a_mean"),
            ("Luminance (YCbCr Y)", "YCbCr_Y_mean")
        ]
        color_labels = [item[0] for item in color_subset]
        color_scores = [round(float(f_vals[f_names.index(item[1])]), 1) for item in color_subset]

        color_df = pd.DataFrame({
            "Color Band": color_labels,
            "Mean Intensity": color_scores
        })

        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown("##### 1. Surface Texture Profile (GLCM Texture Metrics)")
            fig_glcm = px.bar(
                glcm_df,
                x="GLCM Metric",
                y="Value",
                text="Value",
                color="Value",
                color_continuous_scale="Viridis"
            )
            fig_glcm.update_traces(textposition='outside')
            fig_glcm.update_layout(xaxis_tickangle=-35, height=360, yaxis_title="Metric Value")
            st.plotly_chart(fig_glcm, use_container_width=True)
            st.caption("Diseased leaves have higher Entropy and Contrast due to rough, irregular spots.")

        with fc2:
            st.markdown("##### 2. Color Space Analysis")
            fig_col = px.bar(
                color_df,
                x="Color Band",
                y="Mean Intensity",
                text="Mean Intensity",
                color="Mean Intensity",
                color_continuous_scale="Teal"
            )
            fig_col.update_traces(textposition='outside')
            fig_col.update_layout(xaxis_tickangle=-35, height=360, yaxis_title="Intensity (0-255)")
            st.plotly_chart(fig_col, use_container_width=True)
            st.caption("Infected leaves shift toward higher Red and Yellow values (chlorosis & necrosis).")

        # 3. Radar Chart: Leaf Pathology Profile
        st.markdown("##### 3. Diagnostic Radar Signature for This Leaf")
        # Normalize 5 key diagnostic indicators to a 0-100 index for visual comparison
        norm_entropy = min(100.0, float(f_vals[f_names.index("texture_entropy")]) * 40.0)
        norm_contrast = min(100.0, float(f_vals[f_names.index("texture_contrast")]) * 100.0)
        norm_severity = float(sev["severity_pct"])
        norm_red_shift = min(100.0, float(f_vals[f_names.index("RGB_R_mean")]) * 0.9)
        norm_yellow_chroma = min(100.0, max(0.0, (float(f_vals[f_names.index("LAB_b_mean")]) - 120.0) * 1.5))

        radar_categories = ["Surface Entropy", "Lesion Contrast", "Infection Severity %", "Red Necrosis Shift", "Yellow Chlorosis"]
        radar_values = [norm_entropy, norm_contrast, norm_severity, norm_red_shift, norm_yellow_chroma]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_values + [radar_values[0]],
            theta=radar_categories + [radar_categories[0]],
            fill='toself',
            fillcolor='rgba(46, 125, 50, 0.25)',
            line=dict(color='#2E7D32', width=2),
            name="Current Leaf"
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            height=380,
            margin=dict(l=40, r=40, t=30, b=30)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # TAB 3: Model Confidence & Benchmarks
    with tab3:
        st.subheader("Classification Probabilities & GA Optimization Benchmarks")

        pb_col1, pb_col2 = st.columns(2)

        with pb_col1:
            st.markdown("##### 1. Live Leaf Prediction Probabilities")
            st.caption("Real-time classification confidence for the current image.")
            
            probs_df = pd.DataFrame({
                "Disease": list(results["probabilities"].keys()),
                "Probability": [round(v * 100, 1) for v in results["probabilities"].values()]
            }).sort_values(by="Probability", ascending=True)

            fig_prob = px.bar(
                probs_df,
                x="Probability",
                y="Disease",
                orientation="h",
                text="Probability",
                color="Probability",
                color_continuous_scale="Reds"
            )
            fig_prob.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_prob.update_layout(height=340, xaxis_title="Confidence (%)", yaxis_title="", xaxis_range=[0, 115])
            st.plotly_chart(fig_prob, use_container_width=True)

        with pb_col2:
            st.markdown("##### 2. Overall Model Scorecard (Fixed Test Set)")
            st.caption("Accuracy comparison on test set before vs. after Genetic Algorithm optimization.")
            if model_pkg and "benchmarks" in model_pkg:
                bm = model_pkg["benchmarks"]
                bench_df = pd.DataFrame({
                    "Model": ["Random Forest", "Random Forest", "SVM", "SVM"],
                    "Configuration": ["Baseline (All 68)", "GA-Optimized", "Baseline (All 68)", "GA-Optimized"],
                    "Accuracy": [bm["rf_baseline"], bm["rf_optimized"], bm["svm_baseline"], bm["svm_optimized"]]
                })
                fig_bench = px.bar(
                    bench_df,
                    x="Model",
                    y="Accuracy",
                    color="Configuration",
                    barmode="group",
                    text="Accuracy",
                    color_discrete_map={"Baseline (All 68)": "#90CAF9", "GA-Optimized": "#2E7D32"}
                )
                fig_bench.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_bench.update_layout(height=340, yaxis_range=[0, 115], yaxis_title="Accuracy (%)")
                st.plotly_chart(fig_bench, use_container_width=True)
            else:
                st.info("Train the model using `python train.py` to view benchmark metrics.")

        st.markdown("---")
        # GA Convergence History
        if model_pkg and "ga_history" in model_pkg and len(model_pkg["ga_history"]) > 0:
            st.markdown("##### 3. Genetic Algorithm Training Convergence")
            st.caption("Shows how the genetic algorithm discovered more accurate and compact feature combinations over generations during training.")
            hist = model_pkg["ga_history"]
            hist_df = pd.DataFrame({
                "Generation": [h["generation"] for h in hist],
                "Fitness Score": [h["best_fitness"] * 100 for h in hist],
                "Selected Features": [h["num_features"] for h in hist]
            })
            fig_ga = px.line(
                hist_df,
                x="Generation",
                y="Fitness Score",
                markers=True,
                title="Optimization Fitness Progression",
                labels={"Fitness Score": "Fitness (%)"}
            )
            fig_ga.update_layout(height=280)
            st.plotly_chart(fig_ga, use_container_width=True)

    # TAB 4: Export Diagnostic Report
    with tab4:
        st.subheader("Farm Diagnostic Summary")
        report_text = f"""======================================================
TOMATOGUARD AGRICULTURAL DIAGNOSTIC REPORT
======================================================
Diagnosed Condition: {advice['common_name']} ({advice['scientific_name']})
Confidence Level:    {results['probabilities'].get(pred_class, 1.0) * 100:.1f}%
Infection Severity:  {sev['severity_pct']}% - {sev['stage']}
Lesion Surface Area: {sev['lesion_pixels']:,} px / {sev['leaf_pixels']:,} px

IMMEDIATE ACTION:
{advice['urgency_action']}

RECOMMENDED BIOLOGICAL REMEDIES:
{chr(10).join('- ' + b for b in advice['biological_treatments'])}

RECOMMENDED CHEMICAL OPTIONS:
{chr(10).join('- ' + c for c in advice['chemical_treatments'])}

CULTURAL PREVENTION:
{chr(10).join('- ' + p for p in advice['cultural_prevention'])}
======================================================"""

        st.code(report_text, language="text")
        st.download_button(
            label="📥 Download Diagnostic Report (.txt)",
            data=report_text,
            file_name=f"TomatoGuard_Report_{advice['common_name'].replace(' ', '_')}.txt",
            mime="text/plain"
        )


if __name__ == "__main__":
    main()
