# ==========================================================
# - PRISM-GP 1.0 (Pareto Resolved Interpretable Screening Model for GSK-3β and PIM-1)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from mordred import Calculator, descriptors
import joblib
from graphviz import Digraph
import plotly.express as px
import plotly.graph_objects as go
import base64
import os
import warnings

from interpret.glassbox import ExplainableBoostingClassifier 

warnings.filterwarnings("ignore")

# ==========================================================
#  Page Configuration and Advanced Glassmorphism CSS
# ==========================================================
st.set_page_config(
    page_title="GSK-3β and PIM-1 Predictor | PRISM-GP 1.0", 
    page_icon="🧬", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize persistent session states
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'diagnostic_df' not in st.session_state:
    st.session_state.diagnostic_df = None
if 'profile_idx' not in st.session_state:
    st.session_state.profile_idx = 0
if 'mol_idx' not in st.session_state:
    st.session_state.mol_idx = 0
if 'smiles_input_value' not in st.session_state:
    st.session_state.smiles_input_value = ""

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0A0E17 !important;
        color: #E2E8F0 !important;
        font-size: 1.05rem;
    }
    
    .stCodeBlock, code, pre, .SMILES-font {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.1rem !important;
        color: #00E5FF !important;
    }
    
    [data-testid="collapsedControl"] {display: none;}
    #MainMenu, header, footer {visibility: hidden;}
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 96%; }

    div[data-testid="stColumn"]:first-child {
        position: -webkit-sticky !important;
        position: sticky !important;
        top: 2rem !important;
        height: fit-content !important;
        align-self: flex-start !important; 
        z-index: 999;
    }

    .floating-nav-panel {
        background: rgba(16, 24, 39, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    }

    div.row-widget.stRadio > div[role="radiogroup"] { gap: 18px; display: flex; flex-direction: column; }
    div.row-widget.stRadio > div[role="radiogroup"] label {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 18px 24px !important;
        border-radius: 12px;
        color: #94A3B8 !important;
        font-weight: 600;
        font-size: 1.15rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    div.row-widget.stRadio > div[role="radiogroup"] label:hover {
        border-color: #00E5FF;
        color: #FFFFFF !important;
        transform: translateX(5px);
        box-shadow: -4px 0px 15px rgba(0, 229, 255, 0.15);
    }
    div.row-widget.stRadio > div[role="radiogroup"] [data-checked="true"] label {
        background: rgba(0, 229, 255, 0.05) !important;
        border: 1px solid #00E5FF !important;
        color: #00E5FF !important;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.2);
        text-shadow: 0 0 10px rgba(0, 229, 255, 0.5);
    }

    .metric-panel {
        background: rgba(16, 24, 39, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 2.5rem 1.5rem;
        border-radius: 16px;
        text-align: center;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .metric-panel:hover {
        transform: translateY(-5px);
        border-color: rgba(255,255,255,0.2);
    }
    .metric-active { border-bottom: 3px solid #00E5FF; box-shadow: 0 10px 30px rgba(0, 229, 255, 0.05); }
    .metric-inactive { border-bottom: 3px solid #FF003C; }
    
    .mp-value { font-size: 4.5rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; line-height: 1.1; }
    .val-cyan { color: #00E5FF; text-shadow: 0 0 15px rgba(0, 229, 255, 0.4); }
    .val-crimson { color: #FF003C; text-shadow: 0 0 15px rgba(255, 0, 60, 0.4); }
    .val-neutral { color: #F8FAFC; }
    .mp-label { font-size: 1.3rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 2px; font-weight: 700; margin-top: 15px; }

    .main-title-bar {
        background: linear-gradient(135deg, rgba(16, 24, 39, 0.8) 0%, rgba(10, 14, 23, 0.9) 100%);
        backdrop-filter: blur(20px);
        padding: 3rem;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 2.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .main-title-bar h1 { font-size: 3.2rem !important; font-weight: 800; color: #FFFFFF; letter-spacing: -1px; margin:0; }
    
    .stButton>button {
        background: transparent;
        color: #00E5FF;
        border: 1px solid #00E5FF;
        padding: 1rem 2rem;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.2rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: inset 0 0 10px rgba(0, 229, 255, 0.05);
    }
    .stButton>button:hover {
        background: rgba(0, 229, 255, 0.1);
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.3), inset 0 0 15px rgba(0, 229, 255, 0.2);
        transform: translateY(-2px);
        color: #FFFFFF;
    }
    
    [data-testid="stDataFrame"] { background-color: transparent !important; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# ⚙️ Algorithmic Infrastructure
# -----------------------------

GSK3B_FEATURES = [
    "NdsN", "SRW05", "NdssC", "NssNH", "NaaNH", "n5Ring", "NsssN", "n5ARing", 
    "nN", "SlogP_VSA8", "nG12FRing", "nHRing", "nHBDon", "n6aHRing", "nBondsD", 
    "SdssC", "naHRing", "nARing", "PEOE_VSA2", "SMR_VSA3", "nAHRing", "SMR_VSA4", 
    "NaaN", "C4SP3", "EState_VSA8"
]

PIM1_FEATURES = [
    "naHRing", "nHRing", "SlogP_VSA8", "nBondsD", "NaaN", "nBase", "n6aHRing", 
    "SlogP_VSA3", "nO", "ATSC3dv", "ATSC2i", "SMR_VSA9", "NdssC", "nN", 
    "GATS3d", "n9FRing", "SMR_VSA4", "n6HRing", "GATS3dv", "NsNH2", "n5aRing", 
    "SMR_VSA3", "NssNH", "PEOE_VSA12", "NaaaC"
]

REQUIRED_FEATURES = list(set(GSK3B_FEATURES + PIM1_FEATURES))

@st.cache_resource
def load_models_and_params():
    gsk_model = joblib.load("WebTool_Assets/model_gsk.joblib")
    pim_model = joblib.load("WebTool_Assets/model_pim.joblib")
    scaler_gsk = joblib.load("WebTool_Assets/scaler_gsk.joblib")
    scaler_pim = joblib.load("WebTool_Assets/scaler_pim.joblib")
    pca_gsk = joblib.load("WebTool_Assets/pca_gsk.joblib")
    pca_pim = joblib.load("WebTool_Assets/pca_pim.joblib")
    d2_cutoff = 5.991 
    return gsk_model, pim_model, scaler_gsk, scaler_pim, pca_gsk, pca_pim, d2_cutoff

def prepare_features(df_desc, feature_list):
    X = df_desc.copy()
    X = X.replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors="coerce")
    for col in feature_list:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_list]
    X = X.fillna(X.mean(numeric_only=True)).fillna(0)
    return X.astype(float)

def calculate_pca_ad(X_scaled, pca_model, cutoff):
    X_pca = pca_model.transform(X_scaled)
    std_pc1 = np.sqrt(pca_model.explained_variance_[0])
    std_pc2 = np.sqrt(pca_model.explained_variance_[1])
    d_squared = (X_pca[:, 0] / std_pc1)**2 + (X_pca[:, 1] / std_pc2)**2
    status = ["Within" if d <= cutoff else "Outside" for d in d_squared]
    return d_squared, status

def evaluate_dual_prediction(lo_gsk, ad_gsk, lo_pim, ad_pim):
    """
    Evaluates both models and categorizes into base classes.
    Creates a 'Plot_Class' to enforce exactly 5 visual legend categories.
    """
    final_pred = []
    ad_status = []
    plot_class = []

    for i in range(len(lo_gsk)):
        gsk_active = lo_gsk[i] >= -3.05
        pim_active = lo_pim[i] >= -2.75
        
        # Base Prediction
        if gsk_active and pim_active:
            base_status = "Dual Active"
        elif gsk_active:
            base_status = "GSK-3β Active"
        elif pim_active:
            base_status = "PIM-1 Active"
        else:
            base_status = "Inactive"
            
        # AD Flagging
        if ad_gsk[i] == "Outside" or ad_pim[i] == "Outside":
            final_pred.append(f"{base_status} [Outside AD]")
            ad_status.append("Unreliable Extrapolation (Outside AD)")
            plot_class.append("Outside AD") # Forces orange color on plot
        else:
            final_pred.append(base_status)
            ad_status.append("Highly Reliable (Within AD)")
            plot_class.append(base_status)

    return final_pred, ad_status, plot_class

def render_mol_to_base64(mol, neon_glow=False, glow_color="0,229,255"):
    if mol is None:
        return ""
    from io import BytesIO
    img = Draw.MolToImage(mol, size=(420, 420))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode()
    glow = f"box-shadow:0 0 30px rgba({glow_color},0.45);" if neon_glow else ""
    return f"""
    <div style="display:flex;justify-content:center;">
        <img src="data:image/png;base64,{img_b64}" style="width:320px; background:white; border-radius:16px; padding:12px; {glow}">
    </div>
    """

# ==========================================================
# 📱 Application Layout
# ==========================================================
col_nav, col_main = st.columns([1.2, 4.5], gap="large")

with col_nav:
    st.markdown("""
        <div class="floating-nav-panel">
            <h2 style='color:#FFFFFF; font-weight:800; margin-top:0; font-size:2rem; letter-spacing:-1px;'>PRISM-GP 1.0</h2>
            <p style='color:#00E5FF; font-family:"JetBrains Mono", monospace; font-size:1rem; margin-bottom:2.5rem;'>PARETO RESOLVED INTERPRETABLE SCREENING MODEL FOR GSK-3β AND PIM-1</p>
    """, unsafe_allow_html=True)
    
    navigation_selection = st.radio(
        "Navigation",
        ["🔬 Input and Screening", "⚙️ Architecture Methodology", "📊 Validation and Performance", "📚 References and Citation"],
        label_visibility="collapsed",
        key="main_nav_radio"
    )
    
    st.markdown("""
            <hr style="border:none; border-top:1px solid rgba(255,255,255,0.1); margin: 2.5rem 0;">
            <p style='font-size:0.9rem; color:#94A3B8; font-weight:700; letter-spacing:1px;'>PREDICTION TARGETS</p>
            <p style='font-size:1.1rem; color:#FFFFFF; font-weight:600;'>GSK-3β <span style="color:#FF003C;">(Human)</span></p>
            <p style='font-size:1.1rem; color:#FFFFFF; font-weight:600;'>PIM-1 <span style="color:#FF003C;">(Human)</span></p>
            <br>
            <p style='font-size:0.9rem; color:#94A3B8; font-weight:700; letter-spacing:1px;'>DEVELOPED BY:</p>
            <p style='font-size:1.1rem; color:#FFFFFF; font-weight:600;'>D. Kumar, A. J. Martin.</p>
            <p style='font-size:1.1rem; color:#00E5FF; font-weight:600;'>© 2026 Manipal Academy of Higher Education (MAHE).</p>
            <p style='font-size:1.1rem; color:#FFFFFF; font-weight:600;'>All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)

with col_main:
    st.markdown("""
        <div class="main-title-bar">
            <div>
                <h1 style='color: white; margin: 0;'>Dual Target Inhibitor Predictor for GSK-3β and PIM-1</h1>
                <p style='color: #94A3B8; margin: 0.8rem 0 0 0; font-size: 1.3rem; font-weight:400;'>Classification Model for Glycogen Synthase Kinase-3β (GSK-3β) and Proviral Integration Site for Moloney Murine Leukemia Virus-1 (PIM-1)</p>
            </div>
            <div style='background: rgba(0, 229, 255, 0.05); padding: 12px 24px; border-radius: 12px; border: 1px solid rgba(0, 229, 255, 0.3); box-shadow: 0 0 20px rgba(0, 229, 255, 0.1);'>
                <span style='color:#00E5FF; font-weight:800; font-size:1.4rem; font-family:"JetBrains Mono", monospace;'>V1.0</span>
                <span style='color:#FFFFFF; margin-left:12px; font-weight:600; font-size:1.1rem;'>Active</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if navigation_selection == "🔬 Input and Screening":
        st.markdown("<h2 style='font-weight:700; margin-bottom:1.5rem;'>Data Input and Verification</h2>", unsafe_allow_html=True)
        col_in1, col_in2 = st.columns([1.2, 1], gap="large")
        
        with col_in1:
            st.markdown("<p style='color:#94A3B8; font-weight:600; margin-bottom:1rem;'>INPUT MOLECULES</p>", unsafe_allow_html=True)
            input_option = st.radio("Configure Input Stream:", ["Raw SMILES Entry", "Batch Dataset (.CSV)"], horizontal=True, label_visibility="collapsed", key="input_stream_radio")
            
            smiles_list, valid_mols = [], []
            
            if input_option == "Batch Dataset (.CSV)":
                uploaded_file = st.file_uploader("Upload CSV file with column heading 'SMILES' in the first column", type=["csv"], key="file_uploader")
                if uploaded_file is not None:
                    df_input = pd.read_csv(uploaded_file)
                    if "SMILES" not in df_input.columns:
                        st.error("Matrix failure: Missing mandatory 'SMILES' header.")
                        st.stop()
                    smiles_list = [s for s in df_input["SMILES"] if isinstance(s, str)]
            else:
                user_smiles = st.text_area("Enter SMILES Sequence (One in Each Line):", height=220, value=st.session_state.smiles_input_value, placeholder="Paste Input SMILES here", key="smiles_textarea")
                st.session_state.smiles_input_value = user_smiles
                smiles_list = [s.strip() for s in user_smiles.split("\n") if s.strip()]

            if smiles_list:
                for s in smiles_list:
                    m = Chem.MolFromSmiles(s)
                    if m is not None: valid_mols.append(m)

            st.write("") 
            execute = st.button("INITIALIZE SCREENING", use_container_width=True, key="exec_btn")

        with col_in2:
            st.markdown("<p style='color:#94A3B8; font-weight:600; margin-bottom:1rem;'>INPUT STRUCTURE</p>", unsafe_allow_html=True)
            
            if valid_mols:
                if st.session_state.mol_idx >= len(valid_mols) or st.session_state.mol_idx < 0:
                    st.session_state.mol_idx = 0
                current_mol = valid_mols[st.session_state.mol_idx]
                st.markdown(render_mol_to_base64(current_mol, neon_glow=True, glow_color="0,229,255"), unsafe_allow_html=True)
                
                st.write("")
                p1, p2, p3 = st.columns([1, 1.5, 1])
                with p1:
                    if st.button("◀ PREV", use_container_width=True, key="prev_input_mol"):
                        st.session_state.mol_idx -= 1
                        st.rerun()
                with p2:
                    st.markdown(f"<div style='text-align:center; padding:0.5rem; color:#00E5FF; font-family:\"JetBrains Mono\"; font-weight:700;'>[ {st.session_state.mol_idx + 1} / {len(valid_mols)} ]</div>", unsafe_allow_html=True)
                with p3:
                    if st.button("NEXT ▶", use_container_width=True, key="next_input_mol"):
                        st.session_state.mol_idx += 1
                        st.rerun()
            else:
                st.markdown("""
                <div style="background: rgba(16,24,39,0.5); border: 1px dashed rgba(255,255,255,0.1); border-radius: 16px; height: 350px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                    <span style="font-size: 3rem; opacity: 0.2;">⬡</span>
                    <p style="color: #64748B; font-family: 'JetBrains Mono'; margin-top: 1rem;">AWAITING_INPUT_SEQUENCE</p>
                </div>
                """, unsafe_allow_html=True)

        if execute and smiles_list:
            st.session_state.profile_idx = 0
            st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.05); margin:3rem 0;'>", unsafe_allow_html=True)

            with st.expander("ANALYZING (View Operations Log)", expanded=False):
                st.write(">> Canonicalizing SMILES...")
                canonical_smiles, mols = [], []
                for smi in smiles_list:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))
                        mols.append(mol)

                st.write(">> Calculating 2D Mordred descriptors...")
                calc = Calculator(descriptors, ignore_3D=True)
                df_desc = calc.pandas(mols)

                st.write(">> Loading EBM Models and PCA AD Matrices...")
                try:
                    (gsk_model, pim_model, scaler_gsk, scaler_pim, pca_gsk, pca_pim, d2_cutoff) = load_models_and_params()
                except Exception as e:
                    st.error("Failed to load WebTool_Assets. Ensure models are placed in the 'WebTool_Assets' directory.")
                    st.stop()

                st.write(">> Step 1: Processing GSK-3β Space...")
                X_gsk = prepare_features(df_desc, list(gsk_model.feature_names_in_))
                X_gsk_scaled = scaler_gsk.transform(X_gsk)
                X_gsk_final = pd.DataFrame(X_gsk_scaled, columns=gsk_model.feature_names_in_)
                
                md_gsk, ad_gsk = calculate_pca_ad(X_gsk_scaled, pca_gsk, d2_cutoff)
                prob_gsk = gsk_model.predict_proba(X_gsk_final)[:, 1]
                lo_gsk = gsk_model.decision_function(X_gsk_final)

                st.write(">> Step 2: Processing PIM-1 Space...")
                X_pim = prepare_features(df_desc, list(pim_model.feature_names_in_))
                X_pim_scaled = scaler_pim.transform(X_pim)
                X_pim_final = pd.DataFrame(X_pim_scaled, columns=pim_model.feature_names_in_)
                
                md_pim, ad_pim = calculate_pca_ad(X_pim_scaled, pca_pim, d2_cutoff)
                prob_pim = pim_model.predict_proba(X_pim_final)[:, 1]
                lo_pim = pim_model.decision_function(X_pim_final)

                st.write(">> Compiling Tiered Log-Odds Protocol...")
                final_prediction, final_ad_status, plot_class = evaluate_dual_prediction(
                    lo_gsk, ad_gsk, lo_pim, ad_pim
                )
                
                st.write(">> Analysis complete.")

            st.session_state.results_df = pd.DataFrame({
                "SMILES": canonical_smiles,
                "Prediction": final_prediction,
                "Plot_Class": plot_class,
                "AD_Status": final_ad_status,
                "GSK_LogOdds": np.round(lo_gsk, 3),
                "PIM_LogOdds": np.round(lo_pim, 3),
                "GSK_Prob": np.round(prob_gsk, 4),
                "PIM_Prob": np.round(prob_pim, 4),
                "GSK_D2": np.round(md_gsk, 3),
                "PIM_D2": np.round(md_pim, 3)
            })

            st.session_state.diagnostic_df = pd.concat([
                st.session_state.results_df.reset_index(drop=True),
                pd.DataFrame(X_gsk_scaled, columns=[f"SCALED_GSK_{c}" for c in gsk_model.feature_names_in_]),
                pd.DataFrame(X_pim_scaled, columns=[f"SCALED_PIM_{c}" for c in pim_model.feature_names_in_])
            ], axis=1)

        if st.session_state.results_df is not None:
            results_df = st.session_state.results_df
            if st.session_state.profile_idx >= len(results_df) or st.session_state.profile_idx < 0:
                st.session_state.profile_idx = 0
            
            st.markdown("<h2 style='font-weight:700; margin: 2rem 0 1.5rem 0;'>PREDICTION RESULTS</h2>", unsafe_allow_html=True)
            
            hit_count = len(results_df[~results_df["Prediction"].str.contains("Inactive")])
            hit_rate = np.round((hit_count / len(results_df)) * 100, 1) if len(results_df) > 0 else 0
            
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.markdown(f'<div class="metric-panel"><div class="mp-value val-neutral">{len(results_df)}</div><div class="mp-label">Total Screened</div></div>', unsafe_allow_html=True)
            with m2: st.markdown(f'<div class="metric-panel metric-active"><div class="mp-value val-cyan">{hit_count}</div><div class="mp-label">Total Hits</div></div>', unsafe_allow_html=True)
            with m3: st.markdown(f'<div class="metric-panel metric-active"><div class="mp-value val-cyan">{hit_rate}%</div><div class="mp-label">Hit Rate</div></div>', unsafe_allow_html=True)

            st.write("")
            v_col1, v_col2 = st.columns([1.5, 1], gap="large")
            
            with v_col1:
                st.markdown("<p style='color:#94A3B8; font-weight:600; text-transform:uppercase;'>Log-Odds Decision Boundary Plot</p>", unsafe_allow_html=True)
                results_df["_Internal_Index"] = results_df.index
                
                # Enforces exactly 5 legend categories
                fig = px.scatter(
                    results_df, x="GSK_LogOdds", y="PIM_LogOdds", color="Plot_Class",
                    color_discrete_map={
                        "Dual Active": "#00E676",       # Green
                        "GSK-3β Active": "#00BFFF",     # Blue
                        "PIM-1 Active": "#B67BFF",      # Violet
                        "Inactive": "#FF003C",          # Red
                        "Outside AD": "#FF8C00"         # Orange
                    },
                    hover_data={"SMILES": True, "Prediction": True, "Plot_Class": False, "_Internal_Index": False},
                    template="plotly_dark", custom_data=["_Internal_Index"]
                )
                
                # Draw the rigorous decision boundaries
                fig.add_hline(y=-2.75, line_dash="solid", line_color="rgba(255,255,255,0.3)", annotation_text="PIM-1 Threshold", annotation_position="bottom right")
                fig.add_vline(x=-3.05, line_dash="solid", line_color="rgba(255,255,255,0.3)", annotation_text="GSK-3β Threshold", annotation_position="top left")
                
                if st.session_state.profile_idx in results_df.index:
                    sel_row = results_df.iloc[st.session_state.profile_idx]
                    fig.add_trace(go.Scatter(
                        x=[sel_row["GSK_LogOdds"]], y=[sel_row["PIM_LogOdds"]], mode="markers",
                        marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(color="#FFFFFF", width=3)),
                        showlegend=False, hoverinfo="skip"
                    ))

                fig.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0), height=420,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="GSK-3β Affinity (Log-Odds Score)", gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(title="PIM-1 Affinity (Log-Odds Score)", gridcolor="rgba(255,255,255,0.05)"),
                    clickmode="event+select",
                    legend_title_text='Classification'
                )
                scatter_selection = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points", key="scatter_actives")
                if scatter_selection and "selection" in scatter_selection:
                    points = scatter_selection["selection"].get("points", [])
                    if points and "customdata" in points[0]:
                        st.session_state.profile_idx = points[0]["customdata"][0]

            with v_col2:
                st.markdown("<p style='color:#94A3B8; font-weight:600; text-align:center; text-transform:uppercase;'>Compound Analysis</p>", unsafe_allow_html=True)
                target_row = results_df.iloc[st.session_state.profile_idx]
                target_smi = target_row["SMILES"]
                target_pred = target_row["Prediction"]
                target_ad = target_row["AD_Status"]
                target_plot_class = target_row["Plot_Class"]
                
                # Match molecular glow color to the 5 strict plot classes
                glow_map = {"Dual Active": "0,230,118", "GSK-3β Active": "0,191,255", "PIM-1 Active": "182,123,255", "Outside AD": "255,140,0", "Inactive": "255,0,60"}
                glow_color = glow_map.get(target_plot_class, "255,0,60")
                
                # Render the molecule first
                st.markdown(render_mol_to_base64(Chem.MolFromSmiles(target_smi), neon_glow=True, glow_color=glow_color), unsafe_allow_html=True)
                
                # Render the consolidated info box below the molecule
                st.markdown(f"""
                    <div style='background:rgba(16,24,39,0.3); border: 1px solid rgba(255,255,255,0.05); padding:16px; border-radius:10px; margin-top:20px;'>
                        <p style='margin:0; font-size:0.85rem; color:#94A3B8;'>SMILES:</p>
                        <p style='margin:0 0 10px 0; font-family:"JetBrains Mono"; font-size:0.85rem; color:#00E5FF; word-break:break-all;'>{target_smi}</p>
                        <p style='margin:0; font-size:0.85rem; color:#94A3B8;'>Status: <span style='color:white; font-weight:800;'>{target_pred.upper()}</span></p>
                        <p style='margin:5px 0 0 0; font-size:0.85rem; color:#94A3B8;'>AD Check: <span style='color:white; font-weight:800;'>{target_ad.upper()}</span></p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                p1, p2, p3 = st.columns([1, 1.5, 1])
                with p1:
                    if st.button("◀ COMP", use_container_width=True, key="prev_prof_mol"):
                        st.session_state.profile_idx -= 1
                        st.rerun()
                with p2:
                    st.markdown(f"<div style='text-align:center; padding:0.5rem; color:#E2E8F0; font-family:\"JetBrains Mono\"; font-weight:700;'>Entry {st.session_state.profile_idx + 1} / {len(results_df)}</div>", unsafe_allow_html=True)
                with p3:
                    if st.button("COMP ▶", use_container_width=True, key="next_prof_mol"):
                        st.session_state.profile_idx += 1
                        st.rerun()

            st.divider()
            st.markdown("<p style='color:#94A3B8; font-weight:600; text-transform:uppercase;'>COMPLETE PREDICTION RESULTS</p>", unsafe_allow_html=True)
            
            display_df = results_df.drop(columns=["_Internal_Index", "Plot_Class"], errors='ignore')
            
            def style_predictions(x):
                if "[Outside AD]" in str(x): return "background-color: rgba(255, 140, 0, 0.1); color: #FF8C00; font-weight: bold;"
                elif "Dual Active" in str(x): return "background-color: rgba(0, 230, 118, 0.1); color: #00E676; font-weight: bold;"
                elif "GSK" in str(x): return "background-color: rgba(0, 191, 255, 0.1); color: #00BFFF; font-weight: bold;"
                elif "PIM" in str(x): return "background-color: rgba(182, 123, 255, 0.1); color: #B67BFF; font-weight: bold;"
                else: return "color: #FF003C;"

            styled_df = display_df.style.map(style_predictions, subset=["Prediction"])
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            btn_c1, btn_c2 = st.columns(2)
            with btn_c1:
                # UTF-8-SIG guarantees Excel will display the Beta symbol (β) perfectly
                csv = display_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("💾 EXPORT STANDARD RESULTS (CSV)", data=csv, file_name="PRISM-GP_Results.csv", mime="text/csv", key="dl_std")
            
            with btn_c2:
                if st.session_state.diagnostic_df is not None:
                    diag_csv = st.session_state.diagnostic_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("⚠️ EXPORT FULL AD DIAGNOSTIC (CSV)", data=diag_csv, file_name="PRISM-GP_AD_Diagnostic_Export.csv", mime="text/csv", key="dl_diag")

    elif navigation_selection == "⚙️ Architecture Methodology":
        st.markdown("<h2 style='font-weight:700; margin-bottom:1.5rem;'>Methodology of PRISM-GP 1.0</h2>", unsafe_allow_html=True)
        
        col_a, col_b = st.columns([1.2, 1], gap="large")
        with col_a:
            st.markdown("""
<div style="background: rgba(0, 229, 255, 0.05); padding: 1.5rem; border-left: 4px solid #00E5FF; border-radius: 8px; margin-bottom: 2rem;">
    <strong style="color:#00E5FF; font-size:1.15rem; letter-spacing:0.5px;">ML Workflow for Dual Target Prediction</strong>
    <p style="text-align: justify; margin-top: 0.5rem; color: #E2E8F0;">
    While working as a transparent "glass box" model, like a prism, PRISM-GP 1.0 uses a parallel and multi-objective Explainable Boosting Machine (EBM) architecture to classify the inhibtiory potential of a molecule against both GSK-3β and PIM-1 kinases. The pipeline accounts for domain shifts, class imbalance, and structural applicability while ensuring full algorithmic interpretability.
    </p>
</div>
""", unsafe_allow_html=True)
            
            st.markdown("<p style='color:#00E5FF; font-family:\"JetBrains Mono\"; font-size:0.95rem;'>1. DATASET CURATION AND CLASS IMBALANCE</p>", unsafe_allow_html=True)
            st.markdown("""
            <div style="text-align: justify; margin-bottom: 1.5rem; color: #94A3B8;">
            Bioactivity data (IC50 & Ki) for GSK-3β and PIM-1 were curated from ChEMBL, PubChem, and BindingDB. Due to heavy class imbalance within the curated dataset, the property matched decoys were computationally screened. This gave a highly imbalanced dataset of approximately 1:50 active to decoy ratio, ensuring the model to learn and identify true actives against a large set of chemically similar non-binders.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<p style='color:#00E5FF; font-family:\"JetBrains Mono\"; font-size:0.95rem;'>2. DESCRIPTOR REDUCTION AND OPTIMIZATION</p>", unsafe_allow_html=True)
            st.markdown("""
            <div style="text-align: justify; margin-bottom: 1.5rem; color: #94A3B8;">
            To eliminate conformational artifacts and 3D geometry biases, the workflow calculates only 2D topological and physicochemical descriptors using Mordred. The initial feature space was reduced using different diemnsionality reduction and feature selection methods. The final models use mathematically important and independent subsets of 25 elite features per target.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<p style='color:#00E5FF; font-family:\"JetBrains Mono\"; font-size:0.95rem;'>3. EXPLAINABLE BOOSTING MACHINES (EBM)</p>", unsafe_allow_html=True)
            st.markdown("""
            <div style="text-align: justify; margin-bottom: 1.5rem; color: #94A3B8;">
            The predictive engine relies on Explainable Boosting Machines (EBMs). As Generalized Additive Models (GAMs), EBMs provide exact structural interpretability. Pairwise interactions were strictly disabled during training to prevent the Hauck-Donner exploding gradient effect and to ensure pure additive feature contributions without noise memorization.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<p style='color:#00E5FF; font-family:\"JetBrains Mono\"; font-size:0.95rem;'>4. APPLICABILITY DOMAIN (AD)</p>", unsafe_allow_html=True)
            st.markdown("""
            <div style="text-align: justify; margin-bottom: 0.5rem; color: #94A3B8;">
            Molecules are mapped into a Principal Component Analysis (PCA) space predefined by the standardized master training dataset. Extrapolations are filtered using a strict Mahalanobis squared distance metric governed by a 95% Chi-Square distribution cutoff with 2 degrees of freedom.
            </div>
            """, unsafe_allow_html=True)
            
            # Formatted LaTeX Equation for Applicability Domain
            st.latex(r"D^2 = \left( \frac{PC_1}{\sigma_{PC_1}} \right)^2 + \left( \frac{PC_2}{\sigma_{PC_2}} \right)^2 \le 5.991")
            st.write("")

            st.markdown("<p style='color:#00E5FF; font-family:\"JetBrains Mono\"; font-size:0.95rem;'>5. TIERED LOG-ODDS LOGIC</p>", unsafe_allow_html=True)
            st.markdown("""
            <div style="text-align: justify; margin-bottom: 1.5rem; color: #94A3B8;">
            To overcome the probability "squashing" effect caused by the high decoy imbalance, PRISM-GP 1.0 classifies hits using the raw Log-Odds Decision Function. Validation testing established exact imbalance corrected boundaries (GSK-3β ≥ -3.05 and PIM-1 ≥ -2.75) to identify true structural affinity, while rejecting random library decoys.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<p style='color:#00E5FF; font-family:\"JetBrains Mono\"; font-size:0.95rem;'>6. MODEL VALIDATION METHODS</p>", unsafe_allow_html=True)
            st.markdown("""
            <div style="text-align: justify; margin-bottom: 1.5rem; color: #94A3B8;">
            Models were subjected to 5-fold and 10-fold cross-validation. Generalization to entirely novel chemotypes was verified using an 80/20 Scaffold Disjoint split with Murcko frameworks. Leave One Out (LOO) cross-validation and 50 iterations of Y-Randomization were done to ensure that the models learned genuine physicochemical rules, rather than dataset noise.
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("""
            <div style="background: rgba(16, 24, 39, 0.6); padding: 2rem 1.5rem; border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.05); box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
                <h3 style="color:#FFFFFF; font-size:1.3rem; font-weight:800; margin-top:0; letter-spacing:1px; text-align:center;">
                    PRISM-GP 1.0 WORKFLOW
                </h3>
                <p style="color:#00E5FF; font-family:'JetBrains Mono'; font-size:0.85rem; text-align:center; margin-bottom:2rem;">
                    [ DUAL TARGET PREDICTING CLASSIFICATION MODEL ]
                </p>
            """, unsafe_allow_html=True)
            
            dot = Digraph("ParallelFlow", engine="dot")
            dot.attr(rankdir="TB", splines="ortho", nodesep="0.6", ranksep="0.7", bgcolor="transparent")
            dot.attr("node", shape="rect", style="rounded,filled", fontsize="12", fontname="Inter", color="white", penwidth="1")
            dot.attr("edge", color="#00E5FF", penwidth="2")
            
            dot.node("In", "SMILES Input", fillcolor="#0A0E17", fontcolor="#00E5FF", color="#00E5FF")
            dot.node("Desc", "Strict 2D Mordred Descriptors", fillcolor="#0A0E17", fontcolor="#FFFFFF", color="rgba(255,255,255,0.2)")
            
            # GSK Path
            dot.node("GSK_AD", "PCA AD: GSK-3β (D² ≤ 5.991)", fillcolor="#0A0E17", fontcolor="#FFB300", color="#FFB300")
            dot.node("GSK_EBM", "GSK-3β EBM (Log-Odds)", fillcolor="#0A0E17", fontcolor="#FFFFFF", color="rgba(255,255,255,0.2)")
            
            # PIM Path
            dot.node("PIM_AD", "PCA AD: PIM-1 (D² ≤ 5.991)", fillcolor="#0A0E17", fontcolor="#FFB300", color="#FFB300")
            dot.node("PIM_EBM", "PIM-1 EBM (Log-Odds)", fillcolor="#0A0E17", fontcolor="#FFFFFF", color="rgba(255,255,255,0.2)")
            
            dot.node("Pareto", "Binary Decision Matrix (Log-Odds)", fillcolor="#0A0E17", fontcolor="#00E5FF", color="#00E5FF")
            
            dot.edge("In", "Desc")
            dot.edge("Desc", "GSK_AD")
            dot.edge("Desc", "PIM_AD")
            
            dot.edge("GSK_AD", "GSK_EBM")
            dot.edge("PIM_AD", "PIM_EBM")
            
            dot.edge("GSK_EBM", "Pareto")
            dot.edge("PIM_EBM", "Pareto")
            
            st.graphviz_chart(dot, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

    elif navigation_selection == "📊 Validation and Performance":
        st.markdown("<h2 style='font-weight:700; margin-bottom:1.5rem;'>Model Validation and Performance Metrics</h2>", unsafe_allow_html=True)
        
        # --- PAINS Stress Test Summary ---
        st.markdown("""
<div style="background: rgba(16, 24, 39, 0.6); padding: 1.5rem 2rem; border-radius: 12px; border-left: 4px solid #00E5FF; border: 1px solid rgba(255, 255, 255, 0.05); margin-bottom: 2.5rem; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
<h3 style="color:#FFFFFF; font-size:1.2rem; font-weight:800; margin-top:0; letter-spacing:0.5px;">PAINS (Pan Assay Interference Compounds) Screening</h3>
<p style="color:#94A3B8; font-size:1rem; text-align:justify; margin-bottom:1.5rem;">To ensure that the pipeline is capable of identifying genuine Structure-Activity Relationships over promiscuous reactivity, PRISM-GP 1.0 was tested against a dataset of <strong>320 known PAINS Molecules</strong>. The Explainable Boosting Machines mathematically rejected majority of these decoy molecules, giving a <strong><0.5% False Discovery Rate</strong> for Dual Actives.</p>
<div style="display: flex; gap: 15px; flex-wrap: wrap;">
<div style="flex: 1; min-width: 140px; background: rgba(255, 0, 60, 0.08); border: 1px solid rgba(255, 0, 60, 0.4); border-radius: 8px; padding: 15px; text-align: center;">
<div style="font-size: 2.2rem; font-weight: 800; color: #FF003C; font-family: 'JetBrains Mono', monospace; line-height: 1;">285</div>
<div style="font-size: 0.85rem; color: #E2E8F0; font-weight: 600; text-transform: uppercase; margin-top: 10px;">Inactive</div>
<div style="font-size: 0.75rem; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">(89.1%)</div>
</div>
<div style="flex: 1; min-width: 140px; background: rgba(0, 191, 255, 0.08); border: 1px solid rgba(0, 191, 255, 0.4); border-radius: 8px; padding: 15px; text-align: center;">
<div style="font-size: 2.2rem; font-weight: 800; color: #00BFFF; font-family: 'JetBrains Mono', monospace; line-height: 1;">24</div>
<div style="font-size: 0.85rem; color: #E2E8F0; font-weight: 600; text-transform: uppercase; margin-top: 10px;">GSK-3β Active</div>
<div style="font-size: 0.75rem; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">(7.5%)</div>
</div>
<div style="flex: 1; min-width: 140px; background: rgba(182, 123, 255, 0.08); border: 1px solid rgba(182, 123, 255, 0.4); border-radius: 8px; padding: 15px; text-align: center;">
<div style="font-size: 2.2rem; font-weight: 800; color: #B67BFF; font-family: 'JetBrains Mono', monospace; line-height: 1;">10</div>
<div style="font-size: 0.85rem; color: #E2E8F0; font-weight: 600; text-transform: uppercase; margin-top: 10px;">PIM-1 Active</div>
<div style="font-size: 0.75rem; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">(3.1%)</div>
</div>
<div style="flex: 1; min-width: 140px; background: rgba(0, 230, 118, 0.08); border: 1px solid rgba(0, 230, 118, 0.4); border-radius: 8px; padding: 15px; text-align: center;">
<div style="font-size: 2.2rem; font-weight: 800; color: #00E676; font-family: 'JetBrains Mono', monospace; line-height: 1;">1</div>
<div style="font-size: 0.85rem; color: #E2E8F0; font-weight: 600; text-transform: uppercase; margin-top: 10px;">Dual Active</div>
<div style="font-size: 0.75rem; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">(0.3%)</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

        tab_gsk, tab_pim = st.tabs(["GSK-3β Model Metrics", "PIM-1 Model Metrics"])
        
        # --- GSK-3β Tab ---
        with tab_gsk:
            st.markdown("<p style='color:#00E5FF; font-family:\"JetBrains Mono\"; font-size:1.1rem; margin-top:1.5rem;'>EXPLAINABLE BOOSTING MACHINE: GSK-3β</p>", unsafe_allow_html=True)
            
            gsk_metrics = pd.DataFrame({
                "Validation Phase": [
                    "Test Set (Full)", 
                    "Test Set (Inside AD)", 
                    "5-Fold CV", 
                    "10-Fold CV", 
                    "Scaffold Disjoint", 
                    "Subset LOO", 
                    "Y-Randomization"
                ],
                "ROC-AUC": ["0.953", "0.936", "0.952", "0.952", "0.944", "0.906", "0.500"],
                "PR-AUC": ["0.859", "0.732", "0.857", "0.860", "0.768", "0.735", "-"],
                "F1-Score": ["0.762", "0.665", "0.765", "0.768", "0.652", "0.622", "-"],
                "MCC": ["0.722", "0.627", "0.726", "0.730", "0.616", "0.584", "0.000"],
                "Specificity": ["0.970", "0.971", "0.972", "0.972", "0.972", "0.974", "-"],
                "Sensitivity": ["0.708", "0.599", "0.709", "0.713", "0.574", "0.512", "-"],
                "Balanced Acc": ["0.839", "0.785", "0.840", "0.842", "0.773", "0.743", "-"]
            })
            
            st.dataframe(gsk_metrics, use_container_width=True, hide_index=True)
            
            st.write("")
            c_col1, c_col2 = st.columns([1, 1], gap="large")
            
            with c_col1:
                st.markdown("<p style='color:#94A3B8; font-weight:600; text-transform:uppercase;'>Optimized Hyperparameters</p>", unsafe_allow_html=True)
                st.code('''{
  "max_bins": 128,
  "learning_rate": 0.0451,
  "max_leaves": 4,
  "min_samples_leaf": 17,
  "interactions": 0
}''', language='json')
            
            with c_col2:
                st.markdown("<p style='color:#94A3B8; font-weight:600; text-transform:uppercase;'>Selected 2D Features (25)</p>", unsafe_allow_html=True)
                st.markdown("""
                <div style='background:rgba(16,24,39,0.3); border: 1px solid rgba(255,255,255,0.05); padding:12px; border-radius:10px; font-family:"JetBrains Mono"; font-size:0.85rem; color:#E2E8F0; line-height:1.6;'>
                NdsN, SRW05, NdssC, NssNH, NaaNH, n5Ring, NsssN, n5ARing, nN, SlogP_VSA8, nG12FRing, nHRing, nHBDon, n6aHRing, nBondsD, SdssC, naHRing, nARing, PEOE_VSA2, SMR_VSA3, nAHRing, SMR_VSA4, NaaN, C4SP3, EState_VSA8
                </div>
                """, unsafe_allow_html=True)

        # --- PIM-1 Tab ---
        with tab_pim:
            st.markdown("<p style='color:#00E5FF; font-family:\"JetBrains Mono\"; font-size:1.1rem; margin-top:1.5rem;'>EXPLAINABLE BOOSTING MACHINE: PIM-1</p>", unsafe_allow_html=True)
            
            pim_metrics = pd.DataFrame({
                "Validation Phase": [
                    "Test Set (Full)", 
                    "Test Set (Inside AD)", 
                    "5-Fold CV", 
                    "10-Fold CV", 
                    "Scaffold Disjoint", 
                    "Subset LOO", 
                    "Y-Randomization"
                ],
                "ROC-AUC": ["0.977", "0.977", "0.977", "0.977", "0.973", "0.913", "0.501"],
                "PR-AUC": ["0.926", "0.924", "0.924", "0.925", "0.900", "0.702", "-"],
                "F1-Score": ["0.848", "0.845", "0.853", "0.853", "0.826", "0.645", "-"],
                "MCC": ["0.818", "0.816", "0.823", "0.824", "0.801", "0.589", "0.000"],
                "Specificity": ["0.975", "0.976", "0.976", "0.976", "0.979", "0.959", "-"],
                "Sensitivity": ["0.823", "0.819", "0.827", "0.828", "0.794", "0.568", "-"],
                "Balanced Acc": ["0.899", "0.898", "0.902", "0.902", "0.887", "0.763", "-"]
            })
            
            st.dataframe(pim_metrics, use_container_width=True, hide_index=True)
            
            st.write("")
            p_col1, p_col2 = st.columns([1, 1], gap="large")
            
            with p_col1:
                st.markdown("<p style='color:#94A3B8; font-weight:600; text-transform:uppercase;'>Optimized Hyperparameters</p>", unsafe_allow_html=True)
                st.code('''{
  "max_bins": 128,
  "learning_rate": 0.0165,
  "max_leaves": 3,
  "min_samples_leaf": 36,
  "interactions": 0
}''', language='json')

            with p_col2:
                st.markdown("<p style='color:#94A3B8; font-weight:600; text-transform:uppercase;'>Selected 2D Features (25)</p>", unsafe_allow_html=True)
                st.markdown("""
                <div style='background:rgba(16,24,39,0.3); border: 1px solid rgba(255,255,255,0.05); padding:12px; border-radius:10px; font-family:"JetBrains Mono"; font-size:0.85rem; color:#E2E8F0; line-height:1.6;'>
                naHRing, nHRing, SlogP_VSA8, nBondsD, NaaN, nBase, n6aHRing, SlogP_VSA3, nO, ATSC3dv, ATSC2i, SMR_VSA9, NdssC, nN, GATS3d, n9FRing, SMR_VSA4, n6HRing, GATS3dv, NsNH2, n5aRing, SMR_VSA3, NssNH, PEOE_VSA12, NaaaC
                </div>
                """, unsafe_allow_html=True)

            
    elif navigation_selection == "📚 References and Citation":
        st.markdown("<h2 style='font-weight:700; margin-bottom:1.5rem;'>References, Citation and Scientific Literature</h2>", unsafe_allow_html=True)
        
        st.markdown("<p style='color:#00E5FF; font-family:\"JetBrains Mono\";'>DEVELOPERS, INSTITUTIONAL AFFILIATION and COPYRIGHT</p>", unsafe_allow_html=True)
        st.markdown("- **© 2026 Manipal Academy of Higher Education (MAHE).** <span style='color:#94A3B8;'>All rights reserved.</span>", unsafe_allow_html=True)
        st.markdown("- **D.Kumar, A. J. Martin** <span style='color:#94A3B8;'>[Manipal Academy of Higher Education - MAHE, Manipal, Karnataka]</span>", unsafe_allow_html=True)
        
        st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.05); margin: 2rem 0;'>", unsafe_allow_html=True)
        
        st.markdown("<p style='color:#00E5FF; font-family:\"JetBrains Mono\";'>How to Cite PRISM-GP 1.0</p>", unsafe_allow_html=True)
        st.markdown("If you use the PRISM-GP 1.0 webtool in research or publications, please cite:")
        st.markdown("""
<div style="background: rgba(0, 229, 255, 0.05); padding: 1.5rem; border-left: 4px solid #00E5FF; border-radius: 8px; margin-bottom: 1.5rem;">
    <p style="margin:0; color:#FFFFFF;"><b>PRISM-GP 1.0 Webtool</b> | D. Kumar, A. J. Martin | Manipal Academy of Higher Education (MAHE) | Version 1.0 (2026).</p>
</div>
        """, unsafe_allow_html=True)

        st.markdown("<p style='color:#00E5FF; font-family:\"JetBrains Mono\";'>SCIENTIFIC LITERATURE AND COMPUTATIONAL PACKAGES</p>", unsafe_allow_html=True)
        st.markdown("Below is the complete list of scientific literature, software tools, and computational packages used in the development, validation, and deployment of PRISM-GP 1.0.")
        st.markdown("""
<div style="color:#94A3B8; font-size: 0.95rem;">

<strong style="color:white;">1. Machine Learning and Interpretability</strong>
*   **Nori, H. et al.** InterpretML: A Unified Framework for Machine Learning Interpretability. *arXiv:1909.09223* (2019).
*   **Pedregosa et al.** Scikit-Learn: Machine Learning in Python. *JMLR* 12, 2825–2830 (2011).

<strong style="color:white; display:block; margin-top:1rem;">2. Descriptor Generation and Cheminformatics</strong>
*   **Moriwaki et al.** Mordred: A Comprehensive Descriptor Library for Molecular Descriptors. *J. Cheminf.* 10, 4 (2018).
*   **Bemis, G. W., & Murcko, M. A.** The Properties of Known Drugs. 1. Molecular Frameworks. *J. Med. Chem.* 39, 2887-2893 (1996).
*   **RDKit:** Open-source cheminformatics. http://www.rdkit.org.

<strong style="color:white; display:block; margin-top:1rem;">3. Model Interpretation, Validation and Performance Evaluation</strong>
*   **Efron, B.** Better Bootstrap Confidence Intervals. *J. Am. Stat. Assoc.* 82, 171-185 (1987).
*   **Powers, D.** Evaluation: Precision, Recall, F-measure, ROC, Informedness, Markedness. *JMLT* 2, 37–63 (2011).

<strong style="color:white; display:block; margin-top:1rem;">4. Applicability Domain (AD) and False Positive Filters</strong>
*   **Baell, J. B., & Holloway, G. A.** New Substructure Filters for Removal of Pan Assay Interference Compounds (PAINS) from Screening Libraries and for Their Exclusion in Bioassays. *J. Med. Chem.* 53, 2719–2740 (2010).
*   **Sahigara, F. et al.** Comparison of Different Approaches to Define the Applicability Domain. *Molecules (Basel, Switzerland)*,17(5),4791–4810.
*   **Tropsha, A.** Best Practices for QSAR Model Development, Validation, and Exploitation. *Mol. Inf.* 29, 476-488 (2010).

<strong style="color:white; display:block; margin-top:1rem;">5. Datasets and Source</strong>
*   **Zdrazil, B. et al.** The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods. *Nucleic Acids Res.* 52, D1180-D1192 (2024).
*   **Kim, S. et al.** PubChem 2025 update. *Nucleic Acids Res.* 53, D1516-D1525 (2025).
*   **Gilson, M. K. et al.** BindingDB in 2024: a FAIR knowledgebase of protein–small molecule interactions. *Nucleic Acids Res.* 52, D634-D642 (2024).


<strong style="color:white; display:block; margin-top:1rem;">6. Software, Platforms and Versions (Used in PRISM-GP 1.0)</strong>

| Software / Package | Version | Purpose |
| :--- | :--- | :--- |
| **Python** | 3.10 | Core Development |
| **Streamlit** | 1.50 | Web Interface Deployment |
| **RDKit** | 2025.03.6 | SMILES parsing |
| **Mordred** | 1.2.0 | Topological Descriptor generation |
| **InterpretML** | latest | Explainable Boosting Machines |
| **scikit-learn** | 1.4.2 | Model inference and Normalization |
| **NumPy** | 1.25.2 | Numerical computing |
| **Pandas** | 2.3.2 | DataFrame processing |
| **Graphviz** | latest | Flowchart rendering |

</div>
        """, unsafe_allow_html=True)