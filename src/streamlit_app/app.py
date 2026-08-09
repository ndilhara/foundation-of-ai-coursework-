import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import streamlit as st

# ── path setup (app lives in src/streamlit_app/, models/images are in src/) ──
SRC_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(SRC_DIR, 'models')
SAMPLE_DIR = os.path.join(SRC_DIR, 'sample_data', 'images')

DR_LABELS = {0: 'No DR', 1: 'Mild NPDR', 2: 'Moderate NPDR', 3: 'Severe NPDR', 4: 'PDR'}
DR_COLORS = {0: '#4CAF50', 1: '#8BC34A', 2: '#FFC107', 3: '#FF5722', 4: '#B71C1C'}

URGENCY_BADGE = {
    'ANNUAL SCREENING':    ('🟢', '#E8F5E9', '#2E7D32'),
    'VIGILANCE SCREENING': ('🟡', '#F9FBE7', '#827717'),
    'LOCAL MONITORING':    ('🟡', '#FFF8E1', '#F57F17'),
    'ROUTINE REFERRAL':    ('🟠', '#FFF3E0', '#E65100'),
    'PRIORITY REFERRAL':   ('🔴', '#FBE9E7', '#BF360C'),
    'URGENT REFERRAL':     ('🔴', '#FFEBEE', '#B71C1C'),
    'RETAKE IMAGE':        ('⚪', '#F5F5F5', '#424242'),
}

IMG_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached so weights are loaded only once per session)
# ─────────────────────────────────────────────────────────────────────────────

def _build_resnet50(num_classes: int) -> nn.Module:
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(m.fc.in_features, num_classes))
    return m


@st.cache_resource(show_spinner='Loading model weights…')
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths = {
        'm1':  os.path.join(MODEL_DIR, 'best_method1_multiclass_resnet50.pth'),
        'bin': os.path.join(MODEL_DIR, 'best_method2_binary_resnet50.pth'),
        'sev': os.path.join(MODEL_DIR, 'best_method2_severity_resnet50.pth'),
    }
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        return None, device, paths

    m1  = _build_resnet50(5).to(device)
    bin_m = _build_resnet50(2).to(device)
    sev_m = _build_resnet50(4).to(device)

    m1.load_state_dict(torch.load(paths['m1'],  map_location=device))
    bin_m.load_state_dict(torch.load(paths['bin'], map_location=device))
    sev_m.load_state_dict(torch.load(paths['sev'], map_location=device))

    for m in (m1, bin_m, sev_m):
        m.eval()

    return (m1, bin_m, sev_m), device, paths


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(image: Image.Image, models_tuple, device):
    m1, bin_m, sev_m = models_tuple
    tensor = IMG_TRANSFORMS(image).unsqueeze(0).to(device)

    # Method 1 — 5-class
    logits_m1 = m1(tensor)
    probs_m1  = torch.softmax(logits_m1, dim=1).cpu().numpy()[0]
    grade_m1  = int(probs_m1.argmax())

    # Method 2 — hierarchical
    bin_pred = bin_m(tensor).argmax(1).item()
    if bin_pred == 1:
        sev_pred = sev_m(tensor).argmax(1).item()
        grade_m2 = sev_pred + 1
    else:
        grade_m2 = 0

    return grade_m1, probs_m1, grade_m2


# ─────────────────────────────────────────────────────────────────────────────
# Triage engine
# ─────────────────────────────────────────────────────────────────────────────

def calc_tabular_risk(age, duration, uses_insulin, has_hypertension,
                      has_dyslipidemia, has_kidney_disease, on_dialysis):
    if has_kidney_disease or on_dialysis:
        return 'HIGH'
    if duration > 15.5:
        return 'HIGH'
    if duration > 11.5 and uses_insulin and age > 18.5:
        return 'HIGH'
    if has_hypertension and has_dyslipidemia and uses_insulin:
        return 'HIGH'
    if has_hypertension or has_dyslipidemia:
        return 'MODERATE'
    if duration > 5.5 and uses_insulin and age > 18.5:
        return 'MODERATE'
    return 'LOW'


def evaluate_triage(cnn_grade, macular_edema, quality,
                    age, duration, uses_insulin, has_hypertension,
                    has_dyslipidemia, has_kidney_disease, on_dialysis):
    if quality == 'Inadequate':
        return {'action': 'RETAKE IMAGE', 'tabular_risk': 'N/A',
                'rationale': 'Image quality inadequate for reliable classification.'}

    risk = calc_tabular_risk(age, duration, uses_insulin, has_hypertension,
                             has_dyslipidemia, has_kidney_disease, on_dialysis)

    if cnn_grade >= 3 or macular_edema:
        return {'action': 'URGENT REFERRAL', 'tabular_risk': risk,
                'rationale': (f'Critical damage detected (Grade {cnn_grade}'
                              + (' + Macular Edema' if macular_edema else '')
                              + '). Immediate specialist intervention required.')}
    if cnn_grade == 2:
        if risk == 'HIGH':
            return {'action': 'PRIORITY REFERRAL', 'tabular_risk': risk,
                    'rationale': 'Moderate DR aggravated by high systemic risk. Escalated priority.'}
        return {'action': 'ROUTINE REFERRAL', 'tabular_risk': risk,
                'rationale': 'Moderate DR detected. Standard specialist evaluation needed.'}
    if cnn_grade == 1:
        if risk == 'HIGH':
            return {'action': 'ROUTINE REFERRAL', 'tabular_risk': risk,
                    'rationale': 'Mild DR with high systemic risk — proactive specialist review needed.'}
        if risk == 'MODERATE':
            return {'action': 'LOCAL MONITORING', 'tabular_risk': risk,
                    'rationale': 'Mild DR aggravated by cardiovascular risk factors. 6-month follow-up.'}
        return {'action': 'LOCAL MONITORING', 'tabular_risk': risk,
                'rationale': 'Mild DR in a well-controlled patient. 12-month follow-up.'}
    # Grade 0
    if risk == 'HIGH':
        return {'action': 'VIGILANCE SCREENING', 'tabular_risk': risk,
                'rationale': 'Retina healthy but systemic risk is high — 6-month screening recommended.'}
    return {'action': 'ANNUAL SCREENING', 'tabular_risk': risk,
            'rationale': 'Retina healthy, systemic health stable. Standard yearly checkup.'}


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='DR Triage — Inference Demo',
    page_icon='👁',
    layout='wide',
)

st.title('👁 Diabetic Retinopathy Triage System')
st.caption('Hybrid CNN + Clinical Knowledge Base · Inference Demo')

# ── load models ───────────────────────────────────────────────────────────────
models_tuple, device, model_paths = load_models()

with st.sidebar:
    st.header('Model Status')
    for label, key in [('Method 1 (multi-class)', 'm1'),
                        ('Method 2 — binary',       'bin'),
                        ('Method 2 — severity',     'sev')]:
        exists = os.path.exists(model_paths[key])
        st.write(('✅ ' if exists else '❌ ') + label)
    if models_tuple is None:
        st.error('One or more `.pth` files not found in `src/models/`.  '
                 'Run notebooks 02 & 03 to generate the weights.')
    else:
        st.success(f'All models loaded on **{device}**')
    st.divider()
    st.caption('Run `generate_sample_data.py` to refresh sample images.')

# ── two-column layout ─────────────────────────────────────────────────────────
col_img, col_form = st.columns([1, 1], gap='large')

# ── LEFT: image selection ─────────────────────────────────────────────────────
with col_img:
    st.subheader('Fundus Image')

    img_source = st.radio('Image source', ['Upload image', 'Pick from sample set'],
                          horizontal=True)
    selected_image: Image.Image | None = None

    if img_source == 'Upload image':
        uploaded = st.file_uploader('Upload a fundus image',
                                    type=['jpg', 'jpeg', 'png'],
                                    label_visibility='collapsed')
        if uploaded:
            selected_image = Image.open(uploaded).convert('RGB')

    else:  # sample set
        sample_files = sorted(os.listdir(SAMPLE_DIR)) if os.path.isdir(SAMPLE_DIR) else []
        if sample_files:
            chosen = st.selectbox('Select sample image', sample_files)
            selected_image = Image.open(
                os.path.join(SAMPLE_DIR, chosen)).convert('RGB')
        else:
            st.warning('No sample images found. Run `generate_sample_data.py`.')

    if selected_image:
        st.image(selected_image, use_container_width=True)

# ── RIGHT: clinical metadata form ─────────────────────────────────────────────
with col_form:
    st.subheader('Patient Clinical Information')

    age      = st.number_input('Patient age (years)', min_value=1, max_value=120, value=55)
    duration = st.number_input('Diabetes duration (years)', min_value=0.0,
                               max_value=60.0, value=8.0, step=0.5)
    insulin  = st.checkbox('Using insulin')

    st.markdown('**Comorbidities**')
    c1, c2 = st.columns(2)
    with c1:
        hypertension  = st.checkbox('Hypertension')
        kidney        = st.checkbox('Kidney / renal disease')
    with c2:
        dyslipidemia  = st.checkbox('Dyslipidemia / hyperlipidemia')
        dialysis      = st.checkbox('On dialysis')

    st.markdown('**Image Assessment**')
    quality      = st.selectbox('Image quality', ['Adequate', 'Inadequate'])
    mac_edema    = st.checkbox('Macular oedema present')

    run_btn = st.button('▶  Run Inference', type='primary',
                        disabled=(selected_image is None or models_tuple is None),
                        use_container_width=True)

# ── RESULTS ───────────────────────────────────────────────────────────────────
if run_btn and selected_image and models_tuple:
    st.divider()
    st.subheader('Results')

    grade_m1, probs_m1, grade_m2 = predict(selected_image, models_tuple, device)

    decision = evaluate_triage(
        cnn_grade       = grade_m1,
        macular_edema   = mac_edema,
        quality         = quality,
        age             = float(age),
        duration        = float(duration),
        uses_insulin    = insulin,
        has_hypertension  = hypertension,
        has_dyslipidemia  = dyslipidemia,
        has_kidney_disease= kidney,
        on_dialysis       = dialysis,
    )

    # ── grade cards ──────────────────────────────────────────────────────────
    rc1, rc2 = st.columns(2)
    with rc1:
        g = grade_m1
        conf = probs_m1[g] * 100
        st.markdown(f"""
        <div style="background:{DR_COLORS[g]}22;border-left:5px solid {DR_COLORS[g]};
             padding:16px;border-radius:8px;">
          <p style="margin:0;font-size:13px;color:#555;">Method 1 — Multi-class</p>
          <p style="margin:4px 0 0;font-size:22px;font-weight:700;color:{DR_COLORS[g]};">
            Grade {g} — {DR_LABELS[g]}</p>
          <p style="margin:2px 0 0;font-size:13px;color:#666;">Confidence: {conf:.1f}%</p>
        </div>""", unsafe_allow_html=True)

    with rc2:
        g2 = grade_m2
        st.markdown(f"""
        <div style="background:{DR_COLORS[g2]}22;border-left:5px solid {DR_COLORS[g2]};
             padding:16px;border-radius:8px;">
          <p style="margin:0;font-size:13px;color:#555;">Method 2 — Hierarchical</p>
          <p style="margin:4px 0 0;font-size:22px;font-weight:700;color:{DR_COLORS[g2]};">
            Grade {g2} — {DR_LABELS[g2]}</p>
          <p style="margin:2px 0 0;font-size:13px;color:#666;">Binary gate + severity</p>
        </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── confidence bar chart ─────────────────────────────────────────────────
    with st.expander('Method 1 — Class probabilities', expanded=True):
        import pandas as pd
        prob_df = pd.DataFrame({
            'Grade': [f'G{i} {DR_LABELS[i]}' for i in range(5)],
            'Probability': probs_m1,
        })
        st.bar_chart(prob_df.set_index('Grade'), color='#1976D2', height=220)

    # ── triage decision ───────────────────────────────────────────────────────
    action = decision['action']
    icon, bg, fg = URGENCY_BADGE.get(action, ('⚪', '#F5F5F5', '#000'))

    st.markdown(f"""
    <div style="background:{bg};border-left:6px solid {fg};padding:20px;
         border-radius:8px;margin-top:12px;">
      <p style="margin:0;font-size:13px;color:{fg};font-weight:600;letter-spacing:.5px;">
        TRIAGE DECISION</p>
      <p style="margin:6px 0 0;font-size:26px;font-weight:800;color:{fg};">
        {icon} {action}</p>
      <p style="margin:10px 0 0;font-size:14px;color:#444;">
        {decision['rationale']}</p>
      <p style="margin:8px 0 0;font-size:12px;color:#888;">
        Systemic risk level: <strong>{decision['tabular_risk']}</strong></p>
    </div>""", unsafe_allow_html=True)

    # ── clinical summary ──────────────────────────────────────────────────────
    with st.expander('Patient clinical summary'):
        flags = []
        if hypertension:   flags.append('Hypertension')
        if dyslipidemia:   flags.append('Dyslipidemia')
        if kidney:         flags.append('Kidney disease')
        if dialysis:       flags.append('Dialysis')
        st.json({
            'age':             age,
            'diabetes_years':  duration,
            'insulin':         insulin,
            'comorbidities':   flags if flags else 'None',
            'image_quality':   quality,
            'macular_oedema':  mac_edema,
        })
