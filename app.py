import os
import requests
from PIL import Image
import streamlit as st
from cv.src import configs


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
icon_path = os.path.join(BASE_DIR, "app-icon.png")
page_icon = Image.open(icon_path) if os.path.exists(icon_path) else "🚗"

st.set_page_config(
    page_title="Car AI Classifier | Performance Edition",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --bmw-bg: #0B0D10;
        --bmw-surface: #13171E;
        --bmw-surface-2: #191F29;
        --bmw-border: #262E3B;
        --bmw-blue: #1C69D4;
        --bmw-blue-hover: #1554AD;
        --bmw-text: #F4F6F9;
        --bmw-muted: #9AA4B2;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main app background */
    .stApp {
        background: radial-gradient(circle at 85% 10%, rgba(28, 105, 212, 0.12), transparent 40%),
                    var(--bmw-bg);
        color: var(--bmw-text);
    }

    /* Hide default Streamlit top decoration bar */
    header[data-testid="stHeader"] {
        background: rgba(11, 13, 16, 0.85);
        backdrop-filter: blur(8px);
        border-bottom: 1px solid var(--bmw-border);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0E1116;
        border-right: 1px solid var(--bmw-border);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.85rem;
        color: var(--bmw-muted);
    }

    /* Hero banner */
    .bmw-hero {
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.75rem;
        background: linear-gradient(90deg, #13171E 0%, #0F131A 100%);
        border: 1px solid var(--bmw-border);
        border-left: 4px solid var(--bmw-blue);
    }

    .bmw-kicker {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--bmw-blue);
        margin-bottom: 0.35rem;
    }

    .bmw-title {
        font-size: 2.1rem;
        font-weight: 300;
        letter-spacing: 0.02em;
        color: var(--bmw-text);
        margin: 0;
        line-height: 1.2;
    }

    .bmw-title strong {
        font-weight: 700;
    }

    .bmw-subtitle {
        color: var(--bmw-muted);
        font-size: 0.95rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    /* Section labels */
    .bmw-section-label {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--bmw-muted);
        border-bottom: 1px solid var(--bmw-border);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* File uploader */
    div[data-testid="stFileUploader"] section {
        background-color: var(--bmw-surface);
        border: 1px dashed var(--bmw-border);
        border-radius: 2px;
        padding: 1.25rem;
        transition: border-color 0.2s ease;
    }

    div[data-testid="stFileUploader"] section:hover {
        border-color: var(--bmw-blue);
    }

    /* Primary BMW button */
    div.stButton > button[kind="primary"] {
        background-color: var(--bmw-blue);
        color: #FFFFFF;
        border: none;
        border-radius: 2px;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.7rem 1.5rem;
        transition: background-color 0.2s ease, transform 0.1s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: var(--bmw-blue-hover);
        color: #FFFFFF;
    }

    div.stButton > button[kind="primary"]:active {
        transform: translateY(1px);
    }

    /* Secondary buttons (e.g. Browse files) */
    div.stButton > button:not([kind="primary"]),
    div[data-testid="stFileUploader"] button {
        background-color: transparent;
        color: var(--bmw-text);
        border: 1px solid var(--bmw-border);
        border-radius: 2px;
        font-weight: 500;
        letter-spacing: 0.04em;
    }

    div[data-testid="stFileUploader"] button:hover {
        border-color: var(--bmw-blue);
        color: #FFFFFF;
    }

    /* Text inputs */
    div[data-testid="stTextInput"] input {
        background-color: var(--bmw-surface);
        color: var(--bmw-text);
        border: 1px solid var(--bmw-border);
        border-radius: 2px;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: var(--bmw-blue);
        box-shadow: none;
    }

    /* Result card */
    .bmw-result-card {
        background-color: var(--bmw-surface);
        border: 1px solid var(--bmw-border);
        border-top: 3px solid var(--bmw-blue);
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .bmw-result-label {
        font-size: 0.72rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--bmw-muted);
        margin-bottom: 0.25rem;
    }

    .bmw-result-model {
        font-size: 1.75rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: 0.02em;
        margin-bottom: 0.75rem;
    }

    .bmw-result-conf {
        font-size: 1.1rem;
        font-weight: 500;
        color: var(--bmw-blue);
    }

    /* Progress bars */
    div[data-testid="stProgress"] > div > div > div > div {
        background-color: var(--bmw-blue);
    }

    /* Image frame */
    div[data-testid="stImage"] img {
        border: 1px solid var(--bmw-border);
        border-radius: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "prediction" not in st.session_state:
    st.session_state["prediction"] = None
if "last_file_key" not in st.session_state:
    st.session_state["last_file_key"] = None

with st.sidebar:
    st.header("System Configuration")
    api_url = st.text_input("Inference Endpoint", value=configs.API_URL)
    st.markdown("---")
    st.caption("ARCHITECTURE: **ResNet18**")
    st.caption("BACKEND: **FastAPI**")
    st.caption("INTERFACE: **Streamlit**")

st.markdown(
    """
    <div class="bmw-hero">
        <div class="bmw-kicker">Vision Intelligence • Model Recognition</div>
        <h1 class="bmw-title">CAR <strong>AI CLASSIFIER</strong></h1>
        <p class="bmw-subtitle">Upload a vehicle photograph to identify its exact model with neural precision.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([1.15, 1], gap="large")

with col_left:
    st.markdown('<div class="bmw-section-label">01 / Vehicle Input</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Select vehicle image (JPG, JPEG, PNG)",
        type=["jpeg", "png", "jpg"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        current_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state["last_file_key"] != current_key:
            st.session_state["prediction"] = None
            st.session_state["last_file_key"] = current_key

        st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
        run_pred = st.button("Analyze Vehicle", type="primary", use_container_width=True)
    else:
        st.session_state["prediction"] = None
        st.session_state["last_file_key"] = None
        run_pred = False

with col_right:
    st.markdown('<div class="bmw-section-label">02 / Telemetry & Output</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.info("Awaiting vehicle image upload.")
    else:
        if run_pred:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "image/jpeg",
                )
            }

            with st.spinner("Running neural inference..."):
                try:
                    resp = requests.post(api_url, files=files, timeout=30)
                    resp.raise_for_status()
                    st.session_state["prediction"] = resp.json()
                except requests.RequestException as exc:
                    st.session_state["prediction"] = None
                    st.error(f"API request failed: {exc}")
                    st.stop()

        data = st.session_state["prediction"]
        if data is not None:
            model_name = str(data["pred_label"]).replace("_", " ")
            conf_value = float(data["confidence"])

            st.markdown(
                f"""
                <div class="bmw-result-card">
                    <div class="bmw-result-label">Identified Model</div>
                    <div class="bmw-result-model">{model_name}</div>
                    <div class="bmw-result-conf">Confidence: {conf_value:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if data.get("top_3"):
                st.markdown('<div class="bmw-section-label">Top-3 Match Distribution</div>', unsafe_allow_html=True)
                for item in data["top_3"]:
                    label = str(item["label"]).replace("_", " ")
                    score = float(item["confidence"])
                    st.write(f"**{label}** — `{score:.2f}%`")
                    st.progress(min(max(score / 100.0, 0.0), 1.0))
        else:
            st.info("Press **ANALYZE VEHICLE** to run classification.")