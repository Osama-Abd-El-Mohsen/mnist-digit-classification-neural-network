from streamlit_drawable_canvas import st_canvas
import streamlit as st
from PIL import Image
import numpy as np
import joblib

# ===================== Page Config =====================
st.set_page_config(
    page_title="Neural Canvas",
    page_icon="🧠",
    layout="wide"
)

# ===================== Styling =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

* {
    font-family: 'Space Grotesk', sans-serif;
}

code, .mono {
    font-family: 'JetBrains Mono', monospace;
}

:root {
    --neon-cyan: #00f5ff;
    --neon-green: #00ff88;
    --neon-blue: #0088ff;
    --neon-teal: #00d4aa;
    --bg-primary: #020a0a;
    --bg-secondary: #041414;
    --bg-card: rgba(5, 25, 25, 0.85);
    --bg-glass: rgba(0, 245, 255, 0.03);
    --text-primary: #ffffff;
    --text-secondary: #8892b0;
    --text-dim: #4a6868;
    --border-glow: rgba(0, 245, 255, 0.3);
    --gradient-primary: linear-gradient(135deg, #00ff88 0%, #00f5ff 50%, #0088ff 100%);
    --gradient-accent: linear-gradient(135deg, #00f5ff 0%, #00d4aa 100%);
}

/* Main App Background */
.stApp {
    background: var(--bg-primary);
    background-image: 
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0, 255, 136, 0.12), transparent),
        radial-gradient(ellipse 60% 40% at 80% 50%, rgba(0, 245, 255, 0.1), transparent),
        radial-gradient(ellipse 50% 30% at 20% 80%, rgba(0, 136, 255, 0.08), transparent);
}

/* Grid Pattern Overlay */
.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
        linear-gradient(rgba(0, 255, 136, 0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 255, 136, 0.02) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: 0;
}

[data-testid="stMainBlockContainer"] {
    padding: 1.5rem 4rem;
    position: relative;
    z-index: 1;
}

/* Hide Streamlit Elements */
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, .stDeployButton { display: none !important; }

/* ===== HEADER SECTION ===== */
.header {
    text-align: center;
    padding: 2rem 0 3rem;
    position: relative;
}

.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--bg-glass);
    border: 1px solid rgba(0, 255, 136, 0.25);
    padding: 0.5rem 1.25rem;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--neon-green);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}

.header-badge::before {
    content: '';
    width: 6px;
    height: 6px;
    background: var(--neon-green);
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
    box-shadow: 0 0 10px var(--neon-green);
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}

.header-title {
    font-size: 4rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.03em;
    line-height: 1;
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 0 0 80px rgba(0, 255, 136, 0.4);
}

.header-subtitle {
    font-size: 1rem;
    color: var(--text-secondary);
    margin-top: 1rem;
    font-weight: 400;
    max-width: 500px;
    margin-left: auto;
    margin-right: auto;
}

/* ===== GLASS CARD ===== */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 255, 136, 0.1);
    border-radius: 24px;
    padding: 1.75rem;
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 255, 136, 0.5), transparent);
}

.card-label {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
}

.card-label-icon {
    width: 36px;
    height: 36px;
    background: var(--gradient-primary);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
}

.card-label-text {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-primary);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ===== CANVAS STYLES ===== */
.canvas-container {
    background: linear-gradient(135deg, #0a1a1a 0%, #0a2020 100%);
    border-radius: 16px;
    padding: 4px;
    position: relative;
}

.canvas-container::before {
    content: '';
    position: absolute;
    inset: -2px;
    background: var(--gradient-primary);
    border-radius: 18px;
    z-index: -1;
    opacity: 0.5;
    filter: blur(8px);
}

/* Canvas element styling */
[data-testid="stCustomComponentV1"] > div {
    border-radius: 12px !important;
    overflow: hidden;
}

/* ===== PREVIEW IMAGE ===== */
.preview-box {
    background: linear-gradient(135deg, rgba(0, 255, 136, 0.05) 0%, rgba(0, 245, 255, 0.05) 100%);
    border: 1px dashed rgba(0, 255, 136, 0.3);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    min-height: 200px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.preview-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--neon-cyan);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 1rem;
    opacity: 0.8;
}

[data-testid="stImage"] {
    border-radius: 8px;
    overflow: hidden;
    border: 2px solid rgba(0, 255, 136, 0.3);
    box-shadow: 0 0 30px rgba(0, 255, 136, 0.15);
}

/* ===== RESULT DISPLAY ===== */
.result-display {
    text-align: center;
    padding: 2.5rem 1.5rem;
}

.result-digit {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10rem;
    font-weight: 700;
    line-height: 1;
    margin: 0;
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 60px rgba(0, 255, 136, 0.5));
    animation: digitGlow 3s ease-in-out infinite;
}

@keyframes digitGlow {
    0%, 100% { filter: drop-shadow(0 0 60px rgba(0, 255, 136, 0.5)); }
    50% { filter: drop-shadow(0 0 80px rgba(0, 245, 255, 0.7)); }
}

.result-meta {
    margin-top: 1.5rem;
}

.result-label {
    font-size: 0.8rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1.5rem;
}

/* Confidence Meter */
.confidence-meter {
    background: rgba(0, 20, 20, 0.5);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    border: 1px solid rgba(0, 255, 136, 0.1);
}

.confidence-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
}

.confidence-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.confidence-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
    font-weight: 600;
    color: var(--neon-green);
}

.confidence-track {
    height: 6px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 3px;
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    background: var(--gradient-primary);
    border-radius: 3px;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 0 20px rgba(0, 255, 136, 0.4);
}

/* Waiting State */
.waiting-state {
    text-align: center;
    padding: 3rem 2rem;
}

.waiting-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
    opacity: 0.2;
    filter: grayscale(1);
}

.waiting-text {
    color: var(--text-dim);
    font-size: 0.9rem;
    line-height: 1.6;
}

.waiting-text strong {
    color: var(--neon-green);
    font-weight: 600;
}

/* ===== BUTTONS ===== */
.stButton > button {
    background: var(--gradient-primary);
    color: #000;
    border: none;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 1rem 2rem;
    border-radius: 12px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 25px rgba(0, 255, 136, 0.25);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 40px rgba(0, 255, 136, 0.4);
}

.stButton > button:active {
    transform: translateY(-1px);
}

/* Secondary/Clear Button */
[data-testid="column"]:nth-child(2) .stButton > button {
    background: transparent;
    border: 1px solid rgba(0, 255, 136, 0.2);
    color: var(--text-secondary);
    box-shadow: none;
}

[data-testid="column"]:nth-child(2) .stButton > button:hover {
    border-color: rgba(0, 255, 136, 0.5);
    color: var(--neon-green);
    background: rgba(0, 255, 136, 0.05);
    box-shadow: 0 4px 25px rgba(0, 255, 136, 0.15);
}

/* ===== STATS BAR ===== */
.stats-bar {
    display: flex;
    gap: 1rem;
    margin-top: 1.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(0, 255, 136, 0.1);
}

.stat-item {
    flex: 1;
    text-align: center;
    padding: 0.75rem;
    background: rgba(0, 20, 20, 0.4);
    border-radius: 10px;
    border: 1px solid rgba(0, 255, 136, 0.08);
}

.stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--neon-cyan);
}

.stat-label {
    font-size: 0.65rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.25rem;
}

/* ===== FOOTER ===== */
.footer {
    text-align: center;
    padding: 2.5rem 0 1rem;
    margin-top: 2rem;
}

.footer-content {
    display: inline-flex;
    align-items: center;
    gap: 2rem;
    padding: 0.75rem 1.5rem;
    background: var(--bg-glass);
    border: 1px solid rgba(0, 255, 136, 0.1);
    border-radius: 50px;
    backdrop-filter: blur(10px);
}

.footer-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.75rem;
    color: var(--text-dim);
}

.footer-dot {
    width: 4px;
    height: 4px;
    background: var(--neon-green);
    border-radius: 50%;
}

/* Responsive */
@media (max-width: 1024px) {
    .header-title { font-size: 3rem; }
    .result-digit { font-size: 7rem; }
    [data-testid="stMainBlockContainer"] { padding: 1rem 2rem; }
}

@media (max-width: 768px) {
    .header-title { font-size: 2.5rem; }
    .result-digit { font-size: 5rem; }
}
</style>
""", unsafe_allow_html=True)

# ===================== Session State =====================
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

if "processed_image" not in st.session_state:
    st.session_state.processed_image = None

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# ===================== Load Model =====================
MODEL_PATH = "model/mnist_model.pkl"
model = joblib.load(MODEL_PATH)

# ===================== Header Section =====================
st.markdown("""
<div class="header">
    <div class="header-badge">Neural Network Active</div>
    <h1 class="header-title">Neural Canvas</h1>
    <p class="header-subtitle">Draw a digit and watch the AI decode your handwriting in real-time using deep learning</p>
</div>
""", unsafe_allow_html=True)


# ===================== Main Layout =====================
col_left, col_right = st.columns([1.3, 0.7], gap="large")

with col_left:
    st.markdown("""
    <div class="glass-card">
        <div class="card-label">
            <div class="card-label-icon">✏️</div>
            <span class="card-label-text">Input Canvas</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Canvas and Preview side by side
    canvas_col, preview_col = st.columns([2.2, 1], gap="medium")
    
    with canvas_col:
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 1)",
            stroke_width=22,
            stroke_color="#FFFFFF",
            background_color="#0a0a1a",
            height=340,
            width=340,
            drawing_mode="freedraw",
            display_toolbar=True,
            key=f"canvas_{st.session_state.canvas_key}",
        )
    
    with preview_col:
        preview_container = st.empty()
        

    
    # Buttons
    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
    btn_col1, btn_col2 = st.columns(2, gap="medium")
    
    with btn_col1:
        predict_button = st.button("⚡ Analyze", use_container_width=True, key="predict_btn")
    


with col_right:
    st.markdown("""
    <div class="glass-card">
        <div class="card-label">
            <div class="card-label-icon">🎯</div>
            <span class="card-label-text">Prediction</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    result_container = st.empty()



# ===================== Predict =====================
if predict_button and canvas_result.image_data is not None:
    canvas_array = canvas_result.image_data[:, :, :3].astype("uint8")
    gray = np.dot(canvas_array[..., :3], [0.299, 0.587, 0.114]).astype("uint8")
    # No inversion needed - we're drawing white on dark

    gray[gray < 50] = 0
    gray[gray >= 50] = 255

    coords = np.column_stack(np.where(gray > 0))
    if coords.size != 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        digit = gray[y_min:y_max+1, x_min:x_max+1]
    else:
        digit = gray

    h, w = digit.shape
    if h > w:
        new_h = 20
        new_w = max(1, int(w * (20 / h)))
    else:
        new_w = 20
        new_h = max(1, int(h * (20 / w)))

    digit_resized = Image.fromarray(digit).resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS
    )
    digit_resized = np.array(digit_resized)

    final_img = np.zeros((28, 28), dtype=np.uint8)
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2
    final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized

    st.session_state.processed_image = final_img

    image_array = final_img.astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    confidence = float(prediction[0][predicted_class]) * 100

    st.session_state.prediction_result = (predicted_class, confidence)

# ===================== Update Display =====================
if st.session_state.prediction_result is not None:
    digit, conf = st.session_state.prediction_result
    result_container.markdown(f"""
        <div class="result-display">
            <p class="result-digit">{digit}</p>
            <div class="result-meta">
                <p class="result-label">Detected Digit</p>
                <div class="confidence-meter">
                    <div class="confidence-header">
                        <span class="confidence-label">Confidence</span>
                        <span class="confidence-value">{conf:.1f}%</span>
                    </div>
                    <div class="confidence-track">
                        <div class="confidence-fill" style="width: {conf}%;"></div>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    result_container.markdown("""
        <div class="waiting-state">
            <div class="waiting-icon">🧠</div>
            <p class="waiting-text">Draw a digit on the canvas<br>then click <strong>⚡ Analyze</strong></p>
        </div>
    """, unsafe_allow_html=True)

if st.session_state.processed_image is not None:
    preview_container.image(
        st.session_state.processed_image,
        clamp=True,
        use_container_width=True
    )
else:
    preview_container.markdown("""
        <div style="width: 100%; aspect-ratio: 1; background: rgba(0,0,0,0.3); border-radius: 8px; border: 1px dashed rgba(0, 245, 255, 0.2);"></div>
    """, unsafe_allow_html=True)

# ===================== Footer =====================
st.markdown("""
<div class="footer">
    <div class="footer-content">
        <div class="footer-item">
            <span class="footer-dot"></span>
            <span>Streamlit</span>
        </div>
        <div class="footer-item">
            <span class="footer-dot"></span>
            <span>MNIST Dataset</span>
        </div>
        <div class="footer-item">
            <span class="footer-dot"></span>
            <span>Neural Network</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)