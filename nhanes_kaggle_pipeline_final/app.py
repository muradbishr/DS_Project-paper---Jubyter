import io
import base64
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from pathlib import Path

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "lr_tuned.joblib"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

DEFAULT_FEATURES = [
    "age","gender","race","bmi","chol_total","chol_hdl",
    "smoke_now","family_diabetes","physically_active","bp_sys_mean","bp_dia_mean",
]

app = Flask(__name__, template_folder="templates", static_folder="static")

saved = joblib.load(MODEL_PATH)

imputer = saved["imputer"]
scaler = saved["scaler"]
model = saved["lr_cal"]

FEATURE_NAMES = saved.get("feature_names", DEFAULT_FEATURES)

SHAP_BG = saved.get("shap_background", None) or saved.get("shap_bg", None)
if isinstance(SHAP_BG, pd.DataFrame):
    SHAP_BG = SHAP_BG.values
SHAP_BG = None if SHAP_BG is None else np.asarray(SHAP_BG)

# --- decide input space based on SHAP_BG statistics ---
USE_SCALED = True  # default assumption
if SHAP_BG is not None and SHAP_BG.ndim == 2 and SHAP_BG.shape[1] == len(FEATURE_NAMES):
    bg_mean = float(np.mean(SHAP_BG))
    bg_std = float(np.std(SHAP_BG))
    # scaled data usually has mean around 0 and std around ~1 (not perfect, but close)
    USE_SCALED = (abs(bg_mean) < 0.5 and 0.3 < bg_std < 3.0)

print("FEATURE_NAMES:", FEATURE_NAMES)
print("SHAP_AVAILABLE:", SHAP_AVAILABLE)
print("Has SHAP_BG:", SHAP_BG is not None)
if SHAP_BG is not None:
    print("SHAP_BG shape:", SHAP_BG.shape)
    print("SHAP_BG global mean/std:", float(np.mean(SHAP_BG)), float(np.std(SHAP_BG)))
print("USE_SCALED (model input):", USE_SCALED)

# sanity check: model should give a range on bg
if SHAP_BG is not None:
    bg_probs = model.predict_proba(SHAP_BG)[:, 1]
    print("BG probs (class 1) min/mean/max:",
          float(bg_probs.min()), float(bg_probs.mean()), float(bg_probs.max()))

KERNEL_EXPLAINER = None

def preprocess(payload):
    row = {f: payload.get(f, np.nan) for f in FEATURE_NAMES}
    df = pd.DataFrame([row], columns=FEATURE_NAMES).apply(pd.to_numeric, errors="coerce")

    X_imp = imputer.transform(df)
    X_scaled = scaler.transform(X_imp)

    # choose the model input space that matches SHAP_BG / training
    X_model = X_scaled if USE_SCALED else X_imp
    return X_model, df

def make_shap_plot(contributions, top_k=11):
    if not contributions:
        return ""
    contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:top_k]
    names = [x[0] for x in contributions]
    values = [x[1] for x in contributions]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names[::-1], values[::-1])
    ax.set_xlabel("SHAP value (impact on predicted probability)")
    ax.set_title("Feature Contributions")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    buf.seek(0)
    png_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return png_b64

def get_kernel_explainer():
    global KERNEL_EXPLAINER
    if KERNEL_EXPLAINER is None:
        def predict_fn(x):
            return model.predict_proba(x)[:, 1]
        KERNEL_EXPLAINER = shap.KernelExplainer(predict_fn, SHAP_BG)
    return KERNEL_EXPLAINER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    payload = {}
    for f in FEATURE_NAMES:
        val = request.form.get(f)
        try:
            payload[f] = float(val)
        except Exception:
            payload[f] = np.nan

    X_model, _ = preprocess(payload)

    proba_vec = model.predict_proba(X_model)[0]
    prob = float(proba_vec[1])  # class 1 probability

    # Debug (remove later)
    print("payload:", payload)
    print("proba_vec:", proba_vec)

    contributions = []
    shap_png = ""

    if SHAP_AVAILABLE and SHAP_BG is not None:
        explainer = get_kernel_explainer()
        shap_vals = explainer.shap_values(X_model, nsamples=100)
        shap_arr = np.asarray(shap_vals).reshape(-1)
        contributions = sorted(list(zip(FEATURE_NAMES, shap_arr)), key=lambda x: abs(x[1]), reverse=True)
        shap_png = make_shap_plot(contributions)

    return render_template(
        "result.html",
        prob=prob,
        contributions=contributions,
        shap_png=shap_png,
        shap_available=(SHAP_AVAILABLE and SHAP_BG is not None),
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    X_model, _ = preprocess(data)
    prob = float(model.predict_proba(X_model)[0, 1])
    return jsonify({"probability": prob})

if __name__ == "__main__":
    app.run(debug=True)
