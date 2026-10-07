from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import streamlit as st
from PIL import Image

from lung_svm.cli import decision_scores
from lung_svm.features import image_hog
from lung_svm.ood import ood_distances


st.set_page_config(
    page_title="LungLens | X-ray classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .stApp {
        background:
          radial-gradient(ellipse at 8% 0%, rgba(34, 168, 163, .11), transparent 32rem),
          linear-gradient(180deg, #f5f9fc 0%, #edf3f8 100%);
      }
      .block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 3rem; }
      [data-testid="stHeader"] { background: rgba(245, 249, 252, .72); }
      .hero {
        padding: 2.2rem 2.4rem; border-radius: 24px; color: #f4fbff;
        background: linear-gradient(120deg, #102638 0%, #173c50 56%, #087e83 140%);
        box-shadow: 0 18px 50px rgba(15, 42, 62, .16); margin-bottom: 1.35rem;
      }
      .hero-kicker { color: #8fe2d7; font-size: .76rem; font-weight: 750;
        letter-spacing: .14em; text-transform: uppercase; margin-bottom: .7rem; }
      .hero h1 { color: #fff; font-size: clamp(2rem, 5vw, 3.2rem); line-height: 1.08;
        letter-spacing: -.04em; margin: 0 0 .8rem 0; }
      .hero p { color: #d1e2eb; font-size: 1.05rem; max-width: 700px; margin: 0; }
      .panel {
        background: rgba(255,255,255,.88); border: 1px solid #dce7ee;
        border-radius: 20px; padding: 1.25rem 1.35rem;
        box-shadow: 0 8px 26px rgba(26, 58, 78, .06); margin-bottom: 1rem;
      }
      .panel-label { color: #087e83; font-size: .75rem; font-weight: 750;
        letter-spacing: .12em; text-transform: uppercase; margin-bottom: .25rem; }
      .result {
        background: linear-gradient(120deg, #e3f6f2, #f2fbf8);
        border: 1px solid #b8e5da; border-radius: 16px; padding: 1rem 1.1rem;
        color: #153f3d; margin: .35rem 0 .7rem 0;
      }
      .result-title { color: #075e5e; font-size: 1.05rem; font-weight: 750; }
      .result-note { color: #466969; font-size: .88rem; margin-top: .25rem; }
      .fine-print { color: #637789; font-size: .82rem; }
      div.stButton > button { border-radius: 12px; min-height: 3rem; font-weight: 700; }
      div[data-testid="stFileUploader"] { background: #f8fbfd; border-radius: 14px; }
      @media (max-width: 700px) {
        .block-container { padding: 1rem 1rem 2rem; }
        .hero { padding: 1.6rem; border-radius: 18px; }
      }
    </style>
    <section class="hero">
      <div class="hero-kicker">Computer vision · HOG + PCA + SVM</div>
      <h1>LungLens</h1>
      <p>A simple research demo that screens a chest X-ray for 14 labeled findings.
         Upload an image to see the model's predicted labels.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

model_path = Path("artifacts/lung_svm_model.joblib")


@st.cache_resource
def load_model_assets():
    return joblib.load(model_path) if model_path.is_file() else None


model_bundle = load_model_assets()
if model_bundle is None:
    st.error("No trained model was found. Add the trained model artifacts before starting the app.")
    st.stop()
st.caption("Active model: HOG + PCA + SVM, with an out-of-distribution image rejection gate")

left, right = st.columns([1.12, 0.88], gap="large")
with left:
    st.markdown(
        '<div class="panel"><div class="panel-label">Step 1</div>'
        '<h3 style="margin:.15rem 0 .35rem 0">Choose an X-ray</h3>'
        '<p class="fine-print">PNG, JPG, or JPEG · one image at a time</p></div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Upload chest X-ray", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
    )
    if uploaded:
        image = Image.open(uploaded).convert("L")
        st.image(image, caption=uploaded.name, use_container_width=True)
        classify = st.button("Analyze image", type="primary", use_container_width=True)
    else:
        image = None
        classify = False
        st.markdown(
            '<div class="panel fine-print">Your selected image preview will appear here.</div>',
            unsafe_allow_html=True,
        )

with right:
    st.markdown(
        '<div class="panel"><div class="panel-label">Step 2</div>'
        '<h3 style="margin:.15rem 0 .35rem 0">Review the output</h3>'
        '<p class="fine-print">Predicted labels are shown with SVM decision margins, not probabilities.</p></div>',
        unsafe_allow_html=True,
    )
    if classify and image is not None:
        with st.spinner("Checking the image and running the classifier…"):
            with TemporaryDirectory(dir="artifacts") as temp_dir:
                temp_path = Path(temp_dir) / "upload.png"
                image.save(temp_path)
                features = image_hog(temp_path, model_bundle["image_size"])[None, :]
            transformed = model_bundle["pca"].transform(model_bundle["scaler"].transform(features))
            ood_distance = None
            if "ood_reference" in model_bundle:
                ood_distance = float(ood_distances(model_bundle["ood_reference"], transformed)[0])
            if ood_distance is not None and ood_distance > model_bundle["ood_threshold"]:
                findings = None
            else:
                scores = decision_scores(model_bundle["models"], transformed)[0]
                findings = [
                    (name, float(score))
                    for name, score, threshold in zip(
                        model_bundle["diseases"], scores, model_bundle["thresholds"]
                    )
                    if score >= threshold
                ]

        if findings is None:
            st.warning("This image does not look sufficiently similar to the chest X-rays used to train the model. Upload a chest X-ray to get a prediction.")
        elif findings:
            st.markdown('<div class="result-title">Predicted findings</div>', unsafe_allow_html=True)
            for name, score in sorted(findings, key=lambda item: item[1], reverse=True):
                st.markdown(
                    f'<div class="result"><div class="result-title">{name.replace("_", " ")}</div>'
                    f'<div class="result-note">Decision margin: {score:.3f}</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="result"><div class="result-title">No Finding</div>'
                '<div class="result-note">No abnormality passed its selected threshold.</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="panel fine-print">Choose an image, then select <b>Analyze image</b> to view predictions.</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")
st.markdown(
    '<p class="fine-print">Educational project only; not clinically validated and not for diagnosis or treatment. '
    'If hosted online, the selected image is processed by that hosting service. Do not upload identifiable or sensitive health images.</p>',
    unsafe_allow_html=True,
)
