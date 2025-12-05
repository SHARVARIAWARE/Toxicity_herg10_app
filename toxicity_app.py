# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import os
from io import BytesIO
from rdkit import Chem
from rdkit.Chem import Draw
from mordred import Calculator, descriptors
import PIL.Image

st.set_page_config(page_title="QSAR Ensemble (Regression)", layout="wide")

st.title("QSAR Ensemble Predictor — Regression")
st.markdown("Upload an Excel file with a `smiles` column. The app computes Mordred descriptors, scales with saved `x_scaler.pkl`, runs 5 regression NN models, and returns ensemble mean ± std. Download results as Excel.")

# ----------------------------
# Helpers: load models & scalers
# ----------------------------
@st.cache_resource
def load_models_and_scalers(models_folder="models", n_models=5):
    """Load models and scalers from models_folder."""
    # load scalers
    x_scaler = joblib.load(os.path.join(models_folder, "x_scaler.pkl"))
    y_scaler = joblib.load(os.path.join(models_folder, "y_scaler.pkl"))

    # load models
    models = []
    for i in range(1, n_models + 1):
        path = os.path.join(models_folder, f"ensemble_model_{i}.h5")
        models.append(tf.keras.models.load_model(path))
    return models, x_scaler, y_scaler

@st.cache_resource
def create_mordred_calculator():
    # ignore_3D to avoid 3D descriptors
    return Calculator(descriptors, ignore_3D=True)

# ----------------------------
# Descriptor + alignment utils
# ----------------------------
def smiles_to_mol(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError("Invalid SMILES")
    return m

def compute_mordred(smiles, calc):
    mol = smiles_to_mol(smiles)
    res = calc(mol)
    df = res.as_dataframe()
    df.columns = [str(c) for c in df.columns]
    return df.iloc[0]  # Series

def align_descriptors_to_scaler(desc_series: pd.Series, x_scaler):
    """
    Align Mordred descriptor Series to scaler's expected features.
    1) If scaler has feature_names_in_, use that order.
    2) Else fallback to first N numeric descriptors and pad zeros.
    """
    desc_df = desc_series.to_frame().T.replace([np.inf, -np.inf], np.nan)

    # infer n_features
    try:
        n_features = int(x_scaler.n_features_in_)
    except Exception:
        try:
            n_features = int(x_scaler.scale_.shape[0])
        except Exception:
            raise RuntimeError("Could not determine number of expected features from x_scaler.")

    # if feature names available, use them
    feature_names = None
    if hasattr(x_scaler, "feature_names_in_"):
        feature_names = list(x_scaler.feature_names_in_)

    if feature_names:
        vec = []
        for f in feature_names:
            if f in desc_df.columns:
                v = desc_df.iloc[0][f]
                vec.append(0.0 if (pd.isna(v) or not np.isfinite(v)) else float(v))
            else:
                vec.append(0.0)
        return np.array(vec).reshape(1, -1)

    # fallback: select numeric columns
    numeric_cols = []
    for c in desc_df.columns:
        try:
            float(desc_df.iloc[0][c])
            numeric_cols.append(c)
        except Exception:
            continue

    if len(numeric_cols) == 0:
        raise RuntimeError("No numeric Mordred descriptors found for this molecule.")

    if len(numeric_cols) >= n_features:
        selected = numeric_cols[:n_features]
        vec = [float(desc_df.iloc[0][c]) if np.isfinite(desc_df.iloc[0][c]) else 0.0 for c in selected]
        return np.array(vec).reshape(1, -1)
    else:
        vec = [float(desc_df.iloc[0][c]) if np.isfinite(desc_df.iloc[0][c]) else 0.0 for c in numeric_cols]
        vec.extend([0.0] * (n_features - len(vec)))
        return np.array(vec).reshape(1, -1)

def ensemble_predict_from_vector(vec, models, x_scaler, y_scaler):
    # scale
    Xs = x_scaler.transform(vec)
    preds_scaled = []
    for m in models:
        p = float(np.array(m.predict(Xs, verbose=0)).ravel()[0])
        preds_scaled.append(p)
    preds_scaled = np.array(preds_scaled)
    mean_scaled = preds_scaled.mean()
    std_scaled = preds_scaled.std(ddof=0)

    # inverse transform
    mean_pred = float(y_scaler.inverse_transform([[mean_scaled]])[0][0])
    preds = [float(y_scaler.inverse_transform([[pv]])[0][0]) for pv in preds_scaled]
    # convert std scaled to original space if y_scaler has scale_
    std_pred = std_scaled * (y_scaler.scale_[0] if hasattr(y_scaler, "scale_") else 1.0)
    return mean_pred, preds, std_pred

# ----------------------------
# Load resources
# ----------------------------
with st.spinner("Loading models and descriptor calculator..."):
    try:
        models, x_scaler, y_scaler = load_models_and_scalers("models", n_models=5)
        calc = create_mordred_calculator()
    except Exception as e:
        st.error("Failed to load models/scalers. Make sure models/ contains ensemble_model_1..5.h5 and x_scaler.pkl, y_scaler.pkl.")
        st.exception(e)
        st.stop()

# ----------------------------
# Upload area
# ----------------------------
st.sidebar.header("Upload")
uploaded_file = st.sidebar.file_uploader("Upload an Excel (.xlsx) with a column named 'smiles'", type=["xlsx"])
st.sidebar.markdown("Ensure `models/` folder exists with model & scaler files in the repo.")

show_mol = st.sidebar.checkbox("Show molecule image", value=True)
show_per_model = st.sidebar.checkbox("Show per-model predictions", value=True)

if uploaded_file:
    try:
        df_in = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not read Excel file: {e}")
        st.stop()

    if "smiles" not in df_in.columns.str.lower():
        # try case-insensitive
        lower_map = {c: c for c in df_in.columns}
        col_lower = [c.lower() for c in df_in.columns]
        if "smiles" in col_lower:
            orig = df_in.columns[col_lower.index("smiles")]
            df_in = df_in.rename(columns={orig: "smiles"})
        else:
            st.error("Uploaded file must have a 'smiles' column (case-insensitive).")
            st.stop()

    # ensure column name is exactly 'smiles'
    if "smiles" not in df_in.columns:
        df_in = df_in.rename(columns={c: c for c in df_in.columns})

    st.success(f"Loaded {len(df_in)} molecules.")
    st.info("Computing descriptors & predictions (this may take a while for large files).")

    results = []
    progress = st.progress(0)
    for idx, row in enumerate(df_in.itertuples(index=False), 1):
        try:
            smi = str(getattr(row, "smiles"))
            # compute descriptors
            desc_series = compute_mordred(smi, calc)
            vec = align_descriptors_to_scaler(desc_series, x_scaler)
            mean_pred, preds, std_pred = ensemble_predict_from_vector(vec, models, x_scaler, y_scaler)

            out = {
                "smiles": smi,
                "Ensemble_Mean": mean_pred,
                "Ensemble_STD": std_pred
            }
            # per model columns
            for i, p in enumerate(preds, 1):
                out[f"Model_{i}"] = p

            # optional molecule image bytes
            if show_mol:
                try:
                    mol = Chem.MolFromSmiles(smi)
                    pil_img = Draw.MolToImage(mol, size=(300, 200))
                    buffered = BytesIO()
                    pil_img.save(buffered, format="PNG")
                    out["mol_image_bytes"] = buffered.getvalue()
                except Exception:
                    out["mol_image_bytes"] = None

            results.append(out)
        except Exception as e:
            results.append({"smiles": str(getattr(row, "smiles")), "Error": str(e)})
        # update progress
        progress.progress(int(idx / len(df_in) * 100))

    progress.empty()
    df_out = pd.DataFrame(results)

    # show summary stats
    st.subheader("Prediction Summary")
    st.write(df_out[["Ensemble_Mean", "Ensemble_STD"]].describe().T)

    # show top/bottom molecules
    st.subheader("Top predicted molecules (highest Ensemble_Mean)")
    st.dataframe(df_out.sort_values("Ensemble_Mean", ascending=False).head(10).reset_index(drop=True), use_container_width=True)

    st.subheader("Lowest predicted molecules (lowest Ensemble_Mean)")
    st.dataframe(df_out.sort_values("Ensemble_Mean", ascending=True).head(10).reset_index(drop=True), use_container_width=True)

    # show per-model predictions toggle
    if show_per_model:
        cols = ["smiles", "Ensemble_Mean", "Ensemble_STD"] + [f"Model_{i}" for i in range(1, len(models)+1)]
    else:
        cols = ["smiles", "Ensemble_Mean", "Ensemble_STD"]

    st.subheader("All Predictions")
    st.dataframe(df_out[cols], use_container_width=True)

    # display molecule images in an expander (first 6)
    if show_mol:
        st.subheader("Molecule Previews (first 12)")
        mm = df_out[df_out["mol_image_bytes"].notna()].head(12)
        cols = st.columns(4)
        for i, item in mm.iterrows():
            col = cols[i % 4]
            col.image(item["mol_image_bytes"], use_column_width=True, caption=item["smiles"])

    # Download button
    def to_excel_bytes(df_):
        out = BytesIO()
        # drop image bytes before saving
        df_save = df_.copy()
        if "mol_image_bytes" in df_save.columns:
            df_save = df_save.drop(columns=["mol_image_bytes"])
        df_save.to_excel(out, index=False)
        return out.getvalue()

    st.subheader("Download Results")
    st.download_button("Download predictions (Excel)", to_excel_bytes(df_out), file_name="qsar_predictions.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.success("Done!")

else:
    st.info("Upload an Excel (.xlsx) file with a 'smiles' column to begin.")
