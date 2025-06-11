import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# === Page Setup ===
st.set_page_config(page_title="UniMAP Mango Classifier", layout="wide")

# === UniMAP Branding ===
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("unimap_logo.png", width=200)
with col_title:
    st.markdown("""
        <h1 style='margin-bottom:0;'>Harumanis Mango Disease Detection</h1>
        <p style='font-size:18px; margin-top:0;'>Universiti Malaysia Perlis · FKTE · FYP Mechatronic Engineering</p>
    """, unsafe_allow_html=True)
st.markdown("---")

# === Load Models ===
svm_model = joblib.load('model/SVM/svm_best_model.pkl')
rf_model = joblib.load('model/RF/rf_best_model.pkl')
mlp_model = joblib.load('model/MLP/mlp_best_model.pkl')
voting_model = joblib.load('model/Voting/final_voting_classifier.pkl')
dnn_model = tf.keras.models.load_model('model/DNN/dnn_model.h5')
cnn1d_model = tf.keras.models.load_model('model/CNN1D/cnn_1d_final_model.h5')
cnn_img_model = tf.keras.models.load_model('model/CNN_IMG/mobilenetv2_model.h5')

# === Load Preprocessors ===
scaler = joblib.load('model/MLP/mlp_scaler.pkl')
dnn_scaler = joblib.load('model/DNN/dnn_scaler.pkl')
svm_selector = joblib.load('model/SVM/svm_selector.pkl')
rf_selector = joblib.load('model/RF/rf_selector.pkl')

# === Model Selector ===
st.markdown("### 🧠 Select a Model")
model_name = st.selectbox("", [
    "SVM", "Random Forest", "MLP", "Voting Classifier", "DNN", "CNN 1D", "CNN Image"
])
# Predefine upload and folder path variables to avoid scope errors
uploaded_signals = None
uploaded_images = None
folder_path = ""

# === CLEAR ALL BUTTON ===
clear_all = st.button("🧹 Clear All")
if clear_all:
    st.session_state.clear()
    st.rerun()
    st.success("✅ All data cleared.")

# === CSV-Based Models ===
if model_name in ["SVM", "Random Forest", "MLP", "Voting Classifier", "DNN"]:
    st.markdown("### 📁 Upload Ultrasound Feature CSV")
    uploaded_file = st.file_uploader("Choose .csv file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file).dropna()
        if "label" in df.columns:
            X = df.drop("label", axis=1)
            y_true = df["label"]
            has_label = True
        else:
            X = df
            y_true = None
            has_label = False

        try:
            scaled_X = scaler.transform(X) if model_name != "DNN" else dnn_scaler.transform(X)
            if model_name == "SVM":
                scaled_X = svm_selector.transform(scaled_X)
            elif model_name == "Random Forest":
                scaled_X = rf_selector.transform(scaled_X)
        except Exception as e:
            st.error(f"❌ Feature mismatch: {e}")
            st.stop()

        model = {
            "SVM": svm_model,
            "Random Forest": rf_model,
            "MLP": mlp_model,
            "Voting Classifier": voting_model,
            "DNN": dnn_model
        }[model_name]

        y_pred = model.predict(scaled_X)
        y_proba = model.predict_proba(scaled_X)[:, 1] if model_name != "DNN" else model.predict(scaled_X)[:, 0]
        if model_name == "DNN":
            y_pred = (y_proba > 0.5).astype(int)

        df['Predicted'] = y_pred
        df['Class'] = np.where(y_pred == 1, 'Diseased', 'Healthy')
        df['Probability'] = y_proba.round(2)

        if has_label:
            acc = accuracy_score(y_true, y_pred)
            st.success(f"✅ Accuracy: **{acc:.2%}**")
            st.text("📊 Classification Report:")
            st.text(classification_report(y_true, y_pred, target_names=["Healthy", "Diseased"]))
            fig = plt.figure()
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                        xticklabels=["Healthy", "Diseased"],
                        yticklabels=["Healthy", "Diseased"])
            plt.title(f"{model_name} - Confusion Matrix")
            st.pyplot(fig)
        else:
            st.info("ℹ️ No 'label' column found. Showing predictions only.")

        st.subheader("📋 Full Prediction Results")
        st.dataframe(df[['Predicted', 'Class', 'Probability']])
        st.download_button("⬇️ Download Results", df.to_csv(index=False), "predictions.csv", "text/csv")

# === CNN 1D ===
elif model_name == "CNN 1D":
    st.markdown("### 📂 Upload `.npy` Signals or Enter Folder Path")
    uploaded_signals = st.file_uploader("Upload `.npy` files", type=['npy'], accept_multiple_files=True)
    folder_path = st.text_input("Or enter folder path containing `.npy` signals")

    files = []
    if uploaded_signals:
        files = uploaded_signals
    elif folder_path and os.path.isdir(folder_path):
        npy_paths = sorted(glob.glob(os.path.join(folder_path, "*.npy")))
        files = [open(path, 'rb') for path in npy_paths]

    if files:
        results = []
        for file in files:
            try:
                signal = np.load(file)
                signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
                signal = signal.reshape(1, -1, 1)
                prob = cnn1d_model.predict(signal)[0][0]
                pred = "Diseased" if prob > 0.5 else "Healthy"
                results.append({"File": file.name, "Prediction": pred, "Probability": round(prob, 2)})
            except Exception as e:
                st.warning(f"❌ {file.name}: {e}")
        df = pd.DataFrame(results)
        st.dataframe(df)
        st.download_button("⬇️ Download Results", df.to_csv(index=False), "cnn1d_results.csv", "text/csv")

# === CNN Image ===
elif model_name == "CNN Image":
    st.markdown("### 🖼️ Upload `.png` Signal Images or Enter Folder Path")
    uploaded_images = st.file_uploader("Upload `.png` images", type=['png'], accept_multiple_files=True)
    folder_path = st.text_input("Or enter folder path containing `.png` signal images")

    files = []
    if uploaded_images:
        files = uploaded_images
    elif folder_path and os.path.isdir(folder_path):
        files = sorted(glob.glob(os.path.join(folder_path, "*.png")))

    if files:
        results = []
        for file in files:
            try:
                if isinstance(file, str):  # from folder
                    img = tf.keras.preprocessing.image.load_img(file, target_size=(224, 224))
                    name = os.path.basename(file)
                else:  # from uploader
                    img = tf.keras.preprocessing.image.load_img(file, target_size=(224, 224))
                    name = file.name

                arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                arr = np.expand_dims(arr, axis=0)
                prob = cnn_img_model.predict(arr)[0][0]
                pred = "Diseased" if prob > 0.5 else "Healthy"
                results.append({"File": name, "Prediction": pred, "Probability": round(prob, 2)})
            except Exception as e:
                st.warning(f"❌ {file}: {e}")
        df = pd.DataFrame(results)
        st.dataframe(df)
        st.download_button("⬇️ Download Results", df.to_csv(index=False), "cnn_image_results.csv", "text/csv")
