import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
import tempfile

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(page_title="Fibroid AI")

st.title("AI-Based Fibroid Detection System")

st.write("Upload an ultrasound image for analysis.")

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_my_model():
    model = load_model("fibroid_model_final.h5")
    return model

model = load_my_model()

# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Ultrasound Image",
    type=["jpg", "png", "jpeg"]
)

# ==========================================
# PREPROCESS IMAGE
# ==========================================

def preprocess_image(image):

    image = np.array(image)

    image = cv2.resize(image, (224, 224))

    original = image.copy()

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    return image, original

# ==========================================
# GRAD-CAM
# ==========================================

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    heatmap /= (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

# ==========================================
# SHAPE ANALYSIS
# ==========================================

def analyze_fibroid_shape(heatmap):

    heatmap_uint8 = np.uint8(255 * heatmap)

    h, w = heatmap_uint8.shape

    _, thresh = cv2.threshold(
        heatmap_uint8,
        150,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:

        return {
            "Shape": "Unknown",
            "Severity": "Low",
            "Remark": "No significant lesion detected"
        }

    largest = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(largest)

    perimeter = cv2.arcLength(largest, True)

    circularity = 0

    if perimeter > 0:
        circularity = (4 * np.pi * area) / (perimeter * perimeter)

    if circularity > 0.75:
        shape = "Round"
    elif circularity > 0.45:
        shape = "Oval"
    else:
        shape = "Irregular"

    if area > 150:
        severity = "High"
    elif area > 70:
        severity = "Moderate"
    else:
        severity = "Low"

    if severity == "High":
        remark = "Large fibroid attention region detected."
    elif severity == "Moderate":
        remark = "Moderate lesion appearance detected."
    else:
        remark = "Small localized lesion pattern."

    return {
        "Shape": shape,
        "Severity": severity,
        "Remark": remark
    }

# ==========================================
# PDF REPORT GENERATION
# ==========================================

def generate_pdf_report(
    prediction,
    confidence,
    analysis,
    heatmap_image
):

    pdf_path = "Fibroid_AI_Report.pdf"

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter
    )

    styles = getSampleStyleSheet()

    elements = []

    # Title

    title = Paragraph(
        "<b>AI-Based Fibroid Detection Report</b>",
        styles['Title']
    )

    elements.append(title)

    elements.append(Spacer(1, 20))

    # Table Data

    data = [

        ["Parameter", "Result"],

        ["Prediction", prediction],

        ["Confidence", f"{confidence:.2f}%"],

        ["Shape", analysis["Shape"]],

        ["Severity", analysis["Severity"]],

        ["Clinical Remark", analysis["Remark"]]
    ]

    table = Table(data, colWidths=[200, 250])

    table.setStyle(TableStyle([

        ('BACKGROUND', (0,0), (-1,0), colors.grey),

        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),

        ('GRID', (0,0), (-1,-1), 1, colors.black),

        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),

        ('BOTTOMPADDING', (0,0), (-1,0), 12),

    ]))

    elements.append(table)

    elements.append(Spacer(1, 20))

    # Save heatmap temporarily

    temp_img = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    )

    cv2.imwrite(temp_img.name, heatmap_image)

    # Add heatmap image

    elements.append(
        Paragraph(
            "<b>Grad-CAM Heatmap Visualization</b>",
            styles['Heading2']
        )
    )

    elements.append(Spacer(1, 10))

    pdf_img = PDFImage(
        temp_img.name,
        width=300,
        height=300
    )

    elements.append(pdf_img)

    # Build PDF

    doc.build(elements)

    return pdf_path

# ==========================================
# RUN PREDICTION
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array, original = preprocess_image(image)

    pred = model.predict(img_array)[0][0]

    st.subheader("Prediction Result")

    if pred < 0.5:

        confidence = (1 - pred) * 100

        st.success(f"Fibroid Detected ({confidence:.2f}%)")

        # GradCAM
        heatmap = make_gradcam_heatmap(
            img_array,
            model,
            "conv5_block3_out"
        )

        # Shape Analysis
        analysis = analyze_fibroid_shape(heatmap)

        st.subheader("Morphological Analysis")

        st.write(f"Shape: {analysis['Shape']}")
        st.write(f"Severity: {analysis['Severity']}")
        st.write(f"Clinical Remark: {analysis['Remark']}")

        # Heatmap Visualization

        heatmap_resized = cv2.resize(heatmap, (224, 224))

        heatmap_resized = np.uint8(255 * heatmap_resized)

        heatmap_colored = cv2.applyColorMap(
            heatmap_resized,
            cv2.COLORMAP_JET
        )

        superimposed_img = cv2.addWeighted(
            original,
            0.6,
            heatmap_colored,
            0.4,
            0
        )

        superimposed_img = cv2.cvtColor(
            superimposed_img,
            cv2.COLOR_BGR2RGB
        )

        st.subheader("Grad-CAM Heatmap")

        st.image(
            superimposed_img,
            use_container_width=True
        )

        # ======================================
        # GENERATE PDF REPORT
        # ======================================

        pdf_path = generate_pdf_report(
            prediction="Fibroid",
            confidence=confidence,
            analysis=analysis,
            heatmap_image=superimposed_img
        )

        # ======================================
        # DOWNLOAD BUTTON
        # ======================================

        with open(pdf_path, "rb") as pdf_file:

            st.download_button(
                label="📄 Download AI Report",
                data=pdf_file,
                file_name="Fibroid_AI_Report.pdf",
                mime="application/pdf"
            )

    else:

        confidence = pred * 100

        st.error(f"Non-Fibroid ({confidence:.2f}%)")