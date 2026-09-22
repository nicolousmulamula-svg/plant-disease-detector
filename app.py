import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# ==========================================
# 1. SET PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Plant & Animal Disease Detector",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. LOAD MODEL NA DATA
# ==========================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('animal_disease_model_final.h5')
    return model

@st.cache_resource
def load_class_indices():
    with open('class_indices_final.json', 'r') as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}

@st.cache_resource
def load_disease_info():
    with open('disease_info.json', 'r') as f:
        disease_info = json.load(f)
    return disease_info

try:
    model = load_model()
    class_labels = load_class_indices()
    disease_info = load_disease_info()
    model_loaded = True
except Exception as e:
    st.error(f"Kuna tatizo la kupakia model: {e}")
    model_loaded = False

# ==========================================
# 3. SIDEBAR
# ==========================================
with st.sidebar:
    st.title("Plant & Animal Disease Detector")
    st.markdown("---")
    st.markdown("### Kuhusu App Hii")
    st.info(
        "App hii inatumia Deep Learning (MobileNetV2) kutambua magonjwa ya mimea "
        "na wanyama (kuku) kutoka kwenye picha. Ni kwa ajili ya majaribio tu."
    )
    st.markdown("### Jinsi ya Kutumia")
    st.markdown(
        "1. Pakia picha ya mmea au mnyama.\n"
        "2. Subiri model ichunguze.\n"
        "3. Pata matokeo na ushauri."
    )
    st.markdown("---")
    st.markdown("### Madarasa Yanayotambulika")
    if model_loaded:
        for label in class_labels.values():
            st.markdown(f"- {label}")

# ==========================================
# 4. MAIN CONTENT
# ==========================================
st.title("Plant & Animal Disease Detector")
st.markdown("### Upload picha ya mmea au mnyama ili kutambua ugonjwa na kupata ushauri")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Chagua picha (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model_loaded:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Picha Uliyopakia")
        image = Image.open(uploaded_file)
        st.image(image, caption="Picha Uliyopakia", use_container_width=True)
    
    # Convert picha kuwa RGB na float32
    img = image.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Fanya Prediction
    with st.spinner("Inachunguza picha..."):
        try:
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            predicted_class_name = class_labels[predicted_class_index]
        except Exception as e:
            st.error(f"Error wakati wa prediction: {e}")
            st.stop()
    
    with col2:
        st.markdown("#### Matokeo ya Uchunguzi")
        
        if confidence > 70:
            st.success(f"Ugonjwa: {predicted_class_name.upper()}")
        elif confidence > 40:
            st.warning(f"Ugonjwa: {predicted_class_name.upper()}")
        else:
            st.error(f"Ugonjwa: {predicted_class_name.upper()}")
        
        st.metric(label="Uhakika (Confidence)", value=f"{confidence:.2f}%")
        
        st.markdown("##### Uwezekano wa Kila Darasa:")
        for i, prob in enumerate(predictions[0]):
            st.progress(float(prob))
            st.write(f"{class_labels[i]}: {prob*100:.2f}%")
    
    # ==========================================
    # 5. MAELEZO YA UGONJWA (Case-Insensitive)
    # ==========================================
    st.markdown("---")
    st.markdown("## Maelezo ya Ugonjwa")
    
    predicted_key = None
    for key in disease_info.keys():
        if key.lower() == predicted_class_name.lower():
            predicted_key = key
            break
    
    if predicted_key:
        info = disease_info[predicted_key]
        
        st.markdown(f"### {info.get('jina_kamili', predicted_class_name)}")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Sababu", "Dalili", "Matibabu", "Kinga"])
        
        with tab1:
            st.info(info.get('sababu', 'Hakuna maelezo.'))
        
        with tab2:
            st.warning(info.get('dalili', 'Hakuna maelezo.'))
        
        with tab3:
            st.success(info.get('matibabu', 'Hakuna maelezo.'))
        
        with tab4:
            st.markdown(info.get('kinga', 'Hakuna maelezo.'))
    else:
        st.info("Maelezo ya ugonjwa huu hayapatikani kwa sasa.")

# ==========================================
# 6. FOOTER
# ==========================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "© 2024 Plant & Animal Disease Detector | Kwa ajili ya Majaribio tu"
    "</div>",
    unsafe_allow_html=True
)
