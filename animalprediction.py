from distutils.command.upload import upload
import cv2
import numpy as np
import streamlit as st
import time
import tensorflow as tf
from keras.preprocessing import image # dont use tensorflow before keras
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input

@st.cache(allow_output_mutation=True)
def loadModel():
    model = tf.keras.models.load_model("mdl_wts.hdf5")
    return model

model = loadModel()

# Load file

uploaded_file = st.file_uploader("Choose a image file",type='jpg')

map_dict = {0: 'dog',
            1: 'horse',
            2: 'elephant',
            3: 'butterfly',
            4: 'chicken',
            5: 'cat',
            6: 'cow',
            7: 'sheep',
            8: 'spider',
            9: 'squirrel'}

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:

        progress = st.progress(0) # intialize with 0
        for i in range(100):
            time.sleep(0.1)
            progress.progress(i+1)

        prediction = model.predict(img_reshape).argmax()
        st.title("Predicted Label for the image is {}".format(map_dict [prediction]))