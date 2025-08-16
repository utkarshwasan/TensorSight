import os
import tensorflow as tf
from tensorflow import keras
import streamlit as st
import numpy as np

st.header('Image Classification Model')
# Load model from project directory with error handling
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Image_classify.keras')
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model from '{MODEL_PATH}'. {e}")
    st.stop()
data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']
img_height = 180
img_width = 180
image = st.text_input('Enter Image name', 'Apple.jpg')

try:
    image_load = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
    # Convert PIL image to array for model input
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr, 0)

    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)
    st.image(image, width=200)
    st.write('Veg/Fruit in image is ' + data_cat[np.argmax(score)])
    st.write('With accuracy of ' + f"{float(np.max(score)*100):.2f}%")
except FileNotFoundError:
    st.warning(f"Image '{image}' not found. Place it in '{os.getcwd()}' or provide a full path.")
except Exception as e:
    st.error(f"Error processing image: {e}")