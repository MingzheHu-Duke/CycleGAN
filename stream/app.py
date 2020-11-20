import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

st.markdown("Monet Style Generator🎨")
image = st.file_uploader("Upload a photo to generate a Monet Painting!", type=['jpg'])
model = keras.models.load_model("./generator_g.h5")
if image is not None:
	IMAGE_SIZE = [256, 256]
	image = tf.image.decode_jpeg(image.read(), channels=3)
	image = tf.image.resize(image, [256,256])
	image = (tf.cast(image, tf.float32) / 127.5) - 1
	image = tf.reshape(image, [*IMAGE_SIZE, 3])
	prediction = tf.expand_dims(image,0)
	prediction = model(prediction, training=False)[0].numpy()
	prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
	fig, ax = plt.subplots(1, 2, figsize=(12, 12))
	image = (image * 127.5 + 127.5).numpy().astype(np.uint8)
	ax[0].imshow(image)
	ax[1].imshow(prediction)
	ax[0].set_title("Input Photo")
	ax[1].set_title("Monet Style Photo")
	ax[0].axis("off")
	ax[1].axis("off")
	st.pyplot(fig)
