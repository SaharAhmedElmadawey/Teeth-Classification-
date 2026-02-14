import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load model
model = tf.keras.models.load_model("teeth_model.keras")

class_names = ['OLP', 'MC', 'Gum', 'CoS', 'OT', 'CaS', 'OC']


def predict(image):
    image = image.resize((224, 224))
    img_array = preprocess_input(np.array(image))
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    return class_names[class_index]


interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Teeth Classification System",
    description="Upload a dental image to classify it into one of 7 categories."
)

interface.launch()
