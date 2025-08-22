import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Input

inputs = Input(shape=(224, 224, 3))
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_tensor=inputs)
print("EfficientNetB0 loaded successfully!")