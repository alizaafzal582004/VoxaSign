import os
# Force Keras 2 behavior BEFORE importing tensorflow
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflowjs as tfjs
import tensorflow as tf

# 1. Load the model using the legacy 'tf_keras' loader
try:
    import tf_keras
    print("⏳ Loading model using tf_keras (Legacy)...")
    model = tf_keras.models.load_model('signspeak_model.keras')
except ImportError:
    print("⏳ Loading model using standard tf.keras...")
    model = tf.keras.models.load_model('signspeak_model.keras')

# 2. Define the output folder
output_folder = 'web_model'

# 3. Perform the conversion
print(f"📦 Converting to TensorFlow.js format...")
tfjs.converters.save_keras_model(model, output_folder)

print(f"✅ Success! Your web model is in the '{output_folder}' folder.")