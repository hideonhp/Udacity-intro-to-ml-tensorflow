from PIL import Image
import numpy as np
import tensorflow as tf

def preprocess_image(input_image, target_size=224):

  # Convert the image to a NumPy array and copy to avoid modifying the original
  # image = np.asarray(input_image.copy())
  image = input_image.copy()

  # Cast the image data type to float32 for better performance with TensorFlow models
  image_array = tf.cast(image, tf.float32)

  # Resize the image to the target size using TensorFlow's image resizing function
  image_array = tf.image.resize(image_array, (target_size, target_size))

  # Normalize the pixel values by dividing by 255 (assuming image data is in uint8 range)
  image_array /= 255.0

  # Convert the image back to a NumPy array for compatibility with most models
  return image_array.numpy()

def predict(image_path, model, top_k=5):

  # Open the image using PIL
  image = Image.open(image_path)

  # Preprocess the image using the helper function
  preprocessed_image = preprocess_image(np.asarray(image))

  # Expand the dimension of the preprocessed image to create a batch of size 1
  # This is necessary for most TensorFlow models that expect batched input
  preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

  # Make predictions using the model
  predictions = model.predict(preprocessed_image)[0]

  # Extract the top K probabilities and their corresponding class indices
  top_k_probabilities = np.argpartition(predictions, -top_k)[-top_k:]
  predictions = predictions[top_k_probabilities]

  # Adjust class indices to start from 1 for human-friendliness (optional)
  top_k_probabilities += 1  # Uncomment this line if desired
  classes = top_k_probabilities.astype(str)

  return predictions, classes
