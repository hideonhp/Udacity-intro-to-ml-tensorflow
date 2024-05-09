import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json

import prediction_helpers

# # Force TensorFlow to use CPU only
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.set_visible_devices([], 'GPU')

def main():

  parser = argparse.ArgumentParser(
      description="Predict flower classes in an image."
  )
  parser.add_argument(
      "path_to_image",
      type=str,
      nargs=1,
      help="Path and name of the image",
  )
  parser.add_argument(
      "path_to_model",
      type=str,
      nargs=1,
      help="Path and name of the classification model",
  )
  parser.add_argument(
      "--top_k",
      metavar="top_k",
      type=int,
      nargs="?",
      default=1,
      help="Number of the top k most likely classes to return",
  )
  parser.add_argument(
      "--category_names",
      metavar="category_names",
      type=str,
      nargs="?",
      help="Path and name of a JSON file mapping labels to flower names",
  )

  args = parser.parse_args()

  try:
    # Attempt to load the model with default behavior
    model = tf.keras.models.load_model(filepath=args.path_to_model[0])
  except:
    # If loading fails, try loading with custom objects for compatibility
    model = tf.keras.models.load_model(
        filepath=args.path_to_model[0], compile=False, custom_objects={"KerasLayer": hub.KerasLayer}
    )

  # Use the prediction_helpers function with parsed arguments
  probabilities, classes = prediction_helpers.predict(
      args.path_to_image[0], model, args.top_k
  )

  if args.category_names is not None:
    # Load class names from JSON file if provided
    with open(args.category_names, "r") as f:
      class_names = json.load(f)
    classes = [class_names[k] for k in classes]

  print(f"\nTop {args.top_k} classes & probabilities:\n")
  for c, p in zip(classes, probabilities):
    print(f"{c}: {p:.3%}")
  print("\n")


if __name__ == "__main__":
  main()
