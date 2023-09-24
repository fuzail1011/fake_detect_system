import argparse
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.models.load_model(r'C:\Users\Fuzail\detector_model.h5')

def prep(image_path):
    img = Image.open(image_path)
    img = img.resize((96, 96)) 
    img = np.array(img)
    img = img / 255.0  
    return img

def pred(image_path):
    img = prep(image_path)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction[0][0]

def main():
    parser = argparse.ArgumentParser(description='Image Classification')
    parser.add_argument('image_path', type=str, help='Path to the image file for classification')
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print("Error: The specified image file does not exist.")
        return

    prediction = pred(args.image_path)

    print(f"Prediction: {'Fake Image' if prediction > 0.5 else 'Real Image'}")

if __name__ == '__main__':
    main()