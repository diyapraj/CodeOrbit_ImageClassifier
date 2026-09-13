# image classification project
# using a pretrained model called mobilenet, dont have to train anything myself
# googled how transfer learning works and this is basically using a model 
# thats already trained on tons of images (imagenet dataset) to guess whats in my pics

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os

print("loading model, might take a sec...")
model = MobileNetV2(weights="imagenet")


def classify(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # has to be 224x224 for this model
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)  # model wants a batch not just 1 image
    img_arr = preprocess_input(img_arr)

    preds = model.predict(img_arr, verbose=0)
    results = decode_predictions(preds, top=3)[0]  # get top 3 guesses
    return results


folder = "sample_images"

if not os.path.exists(folder):
    print("make a folder called sample_images and put some pics in it")
else:
    files = os.listdir(folder)
    files = [f for f in files if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg")]

    if len(files) == 0:
        print("no images found, add some jpg/png files to the folder")
    else:
        for f in files:
            path = folder + "/" + f
            result = classify(path)

            print("\nimage:", f)
            for i in range(len(result)):
                label = result[i][1]
                conf = result[i][2] * 100
                print(f"{i+1}. {label} - {conf:.2f}%")

print("\ndone. this used mobilenetv2 which is already trained on like a million+ images")
print("so instead of training my own model (which needs tons of data + time) i just")
print("reused this one to predict my images. this is called transfer learning")