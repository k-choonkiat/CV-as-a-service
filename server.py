from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps
import numpy as np
import time
import matplotlib.pyplot as plt
from flask import Flask, render_template
import io

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")
model = None

if __name__ == "__main__":
    app.run(debug=True)