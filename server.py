
import tensorflow_hub as hub
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps
import numpy as np
import time
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file
import io
import requests
from methods import *

app = Flask(__name__)


def detect_img(image_url):
  start_time = time.time()
  image_path = download_and_resize_image(image_url, 640, 480)
  res_img = run_detector(detector, image_path)
  end_time = time.time()
  print("Inference time:",end_time-start_time)
  return res_img

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    path = request.form['imgName']
    
    image = detect_img(path)
    # return the data dictionary as a JSON response
    return render_template("index.html",picture=image)

if __name__ == "__main__":
    module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1" #@param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]

    detector = hub.load(module_handle).signatures['default']
    #resp = urlopen('https://github.com')
    #print(resp.read())
    image_urls = [
    # Source: https://commons.wikimedia.org/wiki/File:The_Coleoptera_of_the_British_islands_(Plate_125)_(8592917784).jpg
    "https://upload.wikimedia.org/wikipedia/commons/1/1b/The_Coleoptera_of_the_British_islands_%28Plate_125%29_%288592917784%29.jpg",
    # By Américo Toledano, Source: https://commons.wikimedia.org/wiki/File:Biblioteca_Maim%C3%B3nides,_Campus_Universitario_de_Rabanales_007.jpg
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Biblioteca_Maim%C3%B3nides%2C_Campus_Universitario_de_Rabanales_007.jpg/1024px-Biblioteca_Maim%C3%B3nides%2C_Campus_Universitario_de_Rabanales_007.jpg",
    # Source: https://commons.wikimedia.org/wiki/File:The_smaller_British_birds_(8053836633).jpg
    "https://upload.wikimedia.org/wikipedia/commons/0/09/The_smaller_British_birds_%288053836633%29.jpg",
    ]
    
    app.run(debug=True)