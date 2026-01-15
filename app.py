print("APP IMPORT STARTED")

try:
    import cv2
    print("CV2 IMPORT OK")
except Exception as e:
    print("CV2 IMPORT FAILED:", e)

from flask import Flask, render_template, request
import cv2
import numpy as np
import os
import uuid
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "static/outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
   output_image = None

   if request.method == "POST":
       file = request.files.get("image")
       if file:
           # Load image
           image = Image.open(file.stream).convert("RGB")
           img = np.array(image)
           # ----------------------------------
           # 1. Convert to HSV
           # ----------------------------------
           hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
           # ----------------------------------
           # 2. Mask NON-background pixels
           # Fish = darker + more saturated
           # ----------------------------------
           lower = np.array([0, 40, 40])
           upper = np.array([180, 255, 220])
           mask = cv2.inRange(hsv, lower, upper)
           # ----------------------------------
           # 3. Morphology to clean mask
           # ----------------------------------
           kernel = np.ones((5, 5), np.uint8)
           mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
           mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
           # ----------------------------------
           # 4. Find contours
           # ----------------------------------
           contours, _ = cv2.findContours(
               mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
           )
           traced = img.copy()
           if contours:
               # Filter out circular objects (dish rim)
               fish_contour = None
               max_area = 0
               for c in contours:
                   area = cv2.contourArea(c)
                   if area < 500:  # ignore noise
                       continue
                   perimeter = cv2.arcLength(c, True)
                   if perimeter == 0:
                       continue
                   circularity = 4 * np.pi * area / (perimeter ** 2)
                   # Dish is VERY circular; fish is not
                   if circularity < 0.6 and area > max_area:
                       max_area = area
                       fish_contour = c
               if fish_contour is not None:
                   cv2.drawContours(
                       traced,
                       [fish_contour],
                       -1,
                       (0, 0, 0),   # black outline
                       thickness=4
                   )
           # ----------------------------------
           # 5. Save output
           # ----------------------------------
           filename = f"{uuid.uuid4().hex}.png"
           Image.fromarray(traced).save(os.path.join(UPLOAD_FOLDER, filename))
           output_image = filename

   return render_template("index.html", output_image=output_image)


@app.route("/team")
def team():
   return render_template("team.html")


@app.route("/innovation")
def innovation():
   return render_template("innovation.html")


@app.route("/video")
def video():
   return render_template("video.html")
