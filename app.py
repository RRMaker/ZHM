print("APP IMPORT STARTED")

from flask import Flask, render_template, request
import cv2
import numpy as np
import os
import uuid
from PIL import Image

app = Flask(__name__)

# Always resolve paths relative to this file (important on servers)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    output_image = None

    if request.method == "POST":
        # ---- Defensive checks (prevent 500 errors) ----
        if "image" not in request.files:
            return "No image uploaded", 400

        file = request.files["image"]

        if file.filename == "":
            return "Empty filename", 400

        try:
            # Load image safely
            image = Image.open(file.stream).convert("RGB")
            img = np.array(image)
        except Exception as e:
            print("Image load failed:", e)
            return "Invalid image file", 400

        # ----------------------------------
        # 1. Convert to HSV
        # ----------------------------------
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # ----------------------------------
        # 2. Mask NON-background pixels
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
            fish_contour = None
            max_area = 0

            for c in contours:
                area = cv2.contourArea(c)
                if area < 500:
                    continue

                perimeter = cv2.arcLength(c, True)
                if perimeter == 0:
                    continue

                circularity = 4 * np.pi * area / (perimeter ** 2)

                if circularity < 0.6 and area > max_area:
                    max_area = area
                    fish_contour = c

            if fish_contour is not None:
                cv2.drawContours(
                    traced,
                    [fish_contour],
                    -1,
                    (0, 0, 0),
                    thickness=4
                )

        # ----------------------------------
        # 5. Save output
        # ----------------------------------
        filename = f"{uuid.uuid4().hex}.png"
        output_path = os.path.join(UPLOAD_FOLDER, filename)

        Image.fromarray(traced).save(output_path)
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

