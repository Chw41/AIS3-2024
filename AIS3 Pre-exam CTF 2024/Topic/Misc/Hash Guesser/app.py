import io
import secrets
import hashlib
from os import getenv

from flask import Flask, request, jsonify, render_template
from flask_turnstile import Turnstile
from PIL import Image

import util

FLAG = getenv("FLAG", "flag{test}")
TURNSTILE_SITE_KEY = getenv("TURNSTILE_SITE_KEY")
TURNSTILE_SECRET_KEY = getenv("TURNSTILE_SECRET_KEY")

app = Flask(__name__, static_folder='static/')
app.config['JSON_AS_ASCII'] = False
app.config['MAX_CONTENT_LENGTH'] = 1024 # 1KB

app.config['TURNSTILE_SITE_KEY'] = TURNSTILE_SITE_KEY
app.config['TURNSTILE_SECRET_KEY'] = TURNSTILE_SECRET_KEY
app.config['TURNSTILE_ENABLED'] = TURNSTILE_SECRET_KEY is not None

turnstile = Turnstile(app)


def generate_image():
    h = hashlib.sha256(secrets.token_bytes(16)).hexdigest()

    image = Image.new("L", (16, 16), 0)
    pixels = [255 if c == '1' else 0 for c in bin(int(h, 16))[2:].zfill(256)]
    image.putdata(pixels)

    return image


def export_image(image, file_format):
    buffer = io.BytesIO()
    image.save(buffer, format=file_format)
    buffer.seek(0)
    return buffer


@app.route('/', methods=["GET"])
def index():
    return render_template("index.html")


@app.route('/guess', methods=["POST"])
def guess():
    if turnstile.is_enabled and not turnstile.verify():
        return jsonify({"error": "Invalid captcha"}), 400

    # compare uploaded image with the secret image
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400    

    try:
        uploaded_image = Image.open(file.stream)
        target_image = generate_image()

        if util.is_same_image(uploaded_image, target_image):
            return jsonify({"flag": FLAG})
        else:
            return jsonify({"error": "Wrong guess"}), 400
    except Exception as e:
        return jsonify({"error": f"{e.__class__.__name__}: {str(e)}"}), 500


if __name__ == "__main__":
    port = getenv("PORT", 7122)
    app.run(host="0.0.0.0", port=port)