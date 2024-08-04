import tempfile
import pathlib
import secrets

from os import getenv, path

import ebookmeta

from flask import Flask, request, jsonify
from flask.helpers import send_from_directory

app = Flask(__name__, static_folder='static/')
app.config['JSON_AS_ASCII'] = False
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024

@app.route('/', methods=["GET"])
def index():
    return send_from_directory('static', 'index.html')


@app.route('/parse', methods=["POST"])
def upload():
    if 'ebook' not in request.files:
        return jsonify({'error': 'No File!'})

    file = request.files['ebook']

    with tempfile.TemporaryDirectory() as directory:
        suffix = pathlib.Path(file.filename).suffix
        fp = path.join(directory, f"{secrets.token_hex(8)}{suffix}")
        file.save(fp)
        app.logger.info(fp)

        try:
            meta = ebookmeta.get_metadata(fp)
            return jsonify({'message': "\n".join([
                f"Title: {meta.title}",
                f"Author: {meta.author_list_to_string()}",
                f"Lang: {meta.lang}",
            ])})
        except Exception as e:
            print(e)
            return jsonify({'error': f"{e.__class__.__name__}: {str(e)}"}), 500


if __name__ == "__main__":
    port = getenv("PORT", 8888)
    app.run(host="0.0.0.0", port=port)