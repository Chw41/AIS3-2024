from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    expression = data['expression'].replace(" ","").replace("_","")
    try:
        result = eval(expression)
    except Exception as e:
        result = str(e)
    return jsonify(result=str(result))

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run("0.0.0.0",5001)
