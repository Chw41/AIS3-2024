from flask import Flask, render_template, request, session, send_from_directory, redirect, render_template_string
import os
import subprocess
from secret import secret
import glob

app = Flask(__name__)
app.secret_key = secret
app.config['MAX_CONTENT_LENGTH'] = 2 * 1000 * 1000
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 60

if os.environ.get('FLAG'):
    FLAG = os.environ.get('FLAG')
else:
    FLAG = "AIS3{REDACTED}"

def create_user_image_dir():
    if session.get('id') == None:
        session['id'] = os.urandom(16).hex()
    if not os.path.exists(f'./image'):
        os.mkdir(f'./image')
    if not os.path.exists(f'./image/{session['id']}'):
        os.mkdir(f'./image/{session['id']}')
    return session['id']

def safe_join(_dir, _sub):
  filepath = os.path.join(_dir, _sub)
  realpath = os.path.realpath(filepath)
  if not _dir in os.path.commonpath((_dir, realpath)):
    return None
  return realpath

def list_files(dir):
    return [f.replace('./','') for f in glob.glob(dir+"/**", recursive=True) if not os.path.isdir(f)]

@app.route('/public_images')
def public_images():
    images = [f'/static/image/{img}' for img in os.listdir('./static/image')]
    return render_template('public_images.html', images=images)

@app.route('/upload', methods=['GET','POST'])
def upload():
    id = create_user_image_dir()
    if request.method == 'POST':
        file = request.files['zipfile']
        tmp = os.path.join('/tmp', os.urandom(16).hex())
        file.save(tmp)
        
        path = safe_join(f'{os.getcwd()}/image',id)
        if path == None:
            return render_template('upload.html', status="Your id is illegal!")
        
        ret = subprocess.run(['unzip', '-qo', tmp, '-d', path]).returncode
        if ret != 0:
            return render_template('upload.html', status="Your file is not ZIP!")
        files = list_files(path)
        for f in files:
            # only image
            ext = os.path.splitext(f)[1].lower()
            if ext != '.png' and ext != '.jpg' and ext != '.jpeg' and ext != '.gif':
                os.unlink(f)
            # no symlink
            if os.path.islink(f):
                os.unlink(f)
        
        return redirect("/my_images")
    else:
        return render_template('upload.html', status="")
    
@app.route('/my_images')
def my_images():
    id = create_user_image_dir()
    images = [f'/{img}' for img in list_files(f'./image/{id}')]
    return render_template('my_images.html', images=images)

@app.route('/image/<path:path>')
def image(path):
    return send_from_directory('image',path, cache_timeout=0)

@app.route('/admin')
def admin():
    if session.get('admin') == True:
        return FLAG
    else:
        return redirect('/')

@app.route('/')
def root():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=51414)