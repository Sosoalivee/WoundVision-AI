from flask import Flask, request, render_template, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import base64

app = Flask(__name__)
app.secret_key = "dev"  # needed if you use flash

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



# Get the base directory (one level up from api folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(BASE_DIR, "runs", "wound_cls4", "weights", "best.pt")

model = YOLO(WEIGHTS_PATH)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    print(">>>> ENTER ROUTE, METHOD =", request.method, flush=True)

    if request.method == 'POST':
        print(">>>> IN POST HANDLER", flush=True)
        print("request.files keys:", list(request.files.keys()), flush=True)

        if 'file' not in request.files:
            print(">>>> 'file' NOT in request.files", flush=True)
            return "No file part", 400  # TEMP: return text instead of redirect

        file = request.files['file']
        print(">>>> Got file:", repr(file.filename), flush=True)

        if file.filename == '':
            print(">>>> Empty filename", flush=True)
            return "No selected file", 400  # TEMP

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(">>>> Saving to:", filepath, flush=True)
            file.save(filepath)




# -------- YOLO + result.html from here ----------
            try:
                print(">>>> Running model.predict...", flush=True)
                results = model.predict(filepath, save=False, verbose=True)
                print(">>>> Prediction done", flush=True)

                prediction = "No prediction"
                confidence = 0.0
                all_probs = {}

                for result in results:
                    probs = result.probs
                    print(">>>> probs:", probs, flush=True)
                    if probs is not None:
                        predicted_class_idx = probs.top1
                        confidence = float(probs.top1conf)
                        class_names = model.names
                        prediction = class_names[predicted_class_idx]

                        for i, prob in enumerate(probs.data):
                            all_probs[class_names[i]] = float(prob)

                with open(filepath, "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

                print(">>>> Rendering result.html", flush=True)
                return render_template(
                    'result.html',
                    prediction=prediction,
                    confidence=confidence,
                    all_probs=all_probs,
                    image_data=img_base64,
                    filename=filename
                )

            except Exception as e:
                print(">>>> ERROR in prediction:", e, flush=True)
                return f"Error during prediction: {e}", 500

        print(">>>> File type not allowed", flush=True)
        return "File type not allowed", 400

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)








