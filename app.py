import os
import numpy as np
import time
from flask import Flask, request , redirect, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

app = Flask(__name__)

# Load trained AI model for plant disease detection
MODEL_PATHS = {
    'Apple': 'models/Apple_Disease.h5',
    'Corn': 'models/Corn_Disease.h5',
    'Grapes': 'models/Grapes_Disease.h5',
    'Potatos': 'models/Potato_Disease2.h5',
    'Tomato': 'models/Tomato_Disease.h5',
    'Wheat': 'models/Wheat_Disease.h5',
    # Add more models as needed
}

models = {}

for crop, path in MODEL_PATHS.items():
    print(f"Loading model for {crop}...")
    models[crop] = load_model(path)
print("All models loaded successfully!")
print('Model loaded. Check http://127.0.0.1:5000/')

apple_labels = {0: 'Apple__Apple_scab', 1: 'Apple_Black_rot', 2: 'Apple_Cedar_apple_rust', 3: 'Apple__healthy'}
corn_labels = {0: 'Corn__Common_Rust', 1: 'Corn_Gray_Leaf_Spot', 2: 'Corn_Healthy', 3: 'Corn__Northern_Leaf_Blight'}
pototo_labels = {0: 'Potato__Early_Blight', 1: 'Potato_Healthy', 2: 'Potato__Late_Blight'}
tomato_labels = {0: 'Tomato__Bacterial_spot', 1: 'Tomato_Early_blight', 2: 'Tomato_Late_blight', 3: 'Tomato_Leaf_Mold',
           4: 'Tomato_Septoria_leaf_spot', 5: 'Tomato_Spider_mites_Two_spotted_spider_mite', 6: 'Tomato_Target_Spot',
           7: 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 8: 'Tomato_Tomato_mosaic_virus', 9: 'Tomato__healthy'}
grapes_labels = {0: 'Grape__Black_rot', 1: 'Grape_Esca(Black_Measles)', 2: 'Grape__Leaf_blight', 3: 'Grape__healthy'}
wheat_labels = {0: 'Wheat__Brown_Rust', 1: 'Wheat_Healthy', 2: 'Wheat__Yellow_Rust'}

# Function to process image and make prediction
def predict_disease(model,image_path, labels):
    """Loads an image, preprocesses it, and returns the predicted disease label."""
    img = load_img(image_path, target_size=(128, 128))
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    
    predictions = model.predict(x)[0]
    predicted_label = labels[np.argmax(predictions)]
    print(predicted_label)
    return predicted_label


# Home Route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        language = request.form.get('language')
        if language == 'marathi':
            translation = "मराठी"
        elif language == 'english':
            translation = "English"
        elif language == 'hindi':
            translation = "हिंदी"
        else:
            translation = "English"
    else:
        translation = "Select a language" 
    return render_template('index.html', translation=translation) 


@app.route('/greetings')
def greetings():
    return render_template('greetings.html')

@app.route('/language')
def select_language():
    return render_template('langselection.html')

# routes for other all vegetabels or fruits
@app.route('/corn', methods=['GET', 'POST'])
def corn():
    prediction = None  # Initialize prediction to None at the beginning

    if request.method == 'POST':
        f = request.files.get('image')  # Use `.get()` to avoid errors if 'image' is missing

        if f and f.filename != '':  # Ensure a file is uploaded
            basepath = os.path.dirname(__file__)
            upload_dir = os.path.join(basepath, 'uploads', 'corn')
            os.makedirs(upload_dir, exist_ok=True)  # Ensure crop-specific upload folder exists

            file_path = os.path.join(upload_dir, secure_filename(f.filename))
            f.save(file_path)

            # Predict using Corn model
            prediction = predict_disease(model=models['Corn'], image_path=file_path, labels=corn_labels)

    return render_template('corn.html', prediction=prediction)


@app.route('/grapes' , methods=['GET', 'POST'])
def grapes():
    prediction = None  # Initialize prediction to None at the beginning

    if request.method == 'POST':
        f = request.files.get('image')  # Use `.get()` to avoid errors if 'image' is missing

        if f and f.filename != '':  # Ensure a file is uploaded
            basepath = os.path.dirname(__file__)
            upload_dir = os.path.join(basepath, 'uploads', 'grapes')
            os.makedirs(upload_dir, exist_ok=True)  # Ensure crop-specific upload folder exists

            file_path = os.path.join(upload_dir, secure_filename(f.filename))
            f.save(file_path)

            # Predict using Corn model
            prediction = predict_disease(model=models['Grapes'], image_path=file_path, labels=grapes_labels)
    return render_template('grapes.html', prediction=prediction)

@app.route('/potato' , methods=['GET', 'POST'])
def potato():
    prediction = None  # Initialize prediction to None at the beginning

    if request.method == 'POST':
        f = request.files.get('image')  # Use `.get()` to avoid errors if 'image' is missing

        if f and f.filename != '':  # Ensure a file is uploaded
            basepath = os.path.dirname(__file__)
            upload_dir = os.path.join(basepath, 'uploads', 'potato')
            os.makedirs(upload_dir, exist_ok=True)  # Ensure crop-specific upload folder exists

            file_path = os.path.join(upload_dir, secure_filename(f.filename))
            f.save(file_path)

            # Predict using Corn model
            prediction = predict_disease(model=models['Potatos'], image_path=file_path, labels=pototo_labels)
    return render_template('potato.html', prediction=prediction)
@app.route('/tomato' , methods=['GET', 'POST'])
def tomato():
    prediction = None  # Initialize prediction to None at the beginning

    if request.method == 'POST':
        f = request.files.get('image')  # Use `.get()` to avoid errors if 'image' is missing

        if f and f.filename != '':  # Ensure a file is uploaded
            basepath = os.path.dirname(__file__)
            upload_dir = os.path.join(basepath, 'uploads', 'tomato')
            os.makedirs(upload_dir, exist_ok=True)  # Ensure crop-specific upload folder exists

            file_path = os.path.join(upload_dir, secure_filename(f.filename))
            f.save(file_path)

            # Predict using Corn model
            prediction = predict_disease(model=models['Tomato'], image_path=file_path, labels=tomato_labels)
    return render_template('tomato.html', prediction=prediction)
@app.route('/wheat' , methods=['GET', 'POST'])
def wheat():
    prediction = None  # Initialize prediction to None at the beginning

    if request.method == 'POST':
        f = request.files.get('image')  # Use `.get()` to avoid errors if 'image' is missing

        if f and f.filename != '':  # Ensure a file is uploaded
            basepath = os.path.dirname(__file__)
            upload_dir = os.path.join(basepath, 'uploads', 'wheat')
            os.makedirs(upload_dir, exist_ok=True)  # Ensure crop-specific upload folder exists

            file_path = os.path.join(upload_dir, secure_filename(f.filename))
            f.save(file_path)

            # Predict using Corn model
            prediction = predict_disease(model=models['Wheat'], image_path=file_path, labels=wheat_labels)
    return render_template('wheat.html', prediction=prediction)
@app.route('/apple' , methods=['GET', 'POST'])
def apple():
    prediction = None  # Initialize prediction to None at the beginning

    if request.method == 'POST':
        f = request.files.get('image')  # Use `.get()` to avoid errors if 'image' is missing

        if f and f.filename != '':  # Ensure a file is uploaded
            basepath = os.path.dirname(__file__)
            upload_dir = os.path.join(basepath, 'uploads', 'apple')
            os.makedirs(upload_dir, exist_ok=True)  # Ensure crop-specific upload folder exists

            file_path = os.path.join(upload_dir, secure_filename(f.filename))
            f.save(file_path)

            # Predict using Corn model
            prediction = predict_disease(model=models['Apple'], image_path=file_path, labels=apple_labels)
    return render_template('apple.html', prediction=prediction)
@app.route('/mango')
def mango():
    return render_template('mango.html')


# Plant Disease Identifier Route
@app.route('/plant-disease', methods=['GET', 'POST'])
def plant_disease():
    return render_template('plant_disease.html')


# Weather Page Route

@app.route('/weather')
def weather():
    weather_data = {
        "city": "Timisoara",
        "date": "Thursday 12 May 2022 17:30",
        "temp": 20,
        "feels_like": 21,
        "condition": "Few clouds",
        "humidity": "50%",
        "pressure": "1018hPa",
        "wind": "2 m/s",
        "rain_chance": "0.00%",
        "clouds": "20%",
        "sunrise": "08:36",
        "sunset": "23:26",
        "hourly_forecast": [
            {"hour": "12:00", "temp": 21},
            {"hour": "13:00", "temp": 21},
            {"hour": "14:00", "temp": 22},
        ],
        "daily_forecast": [
            {"day": "Friday", "condition": "Moderate rain", "low": 15, "high": 28},
            {"day": "Saturday", "condition": "Sunny", "low": 13, "high": 24},
        ]
    }
    return render_template('weather.html', **weather_data)


# Chatbot Page Route
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')


# Feedback Page Route
@app.route('/feedback', methods=['GET', 'POST'])
def submit_feedback():
    if request.method == 'POST':
        feedback_text = request.form.get('feedback')
        print("User Feedback:", feedback_text)  # Replace with database storage logic
        return redirect(url_for('submit_feedback'))
    return render_template('feedback.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('notfound.html'), 404  # Ensure you return a status code

if __name__ == '__main__':
    app.run(debug=True)