import os
import numpy as np
from flask import Flask, request, render_template
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

labels = {0: 'Daisy', 1: 'Dandelion', 2: 'Rose', 3: 'Sunflower', 4: 'Tulips'}

# Function to process image and make prediction
def getResult(image_path):
    img = load_img(image_path, target_size=(128, 128))
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions


# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# routes for other all vegetabels or fruits
@app.route('/corn')
def corn():
    return render_template('corn.html')
@app.route('/grapes')
def grapes():
    return render_template('grapes.html')
@app.route('/potato')
def potato():
    return render_template('potato.html')
@app.route('/tomato')
def tomato():
    return render_template('tomato.html')
@app.route('/wheat')
def wheat():
    return render_template('whaet.html')
@app.route('/apple')
def apple():
    return render_template('apple.html')


# Plant Disease Identifier Route
@app.route('/plant-disease', methods=['GET', 'POST'])
def plant_disease():
    if request.method == 'POST':
        f = request.files['file']
        if f:
            basepath = os.path.dirname(__file__)
            upload_dir = os.path.join(basepath, 'uploads')
            os.makedirs(upload_dir, exist_ok=True)  # Ensure upload directory exists
            
            file_path = os.path.join(upload_dir, secure_filename(f.filename))
            f.save(file_path)
            
            predictions = getResult(file_path)
            predicted_label = labels[np.argmax(predictions)]
            print(f'Predicted Disease: {predicted_label}')
            return predicted_label
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


if __name__ == '__main__':
    app.run(debug=True)
