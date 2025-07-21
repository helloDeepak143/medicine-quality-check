# from flask import Flask, render_template, request
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load model, scaler, and encoders
# with open('medicine_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# with open('scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)

# with open('label_encoders.pkl', 'rb') as f:
#     label_encoders = pickle.load(f)

# # Home route (form)
# @app.route('/')
# def index():
#     return render_template('form.html')

# # Prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.form

#         active_ingredient = data['active_ingredient']
#         encoded_ingredient = label_encoders['Active Ingredient'].transform([active_ingredient])[0]

#         features = [
#             encoded_ingredient,
#             float(data['days_until_expiry']),
#             float(data['storage_temp']),
#             float(data['warning_labels']),
#             float(data['dissolution_rate']),
#             float(data['disintegration_time']),
#             float(data['impurity_level']),
#             float(data['assay_purity'])
#         ]

#         features_scaled = scaler.transform([features])
#         pred = model.predict(features_scaled)[0]
#         result = label_encoders['Safe/Not Safe'].inverse_transform([pred])[0]

#         return render_template('form.html', prediction=result)

#     except Exception as e:
#         return render_template('form.html', prediction=f"Error: {str(e)}")

# # Additional pages

# @app.route('/form')
# def form():
#     return render_template('form.html')

# @app.route('/about')
# def about():
#     return render_template('about.html')

# @app.route('/services')
# def services():
#     return render_template('services.html')

# @app.route('/contact')
# def contact():
#     return render_template('contact.html')

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model, scaler, and encoders
with open('medicine_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Home route
@app.route('/')
def index():
    return render_template('form.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        active_ingredient = data['active_ingredient']
        encoded_ingredient = label_encoders['Active Ingredient'].transform([active_ingredient])[0]

        features = [
            encoded_ingredient,
            float(data['days_until_expiry']),
            float(data['storage_temp']),
            float(data['warning_labels']),
            float(data['dissolution_rate']),
            float(data['disintegration_time']),
            float(data['impurity_level']),
            float(data['assay_purity'])
        ]

        features_scaled = scaler.transform([features])

        # Predict using model
        pred = model.predict(features_scaled)[0]
        print("Raw prediction:", pred)

        # Decode the prediction (Safe or Unsafe)
        result = label_encoders['Safe/Not Safe'].inverse_transform([pred])[0]
        print("Prediction label:", result)

        return render_template('form.html', prediction=result)

    except Exception as e:
        print("Error occurred:", e)
        return render_template('form.html', prediction=f"Error: {str(e)}")

# Extra pages
@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
