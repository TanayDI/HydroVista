from flask import Flask, jsonify, request
from flask_cors import CORS
from model_files.ml_predict import predict_plant, Network
from pyfcm import FCMNotification
import base64
import torch
from decouple import config
from googletrans import Translator  # Import googletrans for translation

app = Flask("Plant Disease Detector")
CORS(app)

# Initialize Google Translator
translator = Translator()

# Initialize push notifications if needed
# push_service = FCMNotification(api_key=config('API_KEY'))

# Initialize model once
model = Network()
try:
    model.load_state_dict(torch.load("backend/model_files/model.pth"))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError("Failed to initialize the model.")

def translate_text(text, language_code):
    """
    Translates the given text to the specified language.
    Returns the original text if translation fails or input is invalid.
    """
    try:
        if not text or not isinstance(text, str):
            return "Invalid input for translation"

        # Validate language code
        valid_languages = ['en', 'mr']  
        if language_code not in valid_languages:
            language_code = 'en'

        return translator.translate(text, src='en', dest=language_code).text
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Fallback to the original text

@app.route('/', methods=['POST'])
def predict():
    """
    Predict plant disease and provide remedy, nutrients information.
    Allows translation of the output based on the specified language.
    """
    try:
        key_dict = request.get_json()
        image = key_dict["image"]
        language = key_dict.get("language", "en") 

        # Decode the image
        imgdata = base64.b64decode(image)

        # Get prediction results
        result, remedy, nutrients = predict_plant(model, imgdata)

        if result is None:
            error_message = "Input image does not match any known plant class. Please try with a valid plant image."
            return jsonify({"error": translate_text(error_message, language)}), 400

        # Extract plant and disease information
        plant, disease = result.split("___")[0], " ".join(result.split("___")[1].split("_"))

        response = {
            "plant": plant,
            "disease": disease,
            "remedy": remedy,
            "nutrients": nutrients,
        }

        if language == "mr":
            response["plant"] = translate_text(plant, "mr")
            response["disease"] = translate_text(disease, "en")
            response["remedy"] = translate_text(remedy, "mr")
            response["nutrients"] = translate_text(nutrients, "mr")

        return jsonify(response)
    
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return jsonify({"error": translate_text(error_message, "en")}), 500

@app.route('/notification', methods=['POST'])
def send_notification():
    """
    Send a notification about the diagnosed plant and disease.
    Allows message translation based on the specified language.
    """
    try:
        key_dict = request.get_json()
        plant = key_dict['plant']
        disease = key_dict['disease']
        user = key_dict['user']
        language = key_dict.get('language', 'en')  # Language for the notification

        # Construct the message
        message = f"{plant} diagnosed with {disease} recently by {user}"
        message = translate_text(message, language)

        data_message = {
            "title": translate_text("HydroVista Alerts", language),
            "body": message,
        }

        # Ensure push_service is initialized
        # if 'push_service' in globals():
        #     result = push_service.notify_topic_subscribers(
        #         data_message=data_message, topic_name="HydroVista")
        #     return jsonify({"status": "Success", "result": result}), 200
        # else:
        #     return jsonify({"error": "Push service not initialized"}), 400

    except Exception as e:
        error_message = f"Failed to send push notification: {str(e)}"
        return jsonify({"error": translate_text(error_message, language)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080, threaded=True, use_reloader=False)
