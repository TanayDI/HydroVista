from flask import Flask, jsonify, request
from flask_cors import CORS
from torch.functional import split
from model_files.ml_predict import predict_plant, Network
from pyfcm import FCMNotification
import base64
from decouple import config
import base64


app = Flask("Plant Disease Detector")
CORS(app)
# push_service = FCMNotification(api_key=config('API_KEY'))

@app.route('/', methods=['POST'])
def predict():
    key_dict = request.get_json()
    image = key_dict["image"]
    imgdata = base64.b64decode(image)
    model = Network()
    result, remedy, nutrients = predict_plant(model, imgdata)
    
    if result is None:
        response = {
            "error": "Input image does not match any known plant class. Please try with a valid plant image."
        }
        return jsonify(response), 400
    
    plant = result.split("___")[0]
    disease = " ".join((result.split("___")[1]).split("_"))
    
    response = {
        "plant": plant,
        "disease": disease,
        "remedy": remedy,
        "nutrients": nutrients,
    }
    response = jsonify(response)
    return response


# with open("backend/model_files/test/TomatoEarlyBlight3.JPG", "rb") as image_file:
#     base64_string = base64.b64encode(image_file.read()).decode('utf-8')
#     print(base64_string)

@app.route('/notification', methods=['POST'])
def send_notification():
    print("Alert came")
    try:
        key_dict = request.get_json()
        plant = key_dict['plant']
        disease = key_dict['disease']
        user = key_dict['user']

        message = "Plant " + plant + " diagnosed with " + disease + " recently by " + user
        print(message)

        data_message = {
            "title": "HydroVista Alerts",
            "body": message,
        }

        result = push_service.notify_topic_subscribers(
            data_message=data_message, topic_name="HydroVista")

        print(result)
        # location = key_dict['location']
        return "Success"

    except Exception as e:
        print(e)
        return "Failed"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080, threaded=True, use_reloader=False)
