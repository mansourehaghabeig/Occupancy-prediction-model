from flask import Flask, jsonify
import pandas as pd
import joblib
import itertools
from datetime import datetime
app = Flask(__name__)



@app.route('/predict', methods=['GET'])
def prediction_ready():
    # make predictable
    current_time = datetime.now()
    # Make 24 predictions for each hour starting at the next full hour
    next_24_hours = pd.date_range(current_time, periods=24, freq='H').ceil('H')
    device_names = list(range(6))

    # produce 24 hourly slots per device:
    xproduct = list(itertools.product(next_24_hours, device_names))
    predictions = pd.DataFrame(xproduct, columns=['time', 'device'])
    predictions['hour'] = predictions['time'].dt.hour
    X = predictions[['device', 'hour']].values
    predictions['activation_predicted']  = clf.predict(X)
    predictions.drop('hour', axis=1, inplace=True)
    # Converting the device numerical data to categorical ones
    predictions["device"].replace(
        {0: "device_1", 1: "device_2", 2: "device_3", 3: "device_4", 4: "device_5", 5: "device_6", 6: "device_7"},
        inplace=True)
    result = predictions.to_json(orient="values")
    return jsonify({'prediction': result})

if __name__ == '__main__':
    clf = joblib.load('model/model.pkl')
    app.run(debug=False, host='0.0.0.0')



