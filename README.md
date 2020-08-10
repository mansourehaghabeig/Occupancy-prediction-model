#  Interview challenge - WATTx Data Science candidates

A simple Flask application that can serve predictions from a scikit-learn model.
Reads a pickled sklearn model into memory when the Flask app is started and returns predictions through the /predict endpoint for the next 24 hours.
 

# 1) Occupancy prediction model


### Implementation details 
sample_solution consists of the below main functions: <br/>
* prepare_input_data: Preparing the orginal data in the way that add the missing values for missing hours of days for different devices and also consider for several report of one device in one hour in 
a specific date just one entry in dataset. For example if we have the following report for device 2 in dataset: <br/>
2016-07-01 06:52:57,device_2,1 <br/>
2016-07-01 06:53:00,device_2,1 <br/>
We consider one entry for hour 6 in 2016-07-01 for device 2. <br/>
* find_best_classifier: Finding the best classifier among ten different classifiers for our dataset and save it as pickled sklearn model into memory when the Flask app can load it.
* predict_future_activation: Preparing the dataset for future prediction of all devices in 24 hours.
    ./sample_solution.py <timestamp> <input file csv> <output file csv>
* prediction = Implementing of prediction over the future prediction dataset based on the best found classifier.


### Dependencies

    * scikit-learn
    * Flask
    * pandas
    * numpy
### Running API
python app.py 

### Endpoint
####/predict (POST)
Returns the predictions of room accupacy for the next 24 hours of current time given a JSON object representing future_predicted dataset.

