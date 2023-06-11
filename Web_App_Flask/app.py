import numpy as np
import pickle
from flask import Flask, render_template, request, url_for

# Initialize the Flask application
app = Flask(__name__, static_url_path='/static')
# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Default page of our web-app
# What we type into our browser to go to different pages
@app.route('/')
def home():
    return render_template('index.html')

# To use the predict button in our web-app
@app.route('/predict', methods=['POST'])
def predict():
    # For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = np.array(int_features).reshape(1, -1)
    
    # Make predictions using the loaded model
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    
    # Determine the view count range based on the prediction result
    if output == 0:
        view_count_range = '0-10K'
    elif output == 1:
        view_count_range = '10K-100K'
    elif output == 2:
        view_count_range = '100K-500K'
    elif output == 3:
        view_count_range = '500K-1M'
    elif output == 4:
        view_count_range = '1M-5M'
    elif output == 5:
        view_count_range = '5M-10M'
    elif output == 6:
        view_count_range = '10M-50M'
    elif output == 7:
        view_count_range = '50M-100M'
    elif output == 8:
        view_count_range = '100M-500M'
    else:
        view_count_range = '>500M'
    
    # Render the prediction result on a new page
    return render_template('index.html', prediction_text=view_count_range)


if __name__ == "__main__":
    app.run(debug=True)
