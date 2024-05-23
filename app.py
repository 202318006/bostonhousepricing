# import pickle
# # from flask import Flask, request, 
# from flask import Flask,request,app,jsonify,url_for,render_template
# import numpy as np
# import pandas as pd

# app = Flask(__name__)
# regmodel = pickle.load(open('regmodel.pkl','rb'))
# scalar = pickle.load(open('scaling.pkl','rb'))


# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/predict_api',methods=['POST'])

# def predict_api():
#     data = request.json['data']
#     print(data)
#     print(np.array(list(data.values())).reshape(1,-1))
#     new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
#     output = regmodel.predict(new_data)[0]
#     print(output[0])
#     return jsonify(output)

# if __name__ == "__main__":
#     app.run(debug=True)
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    try:
        # Convert the input data to a numpy array and reshape
        new_data = np.array(list(data.values())).reshape(1, -1)
        print("New data:", new_data)

        # Scale the data
        new_data_scaled = scalar.transform(new_data)
        print("Scaled data:", new_data_scaled)

        # Make prediction
        output = regmodel.predict(new_data_scaled)
        print("Prediction output:", output)

        # Return the prediction as a JSON response
        return jsonify({'prediction': output[0]})
    except Exception as e:
        # Print the error and return it in the response
        print("Error:", str(e))
        return jsonify({'error': str(e)})

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)
