
# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle

from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            dress_id = int(request.form['dress_id'])
            Style = request.form['Style']
            Price = request.form['Price']
            Rating = float(request.form['Rating'])
            Size = request.form['Size']
            Season = request.form['Season']
            NeckLine = request.form['NeckLine']
            SleeveLength = request.form['SleeveLength']
            waiseline = request.form['waiseline']
            Material = request.form['Material']
            PatternType = request.form['PatternType']
            total_sale = int(request.form['total_sale'])

            userInput = [[dress_id,Style,Price,Rating,Size,Season,NeckLine,SleeveLength,waiseline,Material,PatternType,total_sale]]
            filename = 'dressRecon_model.pickle'
            loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage

            #predictions using the loaded model file
            onehotencoder =  pickle.load(open('onehotencoder_model.pickle','rb')) # need to use as training model is onehot encoded
            prediction=loaded_model.predict(onehotencoder.transform(userInput))
            print('prediction is', prediction)
            # showing the prediction results in a UI
            if prediction[0] == 1:
                prediction1 = 'Yes'
            else:
                prediction1 = 'No'

            return render_template('results.html',prediction= prediction1)
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(host='0.0.0.0', debug=True) # running the app