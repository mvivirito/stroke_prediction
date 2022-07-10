# import the Flask class from the flask module
from flask import Flask, render_template, redirect, url_for, request

#Pandas and Matplotlib
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import app.predictor as predictor


from os.path import join

#other requirements
import io

# create the application object
app = Flask(__name__)

# create stroke data
target = join(app.static_folder, 'data/stroke-data.csv')
stroke_data = pd.read_csv(target)
table_data = stroke_data.head(30)


# use decorators to link the function to a url
@app.route('/')
def home():
    return redirect(url_for('login'))

# Route for handling the login page logic
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect(url_for('main'))
    return render_template('login.html', error=error)

@app.route('/main')
def main():
    return render_template('main.html') 


@app.route('/visual', methods=["POST", "GET"])
def visual():
    return render_template('visual.html',
                           PageTitle = "Visual",
                           table=[table_data.to_html(classes=["table-bordered", "table-striped", "table-hover",], index = False)]
                           )

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        gender = request.values["gender"]
        age = request.values["age"]
        hypertension = request.values["hypertension"]
        heartdisease = request.values["heartdisease"]
        married = request.values["married"]
        work_type = request.values["work_type"]
        residence = request.values["residence"]
        glucose = request.values["glucose"]
        bmi = request.values["bmi"]
        smoking = request.values["smoking"]


        predict_attr = {
                        "gender": gender,
                        "age": age,
                        "hypertension": hypertension,
                        "heartdisease": heartdisease,
                        "married" : married,
                        "work_type" : work_type,
                        "residence" : residence,
                        "glucose" : glucose,
                        "bmi" : bmi,
                        "smoking" : smoking
                        }

        ml_model = predictor.predictor(predict_attr, stroke_data)

        predicted = (ml_model["prediction"])
        score = (round(ml_model["score"], 3)) * 100

        return render_template("predict.html",predict_attr=predict_attr, score=score, prediction=predicted)



if __name__ == "__main__":
    app.run(debug=True)