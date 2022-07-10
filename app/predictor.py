from matplotlib.transforms import Transform
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def predictor(attr, stroke_data):
    ### split data

    # create stroke data
    stroke_data.dropna(inplace=True)

    X = stroke_data.drop(["id", "stroke"], axis=1)
    y = stroke_data["stroke"]


    ### train data
    model = RandomForestClassifier()

    # Define our category features
    category_features = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    one_hot = OneHotEncoder()
    transformer = ColumnTransformer([("one_hot",
                                    one_hot, 
                                    category_features)],
                                remainder="passthrough")

    transformed_X = transformer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=0.2)

    model.fit(X_train, y_train)

    new_data = pd.DataFrame(
    { 'gender': "Female",
      'age': 67.0,
      'hypertension' : 0,
      'heart_disease' : 1,
      'ever_married' : "Yes",
      'work_type' : "Private",
      'Residence_type' : "Urban",
      'avg_glucose_level' : 68.69,
      'bmi': 36.6,
      'smoking_status' : "formerly smoked"
    }, index=[0])

    transformed_new_data = transformer.transform(new_data)


    ### evaluate data
    score = model.score(X_test, y_test)

    ### add user's input
    #our_data = pd.read_csv("data/test-data.csv")
    #our_data["humidity"] = int(attr["humidity"])
    #our_data["temp"] = int(attr["temp"])
    #our_data["weather"] = int(attr["weather"])

    ### predict patients stroke likelyhood
    prediction = model.predict(transformed_new_data)
    ml_model =  {"score": score , "prediction": prediction}

    return ml_model