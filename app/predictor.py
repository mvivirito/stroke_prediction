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
    { 'gender': attr["gender"],
      'age': int(attr["age"]),
      'hypertension' : int(attr["hypertension"]),
      'heart_disease' : int(attr["heartdisease"]),
      'ever_married' : attr["married"],
      'work_type' : attr["work_type"],
      'Residence_type' : attr["residence"],
      'avg_glucose_level' : int(attr["glucose"]),
      'bmi': int(attr["bmi"]),
      'smoking_status' : attr["smoking"]
    }, index=[0])

    transformed_new_data = transformer.transform(new_data)


    ### evaluate data
    score = model.score(X_test, y_test)


    ### predict patients stroke likelyhood
    prediction = model.predict(transformed_new_data)
    ml_model =  {"score": score , "prediction": prediction}

    return ml_model