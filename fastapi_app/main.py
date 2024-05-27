from fastapi import FastAPI
import joblib
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder


app = FastAPI()
model = joblib.load('../Models/knn_model.joblib')
scaler = joblib.load('../Models/scaler.joblib')

# Define a Pydantic model for input data validation
class InputFeatures(BaseModel):
    Year: int
    Engine_Size: float
    Mileage: float
    Type: str
    Make: str
    Options: str


def preprocessing(input_features: InputFeatures):
    dict_f = {
            'Year': input_features.Year,
            'Engine_Size': input_features.Engine_Size, 
            'Mileage': input_features.Mileage, 
            'Type_Accent': input_features.Type == 'Accent',
            'Type_Land Cruiser': input_features.Type == 'Land Cruiser',
            'Make_Hyundai': input_features.Make == 'Hyundai',
            'Make_Mercedes': input_features.Make == 'Mercedes',
            'Options_Full': input_features.Options == 'Full',
            'Options_Standard': input_features.Options == 'Standard'
        }

    # Convert dictionary values to a list in the correct order
    features_list = [dict_f[key] for key in sorted(dict_f)]
    
    # Scale the input features
    scaled_features = scaler.transform([list(dict_f.values())])
    return scaled_features

@app.get("/")
def root():
  return "Welcome To Tuwaiq Academy"


@app.get("/items/{item_id}")
async def read_item(item_id):
    return {"item_id": item_id}

@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model.predict(data)
    return {"pred": y_pred.tolist()[0]}