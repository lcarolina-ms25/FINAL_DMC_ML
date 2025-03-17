
# Cargar librerías necesarias
import pandas as pd
from pycaret.classification import *

# Cargar el dataset
file_path = "Data_Taller_MLE.csv"  
columns = ["id", "age", "income", "loan_amount", "loan_term", "credit_score", "employment_status", "loan_purpose", "existing_loans", "income_variability", "loan_status"]

# Cargar los datos con los nombres correctos de las columnas
data = pd.read_csv(file_path, names=columns, header=0)

# Convertir columnas categóricas a tipo adecuado
data["employment_status"] = data["employment_status"].astype("category")
data["loan_purpose"] = data["loan_purpose"].astype("category")
data["income_variability"] = data["income_variability"].astype("category")
data["loan_status"] = data["loan_status"].astype("int")  # Variable objetivo

# Mostrar las primeras filas
data.head()



# Configurar el experimento en PyCaret
exp_clf = setup(data=data,
                target="loan_status",
                categorical_features=["employment_status", "loan_purpose", "income_variability"],
                numeric_features=["age", "income", "loan_amount", "loan_term", "credit_score", "existing_loans"],
                session_id=123,
                normalize=True,  
                transformation=True, 
                data_split_shuffle=True,
                fix_imbalance=True,
                verbose=False)



# Comparar modelos y seleccionar el mejor
best_model = compare_models()



# Registrar el modelo en MLflow
from pycaret.classification import save_model
save_model(best_model, "best_loan_model")


#creamos un modelo rf
rf_model= create_model('rf')
#mostramos un resumen del modelo
print(rf_model)


#Optimización de hiperparametros

tuned_rf= tune_model(rf_model)

print(tuned_rf)

#Revisamos la CURVA ROC
plot_model(tuned_rf,plot='auc')

#matriz de confusion
plot_model(tuned_rf,plot='confusion_matrix')

#Evaluamos el modelo:
evaluate_model(tuned_rf)

#Finalizando el modelo entrenandolo en todo el dataset
final_rf = finalize_model(tuned_rf)

#Realizar la predicción sobre el mismo dataset
predictions = predict_model(final_rf,data=data)

print(predictions.head())


save_model(final_rf,'final_rf_model')



loaded_model= load_model('final_rf_model')

prediction_loaded = predict_model(loaded_model,data=data)
print(prediction_loaded.head())

import seaborn as sns
import matplotlib.pyplot as plt

# Ver la distribución de loan_status según income_variability
sns.countplot(x="income_variability", hue="loan_status", data=data)
plt.title("Impacto de la Variabilidad de Ingresos en la Aprobación de Préstamos")
plt.show()


import mlflow
import mlflow.sklearn
from pycaret.classification import *
import pandas as pd
import joblib
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import uvicorn
import logging

# Configurar MLflow
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Loan_Classification")

# Inicializar FastAPI
app = FastAPI()

# Seguridad básica para proteger las rutas
security = HTTPBasic()
users = {"admin": "password123"}

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if users.get(credentials.username) != credentials.password:
        raise HTTPException(status_code=401, detail="Acceso denegado, credenciales incorrectas")
    return credentials.username

# Ruta para entrenar el modelo
@app.post("/train")
def train_model(user: str = Depends(authenticate)):
    try:
        logging.info("Entrenamiento iniciado por el usuario: %s", user)
        # Cargar el dataset
        file_path = "Data_Taller_MLE.csv"
        columns = ["id", "age", "income", "loan_amount", "loan_term", "credit_score", "employment_status", "loan_purpose", "existing_loans", "income_variability", "loan_status"]
        data = pd.read_csv(file_path, names=columns, header=0)
        
        # Configurar PyCaret
        exp_clf = setup(data=data,
                        target="loan_status",
                        categorical_features=["employment_status", "loan_purpose", "income_variability"],
                        numeric_features=["age", "income", "loan_amount", "loan_term", "credit_score", "existing_loans"],
                        session_id=123,
                        normalize=True,
                        transformation=True,
                        data_split_shuffle=True,
                        fix_imbalance=True,
                        verbose=False)
        
        # Entrenar y guardar el modelo
        best_model = create_model('rf')
        final_model = finalize_model(best_model)
        joblib.dump(final_model, "final_rf_model.pkl")

        # Registrar en MLflow
        with mlflow.start_run():
            mlflow.sklearn.log_model(final_model, "rf_model")
        
        return {"message": "Modelo entrenado y guardado correctamente"}
    
    except Exception as e:
        return {"error": str(e)}

# Ruta para realizar predicciones
@app.post("/predict")
def predict(data: dict, user: str = Depends(authenticate)):
    try:
        # Cargar el modelo entrenado
        model = joblib.load("final_rf_model.pkl")
        
        # Convertir entrada a DataFrame
        df = pd.DataFrame([data])
        
        # Realizar la predicción
        prediction = model.predict(df)
        
        return {"loan_status_prediction": int(prediction[0])}
    
    except Exception as e:
        return {"error": str(e)}

# Para ejecutar la API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
