import os
from flask import Flask, request
import pickle
import pandas as pd

#carregando modelo
model = pickle.load(open('model/model_RendaPB.pkl', 'rb'))

#Instanciando o Flask
app = Flask(__name__)

@app.route('/predict', methods = ['POST'])
def predict():
    test_json = request.get_json()
    
    # Coleta de dados
    if test_json:
        if isinstance(test_json, dict): #unique value
            df_raw = pd.DataFrame(test_json, index = [0])
        else:
            df_raw = pd.DataFrame(test_json, columns = test_json[0].keys())
    # predição
    pred = model.predict(df_raw)
    
    df_raw['prediction'] = pred
    
    return df_raw.to_json(orient = 'records')

if __name__ == '__main__':
    #iniciando o flask
    port = os.environ.get('PORT', 5000)
    app.run(host = '0.0.0.0', port=port)