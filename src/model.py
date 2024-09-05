import joblib
import pandas as pd

def predict(model, data):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("A entrada deve ser um DataFrame.")
    try:
        predictions = model.predict(data)
    except Exception as e:
        raise RuntimeError(f"Erro ao fazer previsões: {e}")
    predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
    return predictions_df

def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar o modelo: {e}")




