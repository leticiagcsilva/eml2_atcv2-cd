import pytest
import pandas as pd
import numpy as np
import os

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model import predict


class MockModel:
    def predict(self, data):
        return np.array([1.0] * len(data))

@pytest.fixture
def sample_model():
    model = MockModel()
    model_path = 'models/model.pkl' 
    with open(model_path, 'wb') as file:
        import pickle
        pickle.dump(model, file)
    
    yield model

    # Cleanup
    os.remove(model_path)

def test_predict_valid_data(sample_model):
    valid_data = pd.DataFrame({
        '7-Day MA': [75.0, 76.0, 77.0]
    })
    
    predictions = predict(sample_model, valid_data)
    
    print(f"Tipo de previsões: {type(predictions)}")
    print(f"Conteúdo das previsões: {predictions}")
