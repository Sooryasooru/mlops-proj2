from src.train import train_model
import os 

def test_model_training():
    model = train_model()
    assert model is not None
    assert os.path.exists("model.pkl")


