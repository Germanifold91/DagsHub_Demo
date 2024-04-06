"""Business logic for user app."""


def predict_with_mlflow(model, data):
    return model.predict(data)
