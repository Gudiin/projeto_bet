import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

class CornerPredictor:
    def __init__(self, model_path="data/corner_model.pkl"):
        self.model_path = model_path
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Treinando modelo...")
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Modelo treinado! MAE: {mae:.2f}, R2: {r2:.2f}")
        
        self.save_model()
        return mae, r2

    def predict(self, X_new):
        return self.model.predict(X_new)

    def save_model(self):
        joblib.dump(self.model, self.model_path)
        print(f"Modelo salvo em {self.model_path}")

    def load_model(self):
        try:
            self.model = joblib.load(self.model_path)
            print("Modelo carregado com sucesso.")
            return True
        except FileNotFoundError:
            print("Modelo não encontrado. É necessário treinar primeiro.")
            return False
