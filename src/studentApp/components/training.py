from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import  mean_squared_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso, Ridge
import pickle



class Training:
    def __init__(self, config):
        self.config = config
        self.models = {
            "Linear Regression": LinearRegression(),
            "Lasso": Lasso(),
            "Ridge": Ridge(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "XGBRegressor": XGBRegressor(),
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor()
        }
        self.metrics = {
            "Mean Squared Error": mean_squared_error,
            "R2 Score": r2_score
        }

    def save_model(self, model_name, model):
        save_directory = self.config.trained_model_path
        save_path = save_directory / "model.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        print(f"The best model '{model_name}' has been saved.")

    def evaluate_model(self, model, X, y):
        scores = {}
        for metric_name, metric_func in self.metrics.items():
            y_pred = model.predict(X)
            score = metric_func(y, y_pred)
            scores[metric_name] = score
        return scores

    def train(self, X_train, y_train):
        best_model = None
        best_model_name = ""
        best_scores = None
        total_models = len(self.models)
        current_model = 1

        for model_name, model in self.models.items():
            model.fit(X_train, y_train)  # Train model

            # Evaluate the model using multiple metrics
            scores = self.evaluate_model(model, X_train, y_train)

            if best_scores is None or all(score > best_scores[metric] for metric, score in scores.items()):
                best_scores = scores
                best_model_name = model_name
                best_model = model

            # Display metrics for all models
            print(f"Model: {model_name}")
            for metric, score in scores.items():
                print(f"- {metric}: {score:.4f}")

            # Calculate and display percentage progress
            progress = current_model / total_models * 100
            print(f"Training Progress: {progress:.2f}%")
            current_model += 1

        if best_model is not None:
            self.save_model(best_model_name, best_model)
            print("\nBest Model Scores:")
            for metric, score in best_scores.items():
                print(f"- {metric}: {score:.4f}")
        else:
            print("No best model found.")