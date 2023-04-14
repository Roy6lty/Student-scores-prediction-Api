import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            logging.info("models")
            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regressor': LinearRegression(),
                'K-Neigbors Classifer': KNeighborsRegressor(),
                'XGBRegressor': XGBRegressor(),
                'Adaboost': AdaBoostRegressor(),
                'catboostClassifer': CatBoostRegressor(verbose=False)
            }
            
            logging.info("Evaluating_model")

            model_report:dict = evaluate_models(X_train= X_train, y_train=y_train, 
                                            X_test=X_test, y_test=y_test, models = models)
            
            ## To get best model score for Dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name for dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score <0.6:
                raise CustomException('No best model found')
            logging.info(f'Best model found on both testing and training data')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            

        except Exception as e:
            CustomException(e,sys)