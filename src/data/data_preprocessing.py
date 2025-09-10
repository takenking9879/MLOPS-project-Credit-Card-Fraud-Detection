import os
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
import yaml
import joblib
from utils import create_logger, BaseUtils

# Logging configuration

logger = create_logger('data_preprocessing', 'preprocessing_errors.log')

class DataPreprocessing(BaseUtils):
    def __init__(self, params_path: str, raw_data_dir: str, preprocessed_data_dir: str = None):
        super().__init__(logger=logger,params_path=params_path)
        self.raw_data_dir = raw_data_dir
        self.preprocessed_data_dir = preprocessed_data_dir or os.path.join("data", "processed")
        self.params = self.load_params()
        self.scaler = self._choose_scaler()
        self.imputer = self._choose_imputer()

    def _choose_scaler(self):
            try:
                scalers = {
                    "standardscaler": StandardScaler,
                    "minmaxscaler": MinMaxScaler,
                    "robustscaler": RobustScaler
                }
                scaler_type = self.params['data_preprocessing']['scaler_method'].lower()
                if scaler_type not in scalers:
                    raise ValueError(f"Unknown scaler_method '{scaler_type}' in params.yaml")
                
                scaler = scalers[scaler_type]()
                self.logger.debug(f"Scaler chosen: {scaler.__class__.__name__}")
                return scaler
            except Exception as e:
                self.logger.error(f"Error choosing scaler: {e}")
                raise

    def _choose_imputer(self):
        try:
            imputers = {
                "knnimputer": KNNImputer,
                "simpleimputer": lambda: SimpleImputer(strategy='mean')
            }
            imputer_type = self.params['data_preprocessing']['imputer_method'].lower()
            if imputer_type not in imputers:
                raise ValueError(f"Unknown imputer_method '{imputer_type}' in params.yaml")
            
            imputer = imputers[imputer_type]()
            self.logger.debug(f"Imputer chosen: {imputer.__class__.__name__}")
            return imputer
        except Exception as e:
            self.logger.error(f"Error choosing imputer: {e}")
            raise

    def load_data(self, filenames: list[str]) -> dict[str, pd.DataFrame]:
        datasets = {}
        for file in filenames:
            path = os.path.join(self.raw_data_dir, file)
            try:
                df = pd.read_csv(path)
                if 'id' in df.columns:
                    df = df.drop(columns='id')
                datasets[file.split('.')[0]] = df
                self.logger.debug(f"Loaded {file} with shape {df.shape}")
            except Exception as e:
                self.logger.error(f"Error loading {file}: {e}")
                raise
        return datasets

    def fit_imputer(self, df: pd.DataFrame) -> None:
        """
        Fit the imputer using the provided DataFrame (without the target column)
        and logs the operation.
        """
        try:
            data = df.drop(columns=['Class'], errors='ignore')
            self.imputer.fit(data)
            self.logger.debug("Imputer fitted with training data")
        except Exception as e:
            self.logger.error(f"Error fitting imputer: {e}")
            raise

    def normalize(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        try:
            target = df['Class']          # <-- guardamos Class en target
            df = df.drop(columns=['Class']) 
            
            if fit:
                self.fit_imputer(df)
                df_scaled = self.scaler.fit_transform(df)
                self.logger.debug("Data normalized with fit_transform")
            else:
                df_scaled = self.scaler.transform(df)
                self.logger.debug("Data normalized with transform")
            
            # Convertir a DataFrame y agregar la columna target
            df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
            df_scaled['target'] = target   # <-- agregamos target de vuelta

            return df_scaled
        except Exception as e:
            self.logger.error(f"Error normalizing data: {e}")
            raise


    def save_datasets(self, datasets: dict[str, pd.DataFrame]) -> None:
        try:
            os.makedirs(self.preprocessed_data_dir, exist_ok=True)
            for name, df in datasets.items():
                file_path = os.path.join(self.preprocessed_data_dir, f"{name}_processed.csv")
                df.to_csv(file_path, index=False)
                self.logger.debug(f"Saved {name} to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving datasets: {e}")
            raise

    def save_scaler(self, filename: str = "scaler.pkl") -> None:
        try:
            artifacts_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')), "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            save_path = os.path.join(artifacts_dir, filename)
            joblib.dump(self.scaler, save_path)
            self.logger.debug(f"Scaler guardado en {save_path}")
        except Exception as e:
            self.logger.error(f"No se pudo guardar el scaler: {e}")
            raise

    def save_imputer(self, filename: str = "imputer.pkl") -> None:
        try:
            artifacts_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')), "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            save_path = os.path.join(artifacts_dir, filename)
            joblib.dump(self.imputer, save_path)
            self.logger.debug(f"imputer guardado en {save_path}")
        except Exception as e:
            self.logger.error(f"No se pudo guardar el imputer: {e}")
            raise


# Entrypoint
def main():
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        params_path = os.path.join(root_dir, 'params.yaml')
        raw_data_dir = os.path.join(root_dir, 'data/raw')
        processed_data_dir = os.path.join(root_dir, 'data/processed')

        dp = DataPreprocessing(params_path, raw_data_dir, processed_data_dir)

        # Cargar datasets
        filenames = ["train.csv", "val.csv", "test.csv", "predict.csv"]
        datasets = dp.load_data(filenames)

        # Normalizar
        datasets_scaled = {}
        for name, df in datasets.items():
            fit = (name == "train")  # Solo fit en el dataset de entrenamiento
            datasets_scaled[name] = dp.normalize(df, fit=fit)

        # Guardar datasets procesados
        dp.save_datasets(datasets_scaled)

        # Guardar scaler
        dp.save_scaler()

        # Guardar imputer
        dp.save_imputer()

    except Exception as e:
        dp.logger.error(f"Failed to complete the data preprocessing pipeline: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
