import os
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import yaml
import joblib

# Logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


class DataPreprocessing:
    def __init__(self, params_path: str, raw_data_dir: str, preprocessed_data_dir: str = None):
        self.params_path = params_path
        self.raw_data_dir = raw_data_dir
        self.preprocessed_data_dir = preprocessed_data_dir or os.path.join("data", "processed")
        self.params = self.load_params()
        self.scaler = self._choose_scaler()

    def _choose_scaler(self):
        try:
            scaler_type = self.params['data_preprocessing']['scaler_method'].lower()
            if scaler_type == 'standardscaler':
                scaler = StandardScaler()
            elif scaler_type == 'minmaxscaler':
                scaler = MinMaxScaler()
            elif scaler_type == 'robustscaler':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaler_method '{scaler_type}' in params.yaml")
            logger.debug(f"Scaler chosen: {scaler.__class__.__name__}")
            return scaler
        except Exception as e:
            logger.error(f"Error choosing scaler: {e}")
            raise

    def load_params(self) -> dict:
        try:
            with open(self.params_path, 'r') as file:
                params = yaml.safe_load(file)
            logger.debug('Parameters retrieved from %s', self.params_path)
            return params
        except FileNotFoundError:
            logger.error('File not found: %s', self.params_path)
            raise
        except yaml.YAMLError as e:
            logger.error('YAML error: %s', e)
            raise
        except Exception as e:
            logger.error('Unexpected error: %s', e)
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
                logger.debug(f"Loaded {file} with shape {df.shape}")
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
                raise
        return datasets

    def normalize(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        try:
            target = df['Class']          # <-- guardamos Class en target
            df = df.drop(columns=['Class']) 
            
            if fit:
                df_scaled = self.scaler.fit_transform(df)
                logger.debug("Data normalized with fit_transform")
            else:
                df_scaled = self.scaler.transform(df)
                logger.debug("Data normalized with transform")
            
            # Convertir a DataFrame y agregar la columna target
            df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
            df_scaled['target'] = target   # <-- agregamos target de vuelta

            return df_scaled
        except Exception as e:
            logger.error(f"Error normalizing data: {e}")
            raise


    def save_datasets(self, datasets: dict[str, pd.DataFrame]) -> None:
        try:
            os.makedirs(self.preprocessed_data_dir, exist_ok=True)
            for name, df in datasets.items():
                file_path = os.path.join(self.preprocessed_data_dir, f"{name}_processed.csv")
                df.to_csv(file_path, index=False)
                logger.debug(f"Saved {name} to {file_path}")
        except Exception as e:
            logger.error(f"Error saving datasets: {e}")
            raise

    def save_scaler(self, filename: str = "scaler.pkl") -> None:
        try:
            artifacts_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')), "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            save_path = os.path.join(artifacts_dir, filename)
            joblib.dump(self.scaler, save_path)
            logger.debug(f"Scaler guardado en {save_path}")
        except Exception as e:
            logger.error(f"No se pudo guardar el scaler: {e}")
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

    except Exception as e:
        logger.error(f"Failed to complete the data preprocessing pipeline: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
