import os
import pandas as pd
import yaml
import logging
from sklearn.model_selection import train_test_split
import kagglehub
import shutil
import zipfile

# Logging configuration
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("errors.log")
file_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

class DataIngestion:
    def __init__(self, params_path: str, raw_data_dir: str, original_data_dir: str):
        self.params_path = params_path
        self.raw_data_dir = raw_data_dir
        self.original_data_dir = original_data_dir
        self.params = self.load_params()
        self.df = None

    def load_params(self) -> dict:
        try:
            with open(self.params_path, 'r') as f:
                params = yaml.safe_load(f)
            logger.debug(f"Parameters loaded from {self.params_path}")
            return params
        except Exception as e:
            logger.error(f"Error loading YAML parameters: {e}")
            raise

    def download_data(self) -> pd.DataFrame:
        try:
            # Descargar dataset
            dataset_path = kagglehub.dataset_download(
                "nelgiriyewithana/credit-card-fraud-detection-dataset-2023"
            )
            logger.debug(f"Dataset downloaded to {dataset_path}")

            # Crear carpetas
            os.makedirs(self.raw_data_dir, exist_ok=True)
            os.makedirs(self.original_data_dir, exist_ok=True)

            # Buscar ZIP o CSV
            zip_files = [f for f in os.listdir(dataset_path) if f.endswith(".zip")]
            csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]

            if zip_files:
                zip_file = os.path.join(dataset_path, zip_files[0])
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(self.original_data_dir)
                logger.debug(f"ZIP extracted to {self.original_data_dir}")
                # Tomar el CSV dentro del ZIP
                csv_path = os.path.join(self.original_data_dir, "creditcard_2023.csv")
            elif csv_files:
                for csv_file in csv_files:
                    shutil.copy(os.path.join(dataset_path, csv_file), self.original_data_dir)
                logger.debug(f"CSV files copied to {self.original_data_dir}")
                csv_path = os.path.join(self.original_data_dir, csv_files[0])
            else:
                raise FileNotFoundError(f"No ZIP or CSV found in {dataset_path}")

            # Leer CSV
            df = pd.read_csv(csv_path)
            logger.debug(f"Dataset loaded from {csv_path}")
            self.df = df
            return df

        except Exception as e:
            logger.error(f"Error downloading/loading dataset: {e}")
            raise

    def preprocess_data(self) -> pd.DataFrame:
        try:
            df = self.df.copy()
            df.drop_duplicates(inplace=True)
            logger.debug("Data preprocessing completed")
            self.df = df
            return df
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise

    def split_data(self) -> dict:
        try:
            params = self.params['data_ingestion']
            df = self.df.copy()
            if 'Class' not in df.columns:
                raise KeyError(f"'Class' column not found in dataframe. Columns: {df.columns}")

            fraud = df[df['Class'] == 1]
            authentic = df[df['Class'] == 0]

            def split_class(df_class, sizes):
                train, temp = train_test_split(df_class, train_size=sizes['train'], random_state=42, shuffle=True)
                remaining_total = 1 - sizes['train']
                relative_sizes = {k: v / remaining_total for k, v in sizes.items() if k != 'train'}
                val, temp2 = train_test_split(temp, train_size=relative_sizes['val'], random_state=42, shuffle=True)
                test, predict = train_test_split(
                    temp2,
                    train_size=relative_sizes['test'] / (relative_sizes['test'] + relative_sizes['predict']),
                    random_state=42, shuffle=True
                )
                return train, val, test, predict

            sizes_fraud = {
                'train': params['train_size_fraud'],
                'val': params['val_size_fraud'],
                'test': params['test_size_fraud'],
                'predict': params['predict_size_fraud']
            }
            sizes_auth = {
                'train': params['train_size_authentic'],
                'val': params['val_size_authentic'],
                'test': params['test_size_authentic'],
                'predict': params['predict_size_authentic']
            }

            train_fraud, val_fraud, test_fraud, pred_fraud = split_class(fraud, sizes_fraud)
            train_auth, val_auth, test_auth, pred_auth = split_class(authentic, sizes_auth)

            train = pd.concat([train_fraud, train_auth]).sample(frac=1, random_state=42)
            val = pd.concat([val_fraud, val_auth]).sample(frac=1, random_state=42)
            test = pd.concat([test_fraud, test_auth]).sample(frac=1, random_state=42)
            predict = pd.concat([pred_fraud, pred_auth]).sample(frac=1, random_state=42)

            logger.debug(f"Data split completed: train={len(train)}, val={len(val)}, test={len(test)}, predict={len(predict)}")
            return {'train': train, 'val': val, 'test': test, 'predict': predict}

        except Exception as e:
            logger.error(f"Error during data splitting: {e}")
            raise

    def save_data(self, data_splits: dict):
        try:
            os.makedirs(self.raw_data_dir, exist_ok=True)
            for split_name, df_split in data_splits.items():
                file_path = os.path.join(self.raw_data_dir, f"{split_name}.csv")
                df_split.to_csv(file_path, index=False)
                logger.debug(f"{split_name} data saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving data splits: {e}")
            raise

# Entrypoint
def main():
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        params_path = os.path.join(root_dir, 'params.yaml')
        raw_data_dir = os.path.join(root_dir, 'data/raw')
        original_data_dir = os.path.join(root_dir, 'data/original')

        ingestion = DataIngestion(params_path, raw_data_dir, original_data_dir)
        ingestion.download_data()
        ingestion.preprocess_data()
        splits = ingestion.split_data()
        ingestion.save_data(splits)

    except Exception as e:
        logger.error(f"Failed to complete the data ingestion pipeline: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
