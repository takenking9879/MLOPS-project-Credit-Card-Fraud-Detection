import os
import pandas as pd
import numpy as np
import yaml
import logging
from sklearn.model_selection import train_test_split
import kagglehub

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
    def __init__(self, params_path: str, raw_data_dir: str):
        self.params_path = params_path
        self.raw_data_dir = raw_data_dir
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
            dataset_path = kagglehub.dataset_download(
                "nelgiriyewithana/credit-card-fraud-detection-dataset-2023"
            )
            logger.debug(f"Dataset downloaded to {dataset_path}")
            # Supongamos que el CSV se llama "credit_card_fraud.csv" dentro del zip descargado
            csv_path = os.path.join(dataset_path, "credit_card_fraud.csv")
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
            # Eliminamos duplicados y NaNs
            df.drop_duplicates(inplace=True)
            logger.debug("Data preprocessing completed")
            self.df = df
            return df
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise

    def split_data(self) -> dict:
        """
        Split the dataset into train, val, test, and predict subsets
        based on YAML specifications for fraud and authentic transactions.
        """
        try:
            params = self.params['data_ingestion']
            df = self.df.copy()

            # Separar fraudes y no fraudes
            fraud = df[df['Class'] == 1]
            authentic = df[df['Class'] == 0]

            # Función para samplear
            def sample_df(df_sub, size):
                if size < 1:
                    return df_sub.sample(frac=size, random_state=42)
                else:
                    return df_sub.sample(n=int(size), random_state=42)

            # Train
            train = pd.concat([
                sample_df(fraud, params['train_size_fraud']),
                sample_df(authentic, params['train_size_authentic'])
            ]).sample(frac=1, random_state=42)

            # Validation
            val = pd.concat([
                sample_df(fraud.drop(train[train['Class'] == 1].index), params['val_size_fraud']),
                sample_df(authentic.drop(train[train['Class'] == 0].index), params['val_size_authentic'])
            ]).sample(frac=1, random_state=42)

            # Test
            test = pd.concat([
                sample_df(fraud.drop(train.index.union(val.index)), params['test_size_fraud']),
                sample_df(authentic.drop(train.index.union(val.index)), params['test_size_authentic'])
            ]).sample(frac=1, random_state=42)

            # Predict
            predict = pd.concat([
                sample_df(fraud.drop(train.index.union(val.index).union(test.index)), params['predict_size_fraud']),
                sample_df(authentic.drop(train.index.union(val.index).union(test.index)), params['predict_size_authentic'])
            ]).sample(frac=1, random_state=42)

            logger.debug("Data split into train/val/test/predict completed")
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


# === ZenML/DVC ready entrypoint ===
def main():
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        params_path = os.path.join(root_dir, 'params.yaml')
        raw_data_dir = os.path.join(root_dir, 'data/raw')

        ingestion = DataIngestion(params_path, raw_data_dir)
        ingestion.download_data()
        ingestion.preprocess_data()
        splits = ingestion.split_data()
        ingestion.save_data(splits)

    except Exception as e:
        logger.error(f"Failed to complete the data ingestion pipeline: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
