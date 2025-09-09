# src/model/model_building.py 
import os
import json
import time
import logging
from typing import Dict, Any
import re
import importlib


import pandas as pd
import joblib
import yaml

# External libs
try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)

# Logging
logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


class Model_Building:
    """
    ModelBuilder encapsula:
    - Carga de parámetros desde params.yaml
    - Carga de datasets procesados
    - Creación de modelos (factory)
    - Entrenamiento y guardado
    """

    def __init__(self, params_path: str, processed_dir: str, output_dir: str = None):
        self.params_path = params_path
        self.processed_dir = processed_dir
        self.params = self.load_params()
        # -------------------------------------------------------------
        # Cambio: salida de modelos para que DVC detecte correctamente
        # default ahora apunta a models/saved_models (coincide con tu dvc.yaml)
        # <-- CAMBIO
        self.output_dir = output_dir or self.params.get("model_building", {}).get(
            "output_models_dir", "models/saved_models"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        # -------------------------------------------------------------

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

    def _load_preprocessed_csv(self, path: str) -> pd.DataFrame:
        """Carga un CSV procesado"""
        try:
            df = pd.read_csv(path)
            logger.debug("Dataset cargado %s con forma %s", path, df.shape)
            return df
        except Exception as e:
            logger.error("Error cargando CSV %s: %s", path, e)
            raise

    def _model_factory(self, model_key: str, model_cfg: Dict[str, Any]):
        """
        Crea un modelo desde params.yaml usando import dinámico.
        model_cfg debe tener:
        class: "sklearn.linear_model.LogisticRegression"
        params: {C: 1.0, max_iter: 1000}
        """
        class_path = model_cfg.get("class")
        params = model_cfg.get("params", {})

        if not class_path:
            raise ValueError(f"No se especificó 'class' para el modelo {model_key}")

        module_path, class_name = class_path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
        except Exception as e:
            raise ImportError(f"No se pudo importar {class_path}: {e}")

        try:
            model = cls(**params)
        except TypeError as e:
            raise ValueError(f"Error inicializando {class_path} con params {params}: {e}")

        return model

    def train_and_save_models(self):
        """Entrena todos los modelos definidos en params.yaml y los guarda"""
        train_path = self.params.get("model_building", {}).get(
            "train_path", os.path.join(self.processed_dir, "train_processed.csv")
        )
        df_train = self._load_preprocessed_csv(train_path)

        # Detectar columna target
        if 'Class' in df_train.columns:
            target_col = 'Class'
        elif 'target' in df_train.columns:
            target_col = 'target'
        else:
            target_col = df_train.columns[-1]

        X_train = df_train.drop(columns=[target_col])
        y_train = df_train[target_col]

        models_cfg = self.params.get("model_building", {}).get("models", {})
        if not models_cfg:
            raise ValueError("No hay modelos definidos en params.yaml > model_building.models")

        saved_models = []
        for model_key, model_cfg in models_cfg.items():
            try:
                logger.info("Entrenando modelo: %s", model_key)
                model = self._model_factory(model_key, model_cfg)
                model.fit(X_train, y_train)

                ts = time.strftime("%Y%m%d_%H%M%S")
                model_filename = f"{model_key}_{ts}.pkl"
                model_path = os.path.join(self.output_dir, model_filename)
                joblib.dump(model, model_path)

                meta = {
                    "model_key": model_key,
                    "model_class": model_cfg.get("class", model_key),
                    "params": model_cfg.get("params", {})
                }
                meta_path = os.path.splitext(model_path)[0] + "_meta.json"
                with open(meta_path, "w") as mf:
                    json.dump(meta, mf, indent=2)

                saved_models.append({"model_path": model_path, "meta_path": meta_path})
                logger.info("Modelo %s guardado en %s", model_key, model_path)
            except Exception as e:
                logger.error("Error entrenando/guardando modelo %s: %s", model_key, e)

        logger.debug("Modelos entrenados: %s", saved_models)
        return saved_models


def main():
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        params_path = os.path.join(root_dir, "params.yaml")
        processed_dir = os.path.join(root_dir, "data/processed")
        # <-- mantengo la salida en models/saved_models por compatibilidad con DVC
        output_dir = os.path.join(root_dir, "models/saved_models")

        builder = Model_Building(params_path, processed_dir, output_dir)
        builder.train_and_save_models()
    except Exception as e:
        logger.error("Error en main de model_building: %s", e)
        raise


if __name__ == "__main__":
    main()
