# src/model/model_evaluation.py
import os
import json
import logging
import time
from typing import Dict, Any
import glob

import pandas as pd
import numpy as np
import joblib
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)
from mlflow.models import infer_signature
import mlflow
import mlflow.sklearn

# --- DAGSHUB ---
import dagshub

# Logging
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


class Model_Evaluation:
    """
    Clase que evalúa modelos entrenados:
    - Carga de parámetros
    - Carga del dataset de validación
    - Evaluación de cada modelo
    - Logging a MLflow/Dagshub
    """

    def __init__(self, params_path: str):
        self.params_path = params_path
        self.params = self.load_params()

        # --- Inicialización Dagshub ---
        dh_cfg = self.params.get("model_evaluation", {})
        dagshub.init(
            repo_owner='takenking9879',
            repo_name='MLOPS-project-Credit-Card-Fraud-Detection',
            mlflow=True
        )

        # Configuración MLflow usando la sección model_evaluation
        self.tracking_uri = dh_cfg.get("tracking_uri", None)
        self.experiment_name = dh_cfg.get("experiment_name", "default")

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        # Carpeta de datos
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        self.root_dir = root
        self.processed_dir = self.params.get("model_building", {}).get(
            "processed_dir", os.path.join(root, "data/processed")
        )
        self.val_path = self.params.get("model_building", {}).get(
            "val_path", os.path.join(self.processed_dir, "val_processed.csv")
        )
        # path absoluto para modelos (por si params trae relativo)
        self.models_dir = os.path.abspath(self.params.get("model_building", {}).get(
            "output_models_dir", os.path.join(root, "models/saved_models")
        ))

        # --- Crear carpeta de modelos si no existe ---
        os.makedirs(self.models_dir, exist_ok=True)

        # --- Crear carpeta de confusions y archivo metrics si no existen (en models/) ---
        self.cm_dir = os.path.join(root, "models/confusion_matrices")
        os.makedirs(self.cm_dir, exist_ok=True)
        self.metrics_file = os.path.join(root, "models/metrics.json")

        logger.debug(f"Root dir: {self.root_dir}")
        logger.debug(f"Models dir (busqueda .pkl): {self.models_dir}")
        logger.debug(f"Confusion matrices dir: {self.cm_dir}")
        logger.debug(f"Metrics file: {self.metrics_file}")

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

    def load_csv(self, path: str) -> pd.DataFrame:
        """Carga un CSV"""
        try:
            df = pd.read_csv(path)
            logger.debug("Dataset cargado %s con forma %s", path, df.shape)
            return df
        except Exception as e:
            logger.error("Error cargando CSV %s: %s", path, e)
            raise

    def evaluate_and_log(self, model_path: str, meta_path: str, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """Evalúa un modelo y registra resultados en MLflow"""
        logger.info(f"Evaluando modelo: {model_path}")
        try:
            model = joblib.load(model_path)
            logger.debug("Modelo cargado %s", model_path)
        except Exception as e:
            logger.error("No se pudo cargar %s: %s", model_path, e)
            return {}

        model_name = os.path.splitext(os.path.basename(model_path))[0]
        run_name = f"eval_{model_name}_{int(time.time())}"

        # Leer parámetros del modelo
        model_params = {}
        if meta_path and os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as mf:
                    model_params = json.load(mf).get("params", {})
            except Exception as e:
                logger.warning("No se pudo leer meta para %s: %s", model_path, e)

        metrics_dict = {}

        # Empezamos run MLflow
        with mlflow.start_run(run_name=run_name):
            try:
                # Log de parámetros
                for k, v in model_params.items():
                    try:
                        mlflow.log_param(k, v)
                    except Exception:
                        logger.debug("No se pudo loggear param %s=%s", k, v)
                mlflow.log_param("model_name", model_name)  # solo el nombre del modelo
                mlflow.set_tag("model_name", model_name)

                # Predicción (con manejo de excepciones para que no rompa todo)
                try:
                    y_pred = model.predict(X_val)
                    logger.debug("Predicción completada para %s", model_name)
                except Exception as e:
                    logger.error("Error en predict para %s: %s", model_name, e)
                    return {}

                probs = None
                try:
                    # Sólo calcular proba cuando sea binario y tenga predict_proba
                    if len(np.unique(y_val)) == 2 and hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_val)[:, 1]
                except Exception:
                    probs = None

                # Métricas
                try:
                    acc = accuracy_score(y_val, y_pred)
                    prec = precision_score(y_val, y_pred, average="binary" if len(np.unique(y_val)) == 2 else "macro", zero_division=0)
                    rec = recall_score(y_val, y_pred, average="binary" if len(np.unique(y_val)) == 2 else "macro", zero_division=0)
                    f1 = f1_score(y_val, y_pred, average="binary" if len(np.unique(y_val)) == 2 else "macro", zero_division=0)
                except Exception as e:
                    logger.error("Error calculando métricas para %s: %s", model_name, e)
                    return {}

                auc = None
                try:
                    if probs is not None:
                        auc = roc_auc_score(y_val, probs)
                except Exception:
                    auc = None

                # Log metrics (intento individual para evitar fallos)
                try:
                    mlflow.log_metric("accuracy", float(acc))
                    mlflow.log_metric("precision", float(prec))
                    mlflow.log_metric("recall", float(rec))
                    mlflow.log_metric("f1_score", float(f1))
                    if auc is not None:
                        mlflow.log_metric("roc_auc", float(auc))
                except Exception as e:
                    logger.warning("No se pudieron loggear algunas métricas para %s: %s", model_name, e)

                metrics_dict.update({
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1_score": f1
                })
                if auc is not None:
                    metrics_dict["roc_auc"] = auc

                # Reporte de clasificación
                try:
                    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
                    for label, metrics in report.items():
                        if isinstance(metrics, dict):
                            for m_name, m_val in metrics.items():
                                try:
                                    mlflow.log_metric(f"{label}_{m_name}", float(m_val))
                                except Exception:
                                    logger.debug("No se pudo loggear report metric %s_%s", label, m_name)
                                metrics_dict[f"{label}_{m_name}"] = m_val
                except Exception as e:
                    logger.warning("No se pudo generar classification_report para %s: %s", model_name, e)

                # --- Guardar matriz de confusión en models/confusion_matrices ---
                try:
                    cm = confusion_matrix(y_val, y_pred)
                    plt.figure(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt="d")
                    plt.title(f"Confusion matrix: {model_name}")

                    cm_file = os.path.join(self.cm_dir, f"confusion_{model_name}.png")
                    plt.savefig(cm_file)
                    plt.close()
                    logger.info(f"Confusion matrix guardada en {cm_file}")
                except Exception as e:
                    logger.error("No se pudo guardar la matriz de confusión para %s: %s", model_name, e)

                # Guardar modelo en MLflow (no crítico)
                try:
                    input_example = X_val.head(5)
                    signature = infer_signature(input_example, model.predict(input_example))
                except Exception:
                    input_example = None
                    signature = None

                try:
                    mlflow.sklearn.log_model(model, artifact_path=model_name, signature=signature, input_example=input_example)
                except Exception as e:
                    logger.warning("Fallo mlflow.sklearn.log_model para %s: %s", model_name, e)

                mlflow.set_tag("model_name", model_name)
                logger.info("Modelo %s evaluado y registrado en MLflow", model_name)
            except Exception as e:
                logger.error("Error evaluando %s: %s", model_path, e)

        return metrics_dict

    def evaluate_all(self):
        """Evalúa todos los modelos guardados en models_dir y genera metrics.json"""
        df_val = self.load_csv(self.val_path)
        if 'Class' in df_val.columns:
            target_col = 'Class'
        elif 'target' in df_val.columns:
            target_col = 'target'
        else:
            target_col = df_val.columns[-1]

        X_val = df_val.drop(columns=[target_col])
        y_val = df_val[target_col]

        all_metrics = {}
        saved_count = 0
        evaluated_count = 0

        # Buscar recursivamente todos los .pkl (por si hay subcarpetas)
        pkl_paths = glob.glob(os.path.join(self.models_dir, "**", "*.pkl"), recursive=True)
        logger.info(f"Modelos (.pkl) encontrados: {len(pkl_paths)}")
        for p in pkl_paths:
            logger.debug(f" -> {p}")

        for model_path in pkl_paths:
            fname = os.path.basename(model_path)
            evaluated_count += 1
            meta_path = os.path.splitext(model_path)[0] + "_meta.json"
            metrics = self.evaluate_and_log(model_path, meta_path, X_val, y_val)
            if metrics:
                all_metrics[os.path.splitext(fname)[0]] = metrics

        # Guardar metrics.json en models/
        clean_metrics = {}
        for model_name, mets in all_metrics.items():
            try:
                clean_metrics[model_name] = {k: float(v) for k, v in mets.items()}
            except Exception:
                # si hay valores no convertibles, los dejamos tal cual
                clean_metrics[model_name] = mets

        try:
            with open(self.metrics_file, "w") as f:
                json.dump(clean_metrics, f, indent=2)
            logger.info("Archivo metrics.json generado en %s", self.metrics_file)
        except Exception as e:
            logger.error("No se pudo escribir metrics.json en %s: %s", self.metrics_file, e)

        # Contar cuántas matrices existen ahora
        try:
            cm_files = glob.glob(os.path.join(self.cm_dir, "confusion_*.png"))
            saved_count = len(cm_files)
        except Exception:
            saved_count = 0

        logger.info(f"Modelos encontrados: {len(pkl_paths)}, evaluados: {evaluated_count}, matrices guardadas: {saved_count}")

        if saved_count == 0:
            logger.warning(f"'{self.cm_dir}' está vacío después de la ejecución — revisa los logs previos para errores.")

def main():
    try:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        params_path = os.path.join(root, "params.yaml")
        evaluator = Model_Evaluation(params_path)
        evaluator.evaluate_all()
    except Exception as e:
        logger.error("Error en main de model_evaluation: %s", e)
        raise


if __name__ == "__main__":
    main()
