# src/model/model_evaluation.py
import os
import json
import logging
import time
from typing import Dict, Any
import glob
import shutil

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
    - Selección automática de modelo final según pesos (params)
    - Copia del modelo seleccionado y guardado de métricas finales y evaluación sobre test
    """

    def __init__(self, params_path: str):
        self.params_path = params_path
        self.params = self.load_params()

        # --- Inicialización Dagshub ---
        dh_cfg = self.params.get("model_evaluation", {})
        try:
            dagshub.init(
                repo_owner=dh_cfg.get("repo_owner", "takenking9879"),
                repo_name=dh_cfg.get("repo_name", "MLOPS-project-Credit-Card-Fraud-Detection"),
                mlflow=True
            )
        except Exception as e:
            logger.warning("dagshub.init() falló o no está disponible: %s", e)

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

        # --- Crear carpeta de confusions y metrics ---
        self.cm_dir = os.path.join(root, "models/confusion_matrices")
        os.makedirs(self.cm_dir, exist_ok=True)

        # Carpeta para métricas individuales por modelo (validación)
        self.metrics_dir = os.path.join(root, "models/metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        # Archivo resumen de métricas en metrics/
        self.metrics_file = os.path.join(self.metrics_dir, "metrics_summary.json")

        # model_final defaults (si quieres que model_evaluation promueva final)
        mf_cfg = self.params.get("model_final", {})
        self.final_model_dir = os.path.join(root, "models", "final_model")
        self.final_model_path = mf_cfg.get("output_model_path", None)  # can be None -> build default later
        # default final metrics path will be set per-model when saving test metrics
        self.final_metrics_path = mf_cfg.get("output_metrics_path", os.path.join(self.final_model_dir, "metrics.json"))
        self.test_path = mf_cfg.get("test_path", os.path.join(self.processed_dir, "test_processed.csv"))

        logger.debug(f"Root dir: {self.root_dir}")
        logger.debug(f"Models dir (busqueda .pkl): {self.models_dir}")
        logger.debug(f"Confusion matrices dir: {self.cm_dir}")
        logger.debug(f"Metrics dir: {self.metrics_dir}")
        logger.debug(f"Final model dir default: {self.final_model_dir}, final_model_path param: {self.final_model_path}")

        # --- CHANGED: selection config support (overall / specific / combined) ---
        sel_cfg = self.params.get("model_evaluation", {}).get("selection", {})
        self.selection_mode = sel_cfg.get("mode", "overall")
        self.num_classes = sel_cfg.get("num_classes", None)
        self.weights_overall = sel_cfg.get("overall", {})
        self.weights_specific = sel_cfg.get("specific", {})
        self.combined_overall_weight = float(sel_cfg.get("combined_overall_weight", 0.5))

        # backward compatibility: if old single 'weights' present, use it as overall
        legacy_weights = self.params.get("model_evaluation", {}).get("weights", None)
        if legacy_weights and not (self.weights_overall or self.weights_specific):
            self.weights_overall = legacy_weights

        logger.debug(f"Selection mode: {self.selection_mode}, num_classes: {self.num_classes}")
        logger.debug(f"Overall weights keys: {list(self.weights_overall.keys())}")
        logger.debug(f"Specific weights keys: {list(self.weights_specific.keys())}")


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

    def _resolve_metric_value(self, metrics: Dict[str, Any], key: str):
        """
        Intenta resolver el valor de una métrica en el diccionario `metrics`.
        Soporta variantes: 'f1_score', 'f1-score', '1_recall', '1_f1-score', etc.
        Devuelve float o None si no se encuentra.
        """
        try:
            if not isinstance(metrics, dict):
                return None

            # intentos directos
            if key in metrics:
                try:
                    return float(metrics[key])
                except Exception:
                    return None

            # variantes comunes
            variants = []
            # mapping sencillo para underscore/guion
            if key.replace("-", "_") in metrics:
                variants.append(key.replace("-", "_"))
            if key.replace("_", "-") in metrics:
                variants.append(key.replace("_", "-"))

            # si es class-specific (ej "1_recall"), ya cubierto por directos
            # comprobar si la métrica global tiene guion vs underscore
            if key.lower() in ["f1-score", "f1_score", "f1"]:
                for v in ("f1_score", "f1-score", "f1"):
                    variants.append(v)

            for v in variants:
                if v in metrics:
                    try:
                        return float(metrics[v])
                    except Exception:
                        return None

            # por si las metricas vienen como '1': {'precision':..., 'recall':...}
            for k, val in metrics.items():
                # si val es dict y key es '1_recall' -> buscar label '1' y metric 'recall'
                if isinstance(val, dict) and "_" in key:
                    lbl, mname = key.split("_", 1)
                    if k == lbl and mname in val:
                        try:
                            return float(val[mname])
                        except Exception:
                            return None

            return None
        except Exception as e:
            logger.debug("Error resolviendo métrica %s: %s", key, e)
            return None

    def _compute_weighted_score(self, metrics: Dict[str, Any], weights: Dict[str, float]) -> float:
        """
        Calcula score ponderado y lo normaliza a [0,1].
        """
        try:
            if not weights:
                for fallback in ("f1_score", "f1-score", "accuracy"):
                    v = self._resolve_metric_value(metrics, fallback)
                    if v is not None:
                        return float(max(0.0, min(1.0, float(v))))
                return 0.0

            raw = 0.0
            sum_pos = 0.0
            sum_abs = 0.0
            found_any = False
            for k, w in weights.items():
                try:
                    val = self._resolve_metric_value(metrics, k)
                    if val is None:
                        continue
                    found_any = True
                    raw += float(w) * float(val)
                    if float(w) > 0:
                        sum_pos += float(w)
                    sum_abs += abs(float(w))
                except Exception as e:
                    logger.debug("Error al procesar peso %s=%s : %s", k, w, e)
                    continue

            if not found_any:
                for fallback in ("f1_score", "f1-score", "accuracy"):
                    v = self._resolve_metric_value(metrics, fallback)
                    if v is not None:
                        return float(max(0.0, min(1.0, float(v))))
                return 0.0

            if sum_pos > 0:
                normalized = raw / sum_pos
            elif sum_abs > 0:
                normalized = raw / sum_abs
            else:
                normalized = raw

            try:
                normalized = float(normalized)
            except Exception:
                normalized = 0.0

            normalized = max(0.0, min(1.0, normalized))
            return normalized
        except Exception as e:
            logger.error("Error calculando score ponderado: %s", e)
            return 0.0

    def _compute_selection_score(self, metrics: Dict[str, Any]) -> float:
        try:
            if self.selection_mode == "specific":
                return self._compute_weighted_score(metrics, self.weights_specific)
            elif self.selection_mode == "combined":
                overall_score = self._compute_weighted_score(metrics, self.weights_overall)
                specific_score = self._compute_weighted_score(metrics, self.weights_specific)
                w = self.combined_overall_weight
                w = max(0.0, min(1.0, float(w)))
                combined = w * overall_score + (1 - w) * specific_score
                return max(0.0, min(1.0, combined))
            else:
                return self._compute_weighted_score(metrics, self.weights_overall)
        except Exception as e:
            logger.error("Error calculando selection score (mode=%s): %s", self.selection_mode, e)
            return 0.0

    def _promote_final_model(self, selected_model_path: str, model_name: str, validation_metrics: Dict[str, Any]):
        """
        Copia el .pkl seleccionado a final_model/ y guarda las métricas de validación (si se pasan)
        en final_model como '<model>_validation_metrics.json'. Las métricas de test se guardan
        posteriormente en '<model>_test_metrics.json' desde evaluate_on_test.
        """
        try:
            # determinar destino del pkl final
            if self.final_model_path:
                dest_model_path = os.path.abspath(self.final_model_path)
            else:
                os.makedirs(self.final_model_dir, exist_ok=True)
                dest_model_path = os.path.join(self.final_model_dir, f"{model_name}.pkl")

            # copiar el pkl
            try:
                shutil.copyfile(selected_model_path, dest_model_path)
                logger.info("Modelo final copiado a %s", dest_model_path)
            except Exception as e:
                logger.error("No se pudo copiar modelo final %s -> %s : %s", selected_model_path, dest_model_path, e)

            # guardar metrics de validación en final_model con nombre claro
            try:
                if validation_metrics:
                    val_dst = os.path.join(self.final_model_dir, f"{model_name}_validation_metrics.json")
                    os.makedirs(os.path.dirname(val_dst), exist_ok=True)
                    with open(val_dst, "w") as f:
                        json.dump(validation_metrics, f, indent=2)
                    logger.info("Métricas de validación copiada a %s", val_dst)
                else:
                    # intentar copiar el archivo de metrics individual si existe en metrics_dir
                    src_val = os.path.join(self.metrics_dir, f"{model_name}_metrics.json")
                    if os.path.exists(src_val):
                        dst_val = os.path.join(self.final_model_dir, f"{model_name}_validation_metrics.json")
                        shutil.copyfile(src_val, dst_val)
                        logger.info("Métricas de validación copiadas desde %s a %s", src_val, dst_val)
            except Exception as e:
                logger.error("No se pudieron guardar/copiar métricas de validación en final_model: %s", e)

            # write a small best_model.json for other steps to read
            try:
                best_info = {
                    "model_name": model_name,
                    "model_path": os.path.abspath(selected_model_path),
                    "promoted_path": os.path.abspath(dest_model_path),
                    "validation_metrics_path": os.path.abspath(os.path.join(self.final_model_dir, f"{model_name}_validation_metrics.json")),
                    "test_metrics_path": os.path.abspath(os.path.join(self.final_model_dir, f"{model_name}_test_metrics.json")),
                    "timestamp": int(time.time())
                }
                best_file = os.path.join(self.root_dir, "models", "best_model.json")
                with open(best_file, "w") as bf:
                    json.dump(best_info, bf, indent=2)
                logger.info("Best model info written to %s", best_file)
            except Exception as e:
                logger.error("No se pudo escribir best_model.json: %s", e)

        except Exception as e:
            logger.error("Error en promoción del modelo final: %s", e)

    def evaluate_on_test(self, selected_model_path: str) -> Dict[str, Any]:
        """
        Evalúa el modelo seleccionado en test_path y guarda métricas de test en
        final_model/<model>_test_metrics.json. También hace logging en MLflow.
        """
        metrics = {}
        try:
            if not os.path.exists(selected_model_path):
                raise FileNotFoundError(f"Selected model not found: {selected_model_path}")
            model = joblib.load(selected_model_path)
        except Exception as e:
            logger.error("Error cargando modelo seleccionado para evaluación en test: %s", e)
            return metrics

        # Cargar test
        try:
            if not os.path.exists(self.test_path):
                raise FileNotFoundError(f"No se encontró test dataset: {self.test_path}")
            df_test = pd.read_csv(self.test_path)
            logger.info("Test dataset cargado desde %s con shape %s", self.test_path, df_test.shape)
        except Exception as e:
            logger.error("Error cargando test CSV %s: %s", self.test_path, e)
            return metrics

        # Determinar target
        if 'Class' in df_test.columns:
            target_col = 'Class'
        elif 'target' in df_test.columns:
            target_col = 'target'
        else:
            target_col = df_test.columns[-1]

        X_test = df_test.drop(columns=[target_col])
        y_test = df_test[target_col]

        # Predict y calc metrics
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            logger.error("Error en predict sobre test: %s", e)
            return metrics

        try:
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
            metrics["precision"] = precision_score(y_test, y_pred, average="binary" if len(np.unique(y_test)) == 2 else "macro", zero_division=0)
            metrics["recall"] = recall_score(y_test, y_pred, average="binary" if len(np.unique(y_test)) == 2 else "macro", zero_division=0)
            metrics["f1_score"] = f1_score(y_test, y_pred, average="binary" if len(np.unique(y_test)) == 2 else "macro", zero_division=0)
        except Exception as e:
            logger.error("Error calculando métricas en test: %s", e)

        try:
            if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_test, probs)
            else:
                metrics["roc_auc"] = None
        except Exception:
            metrics["roc_auc"] = None

        # classification_report para test (clase específica)
        try:
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            for label, vals in report.items():
                if isinstance(vals, dict):
                    for m_name, m_val in vals.items():
                        key = f"{label}_{m_name}"
                        metrics[key] = m_val
                        metrics[key.replace("-", "_")] = m_val
        except Exception as e:
            logger.warning("No se pudo generar classification_report para test: %s", e)

        # Guardar metrics test en final_model con nombre claro
        try:
            model_basename = os.path.splitext(os.path.basename(selected_model_path))[0]
            dst_metrics = os.path.join(self.final_model_dir, f"{model_basename}_test_metrics.json")
            os.makedirs(os.path.dirname(dst_metrics), exist_ok=True)
            with open(dst_metrics, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info("Métricas de test guardadas en %s", dst_metrics)
            # también actualizar self.final_metrics_path para consistencia
            self.final_metrics_path = dst_metrics
        except Exception as e:
            logger.error("No se pudieron guardar metrics de test: %s", e)

        # Log a MLflow run para métricas de test (usar nested si hay run activa)
        try:
            nested_flag = mlflow.active_run() is not None
            with mlflow.start_run(run_name=f"final_test_{os.path.splitext(os.path.basename(selected_model_path))[0]}_{int(time.time())}", nested=nested_flag):
                for k, v in metrics.items():
                    try:
                        if v is None:
                            continue
                        mlflow.log_metric(k, float(v))
                    except Exception:
                        pass
                mlflow.set_tag("final_test_eval", os.path.basename(selected_model_path))
        except Exception as e:
            logger.warning("No se pudo loggear métricas de test en MLflow: %s", e)

        return metrics

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

        # Empezamos run MLflow (usar nested si ya hay run)
        nested_flag = mlflow.active_run() is not None
        with mlflow.start_run(run_name=run_name, nested=nested_flag):
            try:
                # Log de parámetros
                for k, v in model_params.items():
                    try:
                        mlflow.log_param(k, v)
                    except Exception:
                        logger.debug("No se pudo loggear param %s=%s", k, v)
                mlflow.log_param("model_name", model_name)
                mlflow.set_tag("model_name", model_name)

                # Predicción
                try:
                    y_pred = model.predict(X_val)
                    logger.debug("Predicción completada para %s", model_name)
                except Exception as e:
                    logger.error("Error en predict para %s: %s", model_name, e)
                    return {}

                probs = None
                try:
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

                # Log metrics
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

                # classification_report -> métricas por clase
                try:
                    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
                    for label, mets in report.items():
                        if isinstance(mets, dict):
                            for m_name, m_val in mets.items():
                                metric_key = f"{label}_{m_name}"
                                metrics_dict[metric_key] = m_val
                                alt_key = metric_key.replace("-", "_")
                                metrics_dict[alt_key] = m_val
                                try:
                                    mlflow.log_metric(metric_key, float(m_val))
                                except Exception:
                                    pass
                except Exception as e:
                    logger.warning("No se pudo generar classification_report para %s: %s", model_name, e)

                # Guardar matriz de confusión
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

                # Guardar el metrics_dict localmente en models/metrics/
                try:
                    model_basename = os.path.splitext(os.path.basename(model_path))[0]
                    model_metrics_file = os.path.join(self.metrics_dir, f"{model_basename}_metrics.json")
                    with open(model_metrics_file, "w") as mf:
                        json.dump(metrics_dict, mf, indent=2)
                    logger.info(f"Métricas de {model_basename} guardadas en {model_metrics_file}")
                except Exception as e:
                    logger.warning(f"No se pudieron guardar métricas para {model_name}: {e}")

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
        """Evalúa todos los modelos guardados, selecciona el mejor y lo promueve con métricas de test en un solo run MLflow."""

        # --- Cerrar cualquier run activa de MLflow antes de empezar ---
        if mlflow.active_run() is not None:
            mlflow.end_run()

        # Cargar validación para evaluar modelos
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

        # Buscar todos los modelos .pkl
        pkl_paths = glob.glob(os.path.join(self.models_dir, "**", "*.pkl"), recursive=True)
        logger.info(f"Modelos (.pkl) encontrados: {len(pkl_paths)}")

        for model_path in pkl_paths:
            fname = os.path.basename(model_path)
            meta_path = os.path.splitext(model_path)[0] + "_meta.json"
            metrics = self.evaluate_and_log(model_path, meta_path, X_val, y_val)
            if metrics:
                all_metrics[os.path.splitext(fname)[0]] = {
                    "metrics": metrics,
                    "model_path": os.path.abspath(model_path),
                    "meta_path": os.path.abspath(meta_path) if os.path.exists(meta_path) else None
                }

        if not all_metrics:
            logger.warning("No se evaluaron modelos correctamente; no hay ninguno para promover.")
            return

        # Guardar resumen clean_metrics en models/metrics/
        clean_metrics = {}
        for model_name, info in all_metrics.items():
            mets = info.get("metrics", {})
            try:
                clean_metrics[model_name] = {k: float(v) for k, v in mets.items() if isinstance(v, (int, float))}
            except Exception:
                clean_metrics[model_name] = mets
        try:
            with open(self.metrics_file, "w") as f:
                json.dump(clean_metrics, f, indent=2)
            logger.info("Archivo metrics_summary.json generado en %s", self.metrics_file)
        except Exception as e:
            logger.error("No se pudo escribir metrics_summary en %s: %s", self.metrics_file, e)

        # Calcular score ponderado para selección
        scored = []
        for model_name, info in all_metrics.items():
            mets = info.get("metrics", {})
            try:
                score = self._compute_selection_score(mets)
            except Exception as e:
                logger.error("Error calculando score para %s: %s", model_name, e)
                score = 0.0
            scored.append((model_name, info.get("model_path"), score, mets, info.get("meta_path")))

        # Ordenar por score descendente
        scored.sort(key=lambda x: x[2], reverse=True)
        best_model_name, best_model_path, best_score, best_metrics, best_meta = scored[0]
        logger.info("Mejor modelo: %s con score %s", best_model_name, best_score)

        # Promover modelo final (copia .pkl y validación dentro de final_model)
        try:
            self._promote_final_model(best_model_path, best_model_name, best_metrics)
        except Exception as e:
            logger.error("No se pudo promover el modelo final: %s", e)

        # --- Único run MLflow para selección + métricas de test ---
        test_metrics = {}
        try:
            with mlflow.start_run(run_name=f"final_selected_{best_model_name}_{int(time.time())}"):
                mlflow.log_param("selected_model", best_model_name)
                mlflow.log_metric("selection_score", float(best_score))

                # Evaluar en test (evaluate_on_test internamente guardará test metrics en final_model)
                test_metrics = self.evaluate_on_test(best_model_path)

                # Loggear métricas de test
                for k, v in test_metrics.items():
                    if v is not None:
                        mlflow.log_metric(k, float(v))

                mlflow.set_tag("selected_as_final", best_model_name)
        except Exception as e:
            logger.warning("No se pudo loggear selección y test metrics en MLflow: %s", e)

        # Guardar resumen local (selection_summary.json)
        try:
            selection_summary = {
                "model_name": best_model_name,
                "model_path": best_model_path,
                "score": float(best_score),
                "metrics": test_metrics,
                "selected_at": int(time.time())
            }
            selection_file = os.path.join(self.root_dir, "models", "selection_summary.json")
            with open(selection_file, "w") as sf:
                json.dump(selection_summary, sf, indent=2)
            logger.info("Selection summary guardado en %s", selection_file)
        except Exception as e:
            logger.warning("No se pudo guardar selection_summary: %s", e)


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
