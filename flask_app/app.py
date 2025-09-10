# flask_app/app.py
import os
import io
import glob
import time
import base64
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import threading
from typing import Optional, List, Dict

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, send_file, send_from_directory, Response, jsonify
)
from werkzeug.utils import secure_filename

# Try to import watchdog for event-based file watching (optional)
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except Exception:
    WATCHDOG_AVAILABLE = False

# Config
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_FINAL_DIR = os.path.join(BASE_DIR, "models", "final_model")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "results")
HISTORY_DIR = os.path.join(os.path.dirname(__file__), "history")  # <-- nueva carpeta para history
HISTORY_PATH = os.path.join(HISTORY_DIR, "history.json")          # <-- archivo de historial aquí
ALLOWED_EXT = {"csv"}

# ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(MODEL_FINAL_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)  # <- crea flask_app/history si hace falta

app = Flask(__name__)
app.secret_key = "cambia_esto_por_algo_seguro"  # en prod -> env var
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULTS_FOLDER"] = RESULTS_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB subida límite

# ---------------- Utilities to find/load artifacts ----------------
def _most_recent_file(pattern: str) -> Optional[str]:
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def load_model_artifacts():
    """
    Busca en models/final_model:
    - el .pkl del modelo (no importa el nombre, toma el más reciente)
    - el csv de feature importances (más reciente)
    - el json de metrics (más reciente)
    - intenta cargar imputer/scaler desde artifacts/ o models/final_model
    Devuelve dict con keys: model, fi_df, metrics, imputer_data, scaler_data, files, version_ts
    """
    out = {"model": None, "fi_df": None, "metrics": None,
           "imputer_data": None, "scaler_data": None,
           "files": {}}

    # modelo .pkl (cualquiera)
    pkl_path = _most_recent_file(os.path.join(MODEL_FINAL_DIR, "*.pkl"))
    if pkl_path:
        try:
            out["model"] = joblib.load(pkl_path)
            out["files"]["model_pkl"] = pkl_path
        except Exception as e:
            out["files"]["model_pkl_error"] = str(e)

    # feature importances .csv
    fi_path = _most_recent_file(os.path.join(MODEL_FINAL_DIR, "*.csv"))
    if fi_path:
        try:
            fi_df = pd.read_csv(fi_path)
            if {"feature", "importance"}.issubset(fi_df.columns):
                out["fi_df"] = fi_df.sort_values("importance", ascending=False)
                out["files"]["fi_csv"] = fi_path
            else:
                out["files"]["fi_csv_error"] = "CSV found but missing 'feature'/'importance' columns"
        except Exception as e:
            out["files"]["fi_csv_error"] = str(e)

    # metrics .json
    metrics_path = _most_recent_file(os.path.join(MODEL_FINAL_DIR, "*.json"))
    if metrics_path:
        try:
            with open(metrics_path, "r") as f:
                out["metrics"] = json.load(f)
            out["files"]["metrics_json"] = metrics_path
        except Exception as e:
            out["files"]["metrics_json_error"] = str(e)

    # imputer and scaler artifacts in artifacts/ (repo root) or inside final_model
    artifacts_candidates = [
        os.path.join(BASE_DIR, "artifacts", "imputer.pkl"),
        os.path.join(BASE_DIR, "artifacts", "scaler.pkl"),
        os.path.join(MODEL_FINAL_DIR, "imputer.pkl"),
        os.path.join(MODEL_FINAL_DIR, "scaler.pkl")
    ]
    for cand in artifacts_candidates:
        if os.path.exists(cand):
            try:
                data = joblib.load(cand)
                if isinstance(data, dict) and "imputer" in data:
                    out["imputer_data"] = data
                    out["files"]["imputer"] = cand
                elif isinstance(data, dict) and "scaler" in data:
                    out["scaler_data"] = data
                    out["files"]["scaler"] = cand
                else:
                    if isinstance(data, dict) and "imputer" in data:
                        out["imputer_data"] = data
                        out["files"]["imputer"] = cand
                    if isinstance(data, dict) and "scaler" in data:
                        out["scaler_data"] = data
                        out["files"]["scaler"] = cand
            except Exception as e:
                out.setdefault("files_errors", []).append({cand: str(e)})

    # fallback: look for any pkl in artifacts folder
    more = glob.glob(os.path.join(BASE_DIR, "artifacts", "*.pkl"))
    for cand in more:
        if cand in out.get("files", {}).values():
            continue
        try:
            data = joblib.load(cand)
            if isinstance(data, dict) and "imputer" in data and out["imputer_data"] is None:
                out["imputer_data"] = data
                out["files"]["imputer"] = cand
            if isinstance(data, dict) and "scaler" in data and out["scaler_data"] is None:
                out["scaler_data"] = data
                out["files"]["scaler"] = cand
        except Exception:
            continue

    out["version_ts"] = time.time()
    return out

# initial artifacts load
ARTIFACTS = load_model_artifacts()

# ---------------- History helpers ----------------
_history_lock = threading.Lock()

def load_history() -> List[Dict]:
    """Carga el historial desde HISTORY_PATH"""
    try:
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
    except Exception as e:
        app.logger.exception("Error loading history file: %s", e)
    return []

def save_history(history: List[Dict]) -> None:
    """Guarda el historial de forma atómica"""
    tmp = HISTORY_PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        os.replace(tmp, HISTORY_PATH)
    except Exception as e:
        app.logger.exception("Error saving history file: %s", e)
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass

def _append_history_entry(new_artifacts: dict, note: str = "") -> None:
    """
    Añade entrada al historial si no es duplicada.
    Guarda campos útiles: ts, human_ts, model_pkl, fi_csv, metrics (small), imputer, scaler, note
    """
    try:
        with _history_lock:
            history = load_history()
            entry = {
                "ts": int(time.time()),
                "human_ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "model_pkl": new_artifacts.get("files", {}).get("model_pkl"),
                "fi_csv": new_artifacts.get("files", {}).get("fi_csv"),
                "metrics": new_artifacts.get("metrics"),
                "imputer": new_artifacts.get("files", {}).get("imputer"),
                "scaler": new_artifacts.get("files", {}).get("scaler"),
                "note": note
            }
            # avoid duplicate consecutive entries (same model_pkl and same ts file)
            if history:
                last = history[-1]
                if last.get("model_pkl") == entry.get("model_pkl") and last.get("fi_csv") == entry.get("fi_csv"):
                    app.logger.debug("History: duplicate detected, skipping append.")
                    return
            history.append(entry)
            # keep bounded size to avoid growth infinite (optional)
            MAX_HISTORY = 500
            if len(history) > MAX_HISTORY:
                history = history[-MAX_HISTORY:]
            save_history(history)
            app.logger.info("Appended history entry: %s", entry.get("model_pkl"))
    except Exception as e:
        app.logger.exception("Failed to append history entry: %s", e)

# ---------------- helper: validate new artifacts ----------------
def _validate_new_artifacts(candidate: dict) -> bool:
    """
    Sanity check on candidate artifacts.
    - model must be present
    - if columns provided, build a dummy row (NaNs) and try imputer->scaler->predict
    Returns True if validation passes.
    """
    try:
        model = candidate.get("model")
        if model is None:
            return False

        # determine columns order to build dummy
        cols = None
        if candidate.get("imputer_data") and "columns" in candidate["imputer_data"]:
            cols = list(candidate["imputer_data"]["columns"])
        elif candidate.get("scaler_data") and "columns" in candidate["scaler_data"]:
            cols = list(candidate["scaler_data"]["columns"])
        else:
            fi = candidate.get("fi_df")
            if fi is not None and "feature" in fi.columns:
                cols = fi["feature"].tolist()

        if not cols:
            # minimal predict check if no columns info
            try:
                _ = model.predict(np.zeros((1, 1)))
                return True
            except Exception:
                # accept model presence but warn
                return True

        # build dummy df with NaNs (imputer should handle or scaler)
        df_dummy = pd.DataFrame([[np.nan] * len(cols)], columns=cols)

        # imputer
        if candidate.get("imputer_data") and "imputer" in candidate["imputer_data"]:
            try:
                imputer = candidate["imputer_data"]["imputer"]
                arr = imputer.transform(df_dummy)
                df_dummy = pd.DataFrame(arr, columns=cols)
            except Exception as e:
                app.logger.warning("Sanity imputer transform failed during validation: %s", e)
                # continue to scaler/predict attempt

        # scaler
        if candidate.get("scaler_data") and "scaler" in candidate["scaler_data"]:
            try:
                scaler = candidate["scaler_data"]["scaler"]
                arr = scaler.transform(df_dummy)
                df_dummy = pd.DataFrame(arr, columns=cols)
            except Exception as e:
                app.logger.warning("Sanity scaler transform failed during validation: %s", e)

        # predict
        try:
            _ = model.predict(df_dummy)
            return True
        except Exception as e:
            app.logger.exception("Sanity predict failed during validation: %s", e)
            return False

    except Exception as e:
        app.logger.exception("Unexpected error during artifact validation: %s", e)
        return False

# ---------------- artifact swap utilities ----------------
_artifact_lock = threading.Lock()

def _swap_artifacts_if_valid(new_artifacts: dict, note: str = "watcher-detected") -> bool:
    """
    Validate and swap ARTIFACTS atomically under lock.
    If swapped, append history entry.
    Returns True if swapped, False otherwise.
    """
    try:
        if not _validate_new_artifacts(new_artifacts):
            app.logger.warning("New artifacts failed validation; not swapping.")
            return False

        with _artifact_lock:
            global ARTIFACTS
            ARTIFACTS = new_artifacts
            ARTIFACTS["version_ts"] = time.time()

        # append history entry
        try:
            _append_history_entry(new_artifacts, note=note)
        except Exception:
            app.logger.exception("Failed to append history after swap.")

        app.logger.info("ARTIFACTS swapped successfully at %s", ARTIFACTS["version_ts"])
        return True
    except Exception as e:
        app.logger.exception("Error swapping artifacts: %s", e)
        return False

# ---------------- File-watcher (watchdog if available; else polling) ----------------
_reload_debounce_seconds = 2.0
_last_reload_attempt = 0.0

def _schedule_reload(reason: str = ""):
    """
    Schedules an immediate attempt to load+swap artifacts.
    Debounce to avoid repeated reloads during file writes.
    """
    global _last_reload_attempt
    now = time.time()
    if now - _last_reload_attempt < _reload_debounce_seconds:
        app.logger.debug("Reload suppressed by debounce (%s)", reason)
        return
    _last_reload_attempt = now
    def _do_reload():
        app.logger.info("Scheduled reload triggered (%s). Loading artifacts...", reason)
        try:
            new_artifacts = load_model_artifacts()
        except Exception as e:
            app.logger.exception("Failed to load artifacts during scheduled reload: %s", e)
            return
        _swap_artifacts_if_valid(new_artifacts, note=reason)
    threading.Thread(target=_do_reload, daemon=True).start()

if WATCHDOG_AVAILABLE:
    class _FSHandler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory: return
            app.logger.debug("watchdog: created %s", event.src_path)
            _schedule_reload("created "+os.path.basename(event.src_path))
        def on_modified(self, event):
            if event.is_directory: return
            app.logger.debug("watchdog: modified %s", event.src_path)
            _schedule_reload("modified "+os.path.basename(event.src_path))
        def on_moved(self, event):
            if event.is_directory: return
            app.logger.debug("watchdog: moved %s", event.src_path)
            _schedule_reload("moved "+os.path.basename(event.src_path))
        def on_deleted(self, event):
            if event.is_directory: return
            app.logger.debug("watchdog: deleted %s", event.src_path)
            _schedule_reload("deleted "+os.path.basename(event.src_path))

    def _start_watchdog(paths_to_watch=None):
        paths = paths_to_watch or [MODEL_FINAL_DIR, os.path.join(BASE_DIR, "artifacts")]
        observer = Observer()
        handler = _FSHandler()
        for p in paths:
            if os.path.isdir(p):
                observer.schedule(handler, path=p, recursive=False)
        observer.daemon = True
        observer.start()
        app.logger.info("watchdog observer started for paths: %s", paths)
        return observer

else:
    # Polling fallback (lightweight)
    def _polling_watcher(poll_interval: float = 5.0):
        app.logger.info("Polling watcher started (interval %s s)", poll_interval)
        prev = {}
        def _scan_paths():
            paths = []
            for pattern in (os.path.join(MODEL_FINAL_DIR, "*.pkl"),
                            os.path.join(MODEL_FINAL_DIR, "*.csv"),
                            os.path.join(MODEL_FINAL_DIR, "*.json")):
                p = _most_recent_file(pattern)
                if p: paths.append(p)
            artifacts_dir = os.path.join(BASE_DIR, "artifacts")
            for name in ("imputer.pkl", "scaler.pkl"):
                fp = os.path.join(artifacts_dir, name)
                if os.path.exists(fp): paths.append(fp)
                fp2 = os.path.join(MODEL_FINAL_DIR, name)
                if os.path.exists(fp2): paths.append(fp2)
            return list(dict.fromkeys(paths))

        nonlocal_prev = {"prev": prev}
        while True:
            try:
                paths = _scan_paths()
                cur = {}
                for p in paths:
                    try:
                        cur[p] = os.path.getmtime(p)
                    except Exception:
                        cur[p] = 0.0
                changed = False
                all_keys = set(list(nonlocal_prev["prev"].keys()) + list(cur.keys()))
                for k in all_keys:
                    if nonlocal_prev["prev"].get(k) != cur.get(k):
                        changed = True
                        app.logger.debug("Polling detected change candidate: %s", k)
                        break
                if changed:
                    _schedule_reload("poll-detect")
                nonlocal_prev["prev"] = cur
                time.sleep(poll_interval)
            except Exception as e:
                app.logger.exception("Polling watcher error: %s", e)
                time.sleep(poll_interval)

# Start watcher on first request
def _start_background_watcher():
    try:
        if WATCHDOG_AVAILABLE:
            _start_watchdog(paths_to_watch=[MODEL_FINAL_DIR, os.path.join(BASE_DIR, "artifacts")])
        else:
            t = threading.Thread(target=_polling_watcher, args=(5.0,), daemon=True)
            t.start()
    except Exception as e:
        app.logger.exception("Failed to start watcher: %s", e)

# provide manual reload endpoint (POST) as a fallback / admin button
@app.route("/reload_model", methods=["POST"])
def reload_model():
    try:
        new_artifacts = load_model_artifacts()
        swapped = _swap_artifacts_if_valid(new_artifacts, note="manual-reload")
        if swapped:
            flash("Model/artifacts reloaded successfully.", "success")
        else:
            flash("New artifacts failed validation. Current artifacts preserved.", "warning")
    except Exception as e:
        app.logger.exception("Manual reload failed: %s", e)
        flash(f"Manual reload failed: {e}", "danger")
    return redirect(url_for("index"))

# expose artifact version/time to templates
@app.context_processor
def inject_artifact_info():
    version_ts = ARTIFACTS.get("version_ts")
    if version_ts:
        nice = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(version_ts))
    else:
        mp = ARTIFACTS.get("files", {}).get("model_pkl")
        if mp and os.path.exists(mp):
            nice = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(mp)))
        else:
            nice = None
    return {"artifact_version_ts": nice}

# ---------------- helpers for transform & predict ----------------
def ensure_columns_order(df: pd.DataFrame, cols_order):
    missing = [c for c in cols_order if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = np.nan
    return df[cols_order].copy()

def transform_and_predict(df_inputs: pd.DataFrame):
    imputer_data = ARTIFACTS.get("imputer_data")
    scaler_data = ARTIFACTS.get("scaler_data")
    model = ARTIFACTS.get("model")
    if model is None:
        raise RuntimeError("Model not found.")

    # choose columns order (prefer imputer columns)
    cols = None
    if imputer_data and "columns" in imputer_data:
        cols = list(imputer_data["columns"])
    elif scaler_data and "columns" in scaler_data:
        cols = list(scaler_data["columns"])
    else:
        cols = list(df_inputs.columns)

    df_ordered = ensure_columns_order(df_inputs, cols)

    # impute if imputer present
    if imputer_data and "imputer" in imputer_data and imputer_data["imputer"] is not None:
        imputer = imputer_data["imputer"]
        arr = imputer.transform(df_ordered)
        df_imputed = pd.DataFrame(arr, columns=cols)
    else:
        df_imputed = df_ordered

    # scale if scaler present
    if scaler_data and "scaler" in scaler_data and scaler_data["scaler"] is not None:
        scaler = scaler_data["scaler"]
        arr = scaler.transform(df_imputed)
        df_scaled = pd.DataFrame(arr, columns=cols)
    else:
        df_scaled = df_imputed

    preds = model.predict(df_scaled)
    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(df_scaled)[:, 1]
        except Exception:
            probs = None

    return preds, probs, cols

# ---------------- Routes ----------------
@app.route("/")
def index():
    artifacts = ARTIFACTS
    metrics = artifacts.get("metrics")
    fi_df = artifacts.get("fi_df")
    fi_img_b64 = None
    if fi_df is not None:
        buf = io.BytesIO()
        plt.figure(figsize=(6, 0.6 * min(10, len(fi_df))))
        top = fi_df.head(8).sort_values("importance")
        plt.barh(top["feature"], top["importance"])
        plt.title("Top features preview")
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=120)
        plt.close()
        buf.seek(0)
        fi_img_b64 = base64.b64encode(buf.read()).decode("ascii")
    return render_template("index.html", artifacts=artifacts, metrics=metrics, fi_img_b64=fi_img_b64)

@app.route("/predict", methods=["POST"])
def predict():
    if ARTIFACTS.get("imputer_data") is None and ARTIFACTS.get("scaler_data") is None:
        flash("Imputer/scaler not found - prediction might be unreliable", "warning")

    cols = None
    if ARTIFACTS.get("imputer_data") and "columns" in ARTIFACTS["imputer_data"]:
        cols = list(ARTIFACTS["imputer_data"]["columns"])
    elif ARTIFACTS.get("scaler_data") and "columns" in ARTIFACTS["scaler_data"]:
        cols = list(ARTIFACTS["scaler_data"]["columns"])
    else:
        flash("No columns info saved in artifacts. Cannot predict reliably.", "danger")
        return redirect(url_for("index"))

    data = {}
    missing_any = False
    for c in cols:
        val = request.form.get(c, "").strip()
        if val == "":
            data[c] = np.nan
            missing_any = True
        else:
            try:
                data[c] = float(val)
            except ValueError:
                data[c] = np.nan
                missing_any = True

    df_input = pd.DataFrame([data], columns=cols)

    try:
        preds, probs, used_cols = transform_and_predict(df_input)
    except Exception as e:
        flash(f"Error during transform/predict: {e}", "danger")
        return redirect(url_for("index"))

    cls = int(preds[0])
    prob = float(probs[0]) if (probs is not None) else None
    label = "Fraudulent" if cls == 1 else "Authentic"

    return render_template("index.html",
                           single_pred=True,
                           pred_label=label,
                           pred_prob=prob,
                           input_data=data,
                           artifacts=ARTIFACTS,
                           used_cols=used_cols)

@app.route("/importance")
def importance():
    fi_df = ARTIFACTS.get("fi_df")
    if fi_df is None:
        flash("Feature importances not available.", "warning")
        return redirect(url_for("index"))

    fig, ax = plt.subplots(figsize=(8, max(4, 0.18 * len(fi_df))))
    df = fi_df.copy()
    df = df.sort_values("importance", ascending=True)
    n = len(df)
    colors = []
    imps = (df["importance"].astype(float) - df["importance"].min()) / max(1e-9, (df["importance"].max() - df["importance"].min()))
    for idx, val in enumerate(imps):
        if (n - idx) <= 3:
            colors.append("#0b5394")
        else:
            shade = 0.4 + 0.6 * val
            colors.append((0.2 * shade, 0.4 * shade, 0.9 * shade))
    ax.barh(df["feature"], df["importance"], color=colors)
    ax.set_title("Features en orden de importancia")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("ascii")

    return render_template("importance.html", fi_img_b64=img_b64, fi_df=ARTIFACTS.get("fi_df"))

@app.route("/bulk", methods=["GET", "POST"])
def bulk():
    if request.method == "GET":
        return render_template("bulk.html")

    # 1) Validaciones básicas del upload
    if "file" not in request.files:
        flash("No file part", "danger")
        return redirect(request.url)
    file = request.files["file"]
    if file.filename == "":
        flash("No selected file", "danger")
        return redirect(request.url)
    filename = secure_filename(file.filename)
    if "." not in filename or filename.rsplit(".", 1)[1].lower() not in ALLOWED_EXT:
        flash("Only CSV files allowed", "danger")
        return redirect(request.url)

    preprocessed = request.form.get("preprocessed", "no") == "yes"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{int(time.time())}_{filename}")
    file.save(save_path)

    # Guardamos original_df EXACTO como fue subido (antes de dropear columnas / reordenar)
    try:
        original_df = pd.read_csv(save_path)
    except Exception as e:
        # intentar limpiar el archivo subido si hay error al leer
        try:
            os.remove(save_path)
        except Exception:
            pass
        flash(f"Error reading CSV: {e}", "danger")
        return redirect(request.url)

    # Hacemos una copia de trabajo (sera la que preprocesamos y pasamos al modelo)
    df = original_df.copy()

    try:
        # 2) Si el usuario indica que NO está preprocesado: eliminar id (si existe), reordenar columnas,
        #    aplicar imputer y scaler en ese orden (si existen).
        if not preprocessed:
            # quitamos id aquí para el pipeline, pero original_df sigue manteniendo la columna id
            if "id" in df.columns:
                df = df.drop(columns=["id"])

            # obtener orden de columnas desde artifacts
            cols = None
            if ARTIFACTS.get("imputer_data") and "columns" in ARTIFACTS["imputer_data"]:
                cols = list(ARTIFACTS["imputer_data"]["columns"])
            elif ARTIFACTS.get("scaler_data") and "columns" in ARTIFACTS["scaler_data"]:
                cols = list(ARTIFACTS["scaler_data"]["columns"])
            else:
                flash("No artifacts with columns info available to preprocess file.", "danger")
                # borrar upload para no dejar basura
                try: os.remove(save_path)
                except Exception: pass
                return redirect(request.url)

            # Reindex / añadir las columnas faltantes con NaN si hace falta
            df = ensure_columns_order(df, cols)

            # Imputer (si existe)
            if ARTIFACTS.get("imputer_data") and "imputer" in ARTIFACTS["imputer_data"]:
                imputer = ARTIFACTS["imputer_data"]["imputer"]
                df = pd.DataFrame(imputer.transform(df), columns=cols)

            # Scaler (si existe)
            if ARTIFACTS.get("scaler_data") and "scaler" in ARTIFACTS["scaler_data"]:
                scaler = ARTIFACTS["scaler_data"]["scaler"]
                df = pd.DataFrame(scaler.transform(df), columns=cols)

        else:
            # 3) Si ya está preprocesado: sólo nos aseguramos del orden de columnas para el modelo
            if ARTIFACTS.get("scaler_data") and "columns" in ARTIFACTS["scaler_data"]:
                cols = list(ARTIFACTS["scaler_data"]["columns"])
                df = ensure_columns_order(df, cols)
            else:
                # si no hay columnas info, aceptamos la tabla tal cual (pero avisamos)
                flash("Preprocessed flag set but no columns info in artifacts — using provided CSV as-is.", "warning")

        # 4) Predecir en df (este df está en formato que espera el modelo)
        preds, probs, used_cols = transform_and_predict(df)

        # 5) Construir df de salida: **original_df** + columnas de predicción
        df_out = original_df.copy()

        # Asegurarse de que el número de filas coincida entre original y predicción
        if len(df_out) != len(preds):
            # Si el tamaño no bate, esto es un error serio: notificamos y no guardamos
            flash("Number of predictions does not match number of rows in uploaded file.", "danger")
            # intentar limpiar upload
            try: os.remove(save_path)
            except Exception: pass
            return redirect(request.url)

        df_out["predicted_class"] = preds
        if probs is not None:
            df_out["predicted_proba"] = probs

        # 6) Guardar resultado final con columnas originales + predicción
        result_name = f"predicted_{int(time.time())}_{filename}"
        result_path = os.path.join(app.config["RESULTS_FOLDER"], result_name)
        df_out.to_csv(result_path, index=False)

        # 7) Generar grafico resumen (fraud vs authentic)
        counts = pd.Series(preds).value_counts().sort_index()
        labels = ["Authentic", "Fraudulent"]
        values = [int(counts.get(0, 0)), int(counts.get(1, 0))]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, values, color=["#1f77b4", "#d62728"])
        ax.set_title("Comparativa: Fraudulent vs Authentic")
        ax.set_ylabel("Count")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120)
        plt.close()
        buf.seek(0)
        bar_b64 = base64.b64encode(buf.read()).decode("ascii")

        flash(f"File processed and predictions saved: {result_name}", "success")

        # 8) LIMPIEZA: borrar el CSV subido de uploads/ — lo hacemos al final para asegurarnos de que
        #    la lectura/transform/predict terminaron correctamente y no dejar ficheros temporales.
        try:
            os.remove(save_path)
            app.logger.debug("Uploaded file removed: %s", save_path)
        except Exception as e:
            app.logger.warning("Could not remove uploaded file %s: %s", save_path, e)

        # 9) Devolver template con info del resultado (archivo guardado y gráfico)
        return render_template("bulk.html", result_file=result_name, bar_b64=bar_b64, counts=values)

    except Exception as e:
        # intentar borrar upload en caso de error para no acumular basura
        try:
            if os.path.exists(save_path):
                os.remove(save_path)
        except Exception:
            pass
        flash(f"Error processing file: {e}", "danger")
        return redirect(request.url)


@app.route("/download/<path:filename>")
def download_file(filename):
    full_path = os.path.join(app.config["RESULTS_FOLDER"], filename)
    if not os.path.exists(full_path):
        flash("File not found.", "danger")
        return redirect(url_for("bulk"))
    return send_file(full_path, as_attachment=True)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/results/<path:filename>")
def results_file(filename):
    return send_from_directory(app.config["RESULTS_FOLDER"], filename)

# ---------------- History endpoints & SSE ----------------
@app.route("/history")
def history_page():
    history = load_history()
    latest = history[-1] if history else None
    # Render a template history.html (you must create it under templates/)
    # Pass history list and latest
    return render_template("history.html", history=history, latest=latest)

@app.route("/history.json")
def history_json():
    history = load_history()
    return jsonify(history)

def _sse_pack(data: str, event: Optional[str] = None) -> str:
    """Format data for SSE"""
    s = ""
    if event:
        s += f"event: {event}\n"
    # escape newlines in data
    for line in data.splitlines():
        s += f"data: {line}\n"
    s += "\n"
    return s

@app.route("/history/stream")
def history_stream():
    """
    Server-Sent Events endpoint. Clients can connect and will receive a small JSON
    payload when the history file is updated. This is a lightweight poll-based SSE:
    it checks the mtime of HISTORY_PATH and yields when it changes.
    """
    def gen():
        last_mtime = os.path.getmtime(HISTORY_PATH) if os.path.exists(HISTORY_PATH) else 0.0
        # Immediately send current state once
        try:
            initial = load_history()
            payload = json.dumps({"type": "init", "history_len": len(initial)})
            yield _sse_pack(payload, event="history_init")
        except Exception:
            pass
        while True:
            try:
                time.sleep(1.0)
                cur = os.path.getmtime(HISTORY_PATH) if os.path.exists(HISTORY_PATH) else 0.0
                if cur != last_mtime:
                    last_mtime = cur
                    history = load_history()
                    payload = json.dumps({"type": "update", "history_len": len(history), "latest": history[-1] if history else None})
                    yield _sse_pack(payload, event="history_update")
            except GeneratorExit:
                break
            except Exception:
                time.sleep(1.0)
                continue
    return Response(gen(), mimetype="text/event-stream")

# ---------------- run ----------------
if __name__ == "__main__":
    # ensure initial load is fresh
    ARTIFACTS = load_model_artifacts()

    # add initial history entry if none exists
    try:
        if not os.path.exists(HISTORY_PATH):
            _append_history_entry(ARTIFACTS, note="initial-load")
    except Exception:
        pass

    # start watcher in background thread
    try:
        if WATCHDOG_AVAILABLE:
            _start_watchdog(paths_to_watch=[MODEL_FINAL_DIR, os.path.join(BASE_DIR, "artifacts")])
        else:
            t = threading.Thread(target=_polling_watcher, args=(5.0,), daemon=True)
            t.start()
    except Exception as e:
        app.logger.exception("Failed to start watcher: %s", e)

    # run Flask
    app.run(host="0.0.0.0", port=5000, debug=True)