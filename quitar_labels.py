import pandas as pd

# Cargar CSV
df = pd.read_csv("/home/jorge/DocumentsWLS/Data_Science_Projects/MLOPS-project-Credit-Card-Fraud-Detection/data/raw/predict.csv")

# Eliminar última columna
df = df.iloc[:, :-1]

# Guardar de nuevo
df.to_csv("predict_sin_labels.csv", index=False)