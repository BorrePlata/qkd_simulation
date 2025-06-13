import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

def load_and_preprocess_data(test_size: float = 0.2, sample_size: int = None):
    # Cargar el conjunto de datos NSL-KDD desde archivos locales
    train_file_path = 'data/KDDTrain.txt'
    test_file_path = 'data/KDDTest+.txt'
    
    column_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
                    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
                    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files",
                    "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
                    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
                    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
                    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
                    "label"]

    # Cargar los datos
    train_df = pd.read_csv(train_file_path, names=column_names)
    test_df = pd.read_csv(test_file_path, names=column_names)

    # Combinar los datos de entrenamiento y prueba para preprocesar juntos
    df = pd.concat([train_df, test_df])

    # Verificar y corregir tipos de columnas
    for col in df.columns:
        if col not in ["protocol_type", "service", "flag", "label"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Eliminar filas con todos los valores NaN
    df.dropna(how='all', inplace=True)

    # Reducir el tamaño del conjunto de datos si se especifica un tamaño de muestra
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)

    # Codificar las etiquetas
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])

    # Convertir variables categóricas en variables dummy
    df = pd.get_dummies(df, columns=["protocol_type", "service", "flag"])

    # Manejar valores NaN e infinitos
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)

    # Verificar columnas con NaN y eliminarlas si existen
    nan_columns = df.columns[df.isna().any()].tolist()
    if nan_columns:
        print(f"Columnas con NaN después del preprocesamiento: {nan_columns}")
        # Imputar valores NaN con la media de nuevo
        for col in nan_columns:
            df[col].fillna(df[col].mean(), inplace=True)
        # Verificar de nuevo
        nan_columns = df.columns[df.isna().any()].tolist()
        if nan_columns:
            print(f"Aún quedan NaNs en las columnas: {nan_columns}")
            # Imputar valores NaN con un valor constante si es necesario
            for col in nan_columns:
                df[col].fillna(0, inplace=True)  # Alternativamente, puedes usar un valor constante como 0 o -1

    # Eliminar filas que aún contengan valores NaN después de la imputación
    df.dropna(inplace=True)

    # Verificar que hay suficientes datos después del preprocesamiento
    if df.shape[0] == 0:
        raise ValueError("No quedan datos después del preprocesamiento. Revisa el conjunto de datos y los pasos de preprocesamiento.")

    # Dividir los datos en características y etiquetas
    X = df.drop("label", axis=1)
    y = df["label"]

    # Imprimir los tipos de datos y la cantidad de datos después de la conversión
    print("Tipos de datos después de la conversión:")
    print(df.dtypes)
    print(f"Número de muestras después del preprocesamiento: {df.shape[0]}")

    # Estandarizar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=test_size, random_state=42)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data(sample_size=1000)
    print("Preprocesamiento completo y exitoso")
