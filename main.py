# MIT License
# Copyright (c) 2025 Samuel Plata
# Este archivo es parte del proyecto QKD Simulation y está licenciado bajo la Licencia MIT.


import sys
import os
import numpy as np
import logging
from simulation.qkd_bioinspired import QKDProtocolBioinspired
from config import Config
from utils.visualization import plot_keys
from optimization import MushMindOptimization
from data_preparation import load_and_preprocess_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def calculate_error_rate(key, received_key):
    errors = np.sum(key != received_key)
    return errors / len(key)

def main():
    try:
        # Define parameter space for optimization
        logging.info("Definiendo espacio de parámetros para la optimización.")
        param_space = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (50, 100, 50), (100, 100, 50)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'max_iter': [200]
        }

        # Load real data with a sample size
        logging.info("Cargando y preprocesando los datos.")
        X_train, X_test, y_train, y_test = load_and_preprocess_data(sample_size=10000)

        logging.info("Iniciando optimización.")
        optimizer = MushMindOptimization(n_agents=Config.N_AGENTS)
        best_params, best_score = optimizer.optimize(param_space, X_train, y_train, iterations=Config.OPTIMIZATION_ITERATIONS)

        logging.info(f"Optimización completada. Best params: {best_params}")
        logging.info(f"Best score: {best_score:.4f}")

        # Use optimized parameters for QKD simulation
        logging.info("Ejecutando simulación QKD bioinspirada.")
        protocol = QKDProtocolBioinspired(
            wavelength=Config.LASER_WAVELENGTH,
            distance=Config.DISTANCE,
            efficiency=Config.DETECTOR_EFFICIENCY,
            error_rate=Config.ERROR_RATE,
            node_density=Config.NODE_DENSITY
        )

        key, received_key, errors = protocol.simulate()
        logging.info("Simulación completada. Graficando claves.")
        plot_keys(key, received_key)
        error_rate = calculate_error_rate(key, received_key)
        logging.info(f"Observed Error Rate: {error_rate}")

        summary = optimizer.get_optimization_summary()
        logging.info("\nOptimization Summary:")
        logging.info(summary.describe())

        # Confirmación de finalización
        print("Script main.py ejecutado con éxito.")
        logging.info("Script main.py ejecutado con éxito.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"Error en main.py: {e}")

if __name__ == "__main__":
    main()
