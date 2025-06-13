import sys
import os
import unittest
import numpy as np
from simulation.qkd_bioinspired import QKDProtocolBioinspired
from config import Config
from optimization import AdvancedMycoOptimization
from data_preparation import load_and_preprocess_data

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestQKDProtocolBioinspired(unittest.TestCase):
    def setUp(self):
        self.protocol = QKDProtocolBioinspired(Config.LASER_WAVELENGTH, Config.DISTANCE, Config.DETECTOR_EFFICIENCY, Config.ERROR_RATE, Config.NODE_DENSITY)

    def test_simulate(self):
        key, received_key, errors = self.protocol.simulate()
        self.assertEqual(len(key), 1024)
        self.assertEqual(len(received_key), 1024)
        self.assertEqual(len(errors), 1024)

    def test_optimization(self):
        param_space = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (50, 100, 50), (100, 100, 50)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'max_iter': [200]
        }

        X_train, X_test, y_train, y_test = load_and_preprocess_data()
        optimizer = AdvancedMycoOptimization(n_agents=Config.N_AGENTS)
        best_params, best_score = optimizer.optimize(param_space, X_train, y_train, iterations=Config.OPTIMIZATION_ITERATIONS)

        self.assertIsInstance(best_params, dict)
        self.assertGreater(best_score, 0)

if __name__ == '__main__':
    unittest.main()
