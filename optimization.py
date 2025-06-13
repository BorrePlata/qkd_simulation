import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from joblib import Parallel, delayed
import logging
import warnings
from typing import Dict, List, Tuple, Any

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MushMindOptimization:
    def __init__(self, n_agents: int, decay: float = 0.1, alpha: float = 1, beta: float = 2,
                 adaptation_rate: float = 0.05, exploration_rate: float = 0.1):
        self.n_agents = n_agents
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.adaptation_rate = adaptation_rate
        self.exploration_rate = exploration_rate
        self.best_params = None
        self.best_score = -np.inf
        self.pheromones = {}
        self.learning_rate = 0.01
        self.history = []

    def initialize_pheromones(self, param_space: Dict[str, List[Any]]) -> None:
        self.pheromones = {param: np.ones(len(values)) for param, values in param_space.items()}

    def choose_params(self, param_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        return {
            param: (random.choice(values) if np.random.rand() < self.exploration_rate
                    else random.choices(values, weights=self.pheromones[param]**self.alpha / np.sum(self.pheromones[param]**self.alpha), k=1)[0])
            for param, values in param_space.items()
        }

    def update_pheromones(self, param_space: Dict[str, List[Any]], chosen_params: Dict[str, Any], score: float) -> None:
        for param, values in param_space.items():
            index = values.index(chosen_params[param])
            self.pheromones[param][index] += score * self.beta
            self.pheromones[param] = self.pheromones[param] * (1 - self.decay) + self.decay

    def adjust_hyperparameters(self, score: float) -> None:
        if score > self.best_score:
            self.learning_rate *= (1 + self.adaptation_rate)
            self.exploration_rate *= (1 - self.adaptation_rate)
        else:
            self.learning_rate *= (1 - self.adaptation_rate)
            self.exploration_rate *= (1 + self.adaptation_rate)
        
        self.learning_rate = np.clip(self.learning_rate, 0.0001, 0.1)
        self.exploration_rate = np.clip(self.exploration_rate, 0.01, 0.5)

    def evaluate_model(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        model = MLPClassifier(hidden_layer_sizes=params['hidden_layer_sizes'],
                              activation=params['activation'],
                              solver=params['solver'],
                              learning_rate_init=self.learning_rate,
                              max_iter=200,
                              early_stopping=True,
                              n_iter_no_change=10,
                              validation_fraction=0.1)
        
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
        accuracy = np.mean(scores)
        
        model.fit(X, y)
        y_pred = model.predict(X)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
        
        return accuracy, {'precision': precision, 'recall': recall, 'f1': f1}

    def optimize(self, param_space: Dict[str, List[Any]], X: np.ndarray, y: np.ndarray, iterations: int = 100) -> Tuple[Dict[str, Any], float]:
        self.initialize_pheromones(param_space)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for iteration in range(iterations):
            logging.info(f"Iteration {iteration + 1}/{iterations}")
            
            results = Parallel(n_jobs=-1)(
                delayed(self.agent_optimization)(param_space, X_scaled, y)
                for _ in range(self.n_agents)
            )

            for chosen_params, score, metrics in results:
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = chosen_params
                    logging.info(f"New best score: {score:.4f}, Params: {chosen_params}")
                
                self.update_pheromones(param_space, chosen_params, score)
                self.adjust_hyperparameters(score)
                
                self.history.append({
                    'iteration': iteration,
                    'params': chosen_params,
                    'score': score,
                    'metrics': metrics
                })

            if self.check_convergence():
                logging.info(f"Converged after {iteration + 1} iterations")
                break

        return self.best_params, self.best_score

    def agent_optimization(self, param_space: Dict[str, List[Any]], X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, Any], float, Dict[str, float]]:
        chosen_params = self.choose_params(param_space)
        score, metrics = self.evaluate_model(X, y, chosen_params)
        return chosen_params, score, metrics

    def check_convergence(self, window: int = 10, threshold: float = 0.001) -> bool:
        if len(self.history) < window:
            return False
        recent_scores = [entry['score'] for entry in self.history[-window:]]
        return np.std(recent_scores) < threshold

    def get_optimization_summary(self) -> pd.DataFrame:
        df = pd.DataFrame(self.history)
        df['learning_rate'] = [self.learning_rate] * len(df)
        df['exploration_rate'] = [self.exploration_rate] * len(df)
        return df

def load_and_preprocess_data(test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.datasets import load_digits
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=test_size, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    param_space = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (50, 100, 50), (100, 100, 50)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam']
    }
    
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    optimizer = MushMindOptimization(n_agents=10)
    best_params, best_score = optimizer.optimize(param_space, X_train, y_train, iterations=50)

    logging.info(f"Optimization completed. Best params: {best_params}")
    logging.info(f"Best score: {best_score:.4f}")

    summary = optimizer.get_optimization_summary()
    logging.info("\nOptimization Summary:")
    logging.info(summary.describe())
