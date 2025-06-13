import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from config import Config

class QKDProtocolBioinspired:
    def __init__(self, wavelength, distance, efficiency, error_rate, node_density):
        self.wavelength = wavelength
        self.distance = distance
        self.efficiency = efficiency
        self.error_rate = error_rate
        self.node_density = node_density  # Densidad de nodos en la red
    
    def quorum_sensing(self):
        # Ajuste de parámetros basado en la densidad de nodos
        adjustment_factor = 1 + (self.node_density - 1) * 0.1
        self.error_rate *= adjustment_factor
        self.efficiency /= adjustment_factor
    
    def simulate(self):
        self.quorum_sensing()  # Ajustar parámetros antes de la simulación
        key_length = 1024
        key = np.random.randint(2, size=key_length)
        errors = np.random.binomial(1, self.error_rate, key_length)
        received_key = key ^ errors
        
        return key, received_key, errors

if __name__ == "__main__":
    protocol = QKDProtocolBioinspired(Config.LASER_WAVELENGTH, Config.DISTANCE, Config.DETECTOR_EFFICIENCY, Config.ERROR_RATE, Config.NODE_DENSITY)
    key, received_key, errors = protocol.simulate()
    print(f"Generated Key: {key}")
    print(f"Received Key: {received_key}")
    print(f"Errors: {errors}")
