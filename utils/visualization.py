import matplotlib.pyplot as plt

def plot_keys(key, received_key):
    plt.figure(figsize=(10, 6))
    plt.plot(key, label='Generated Key', color='blue')
    plt.plot(received_key, label='Received Key', color='red', linestyle='dashed')
    plt.legend()
    plt.title('Comparison of Generated and Received Keys')
    plt.xlabel('Bit Index')
    plt.ylabel('Bit Value')
    plt.show()
