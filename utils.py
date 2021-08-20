import matplotlib.pyplot as plt

def draw_graph(accs):
    plt.plot(accs)
    plt.title("Test Accuracy")
    plt.xlabel("# Epochs")
    plt.ylabel("Accuracy (%)")
    plt.show()
