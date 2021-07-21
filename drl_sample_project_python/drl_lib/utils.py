import matplotlib.pyplot as plt
import numpy as np

def graph_score(title, scores, scale):
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Score moyen pour " + str(scale) + " parties")
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.show()

def graph_score_bar(title, scores):
    fig = plt.figure(figsize=(10, 5))
    tile_x = ['Win', 'Loss/Egalite']
    plt.bar(tile_x, scores,width=0.4)
    plt.ylabel("Nombre de parties")
    plt.title(title)
    plt.show()