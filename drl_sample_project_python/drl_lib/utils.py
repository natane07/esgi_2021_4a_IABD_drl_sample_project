import matplotlib.pyplot as plt
import numpy as np

def graph_score(title, scores, scale):
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Score moyen pour " + str(scale) + " parties")
    plt.plot(np.arange(1, len(scores) + 1), scores)
    fname = "./drl_lib/graph/" +title + ".png"
    plt.savefig(fname, dpi=72, bbox_inches='tight')
    plt.show()

def graph_score_bar(title, scores):
    fig = plt.figure(figsize=(10, 5))
    tile_x = ['Win', 'Loss/Egalite']
    plt.ylabel("Nombre de parties")
    plt.title(title)
    plt.bar(tile_x, scores,width=0.4)

    _nb = 0
    for i in scores:
      pct = (i / sum(scores)) * 100
      pct = round(pct,2)
      plt.annotate(str(pct)+'%', xy=(_nb, i+50), ha='center', va='bottom')
      _nb +=1
    fname = "./drl_lib/graph/" +title + ".png"
    plt.savefig(fname, dpi=72, bbox_inches='tight')
    plt.show()