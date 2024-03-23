import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
#labeled faces in the wild, dataset di foto di personaggi famosi con labels associati

plt.rcParams["figure.dpi"]=120
rng=np.random.RandomState(1)

faces=fetch_lfw_people(min_faces_per_person=50)
#vogliamo le persone che hanno almeno 50 fotografie associate

print(faces.target_names)