#!preambule! Chargement packages
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
import tensorflow as tf
import matplotlib.pyplot as plt 
from sklearn import preprocessing

#!Preambule! Configuration graphique 
# Création du dossier de sortie 
OUTPUT_DIR = os.path.join('output', 'preprocess')
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Fonction de sauvegarde
def save_fig(fig, filename):
    """Sauvegarde une figure dans le dossier output/preprocess/"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Figure sauvegardée: {filepath}")

# Chargement du dataset
df = pd.read_csv('data/usaWithWeather.csv')

#*------ Colonne Date en index ------*#
# Place l'index du dataframe sur la colonne LocalTime
df = df.rename(columns={'Unnamed: 0': "LocalTime"})
df = df.set_index(df['LocalTime'])

# Conversion de la colonne Date au format datetime
df.index = pd.to_datetime(df.index)

# Suppression de la colonne Unnamed:0
df_etude = df.drop("LocalTime", axis=1)
df_etude

#*------ Focus Etat de Floride ------*#
# Mapping des coordonnées aux états
coords_to_state = {
    (30.55, -86.65): 'Florida',
    (35.55, -114.45): 'Arizona',
    (42.65, -78.65): 'New York',
    (45.35, -104.15): 'Montana'
}
# Création de la var State
# Initialisation de la colonne
df['State'] = None  

# Pour chaque paire de coordonnées dans le dictionnaire
for coords, state in coords_to_state.items():
    lat_val, long_val = coords
    # Assignation de l'état à toutes les lignes correspondant à ces coordonnées
    df.loc[(df['lat'] == lat_val) & (df['long'] == long_val), 'State'] = state
# Vérification
print(df['State'].value_counts())
# Filtre Floride
florida_df = df[df['State'] == 'Florida']

#*------ Suppression des vars non-utilisés (pour DA-RNN) ------*#
df_etude = df_etude.drop(['Year', 'Month', 'Day', 'Hour', 'Minute', 'lat', 'long'], axis=1)
df_etude.columns

#*------ Séparation des données test & d'entraînement ------*#
pourcentage = 0.8 # train 80 % / test 20%
temps_separation = int(len(florida_df) * pourcentage)
date_separation = florida_df.index[temps_separation]

# Préparation des séries (Power(MW) uniquement)
serie_entrainement_X = np.array(florida_df['Power(MW)'].values[:temps_separation], dtype=np.float32)
serie_test_X = np.array(florida_df['Power(MW)'].values[temps_separation:], dtype=np.float32)

print(f"Taille de l'entrainement : {len(serie_entrainement_X)}")
print(f"Taille de la validation : {len(serie_test_X)}")

# Normalisation des données
min_max_scaler = preprocessing.MinMaxScaler()
serie_entrainement_X_norm = min_max_scaler.fit_transform(tf.reshape(serie_entrainement_X, shape=(len(serie_entrainement_X), 1)))
# Important: utiliser transform et non fit_transform sur le jeu de test
serie_test_X_norm = min_max_scaler.transform(tf.reshape(serie_test_X, shape=(len(serie_test_X), 1)))

# Visualisation de la séparation train/test
fig, ax = plt.subplots(constrained_layout=True, figsize=(15, 5))
ax.plot(florida_df.index[:temps_separation], serie_entrainement_X_norm, label="Entrainement")
ax.plot(florida_df.index[temps_separation:], serie_test_X_norm, label="Test")
ax.set_title("Séparation des données d'entrainement et de test")
ax.legend()
plt.show()
save_fig(fig, 'train_test_split.png')

#*------------ CREATION DES DATASETS ------------*#
#*------ Fonction de préparation des datasets ------*#
def prepare_dataset_XY(series, longueur_sequence, longueur_sortie, batch_size, shift):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(longueur_sequence+longueur_sortie, shift=shift, drop_remainder=True)
    dataset = dataset.flat_map(lambda x: x.batch(longueur_sequence + longueur_sortie))
    dataset = dataset.map(lambda x: (x[0:longueur_sequence][:, :], 
                                    tf.expand_dims(x[-longueur_sortie:][:, 0], 1)))
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
    return dataset

# Paramètres
batch_size = 128
longueur_sequence = 4*48     # 4 jours (48 observations par jour avec données toutes les 30 min)
longueur_sortie = 48         # 1 jour (prédiction pour le jour suivant)
shift = 1                    # Décalage d'une observation pour créer des séquences chevauchantes

#*------ Création des datasets ------*#
dataset = prepare_dataset_XY(serie_entrainement_X_norm, longueur_sequence, longueur_sortie, batch_size, shift)
dataset_val = prepare_dataset_XY(serie_test_X_norm, longueur_sequence, longueur_sortie, batch_size, shift)
# Vérification des dimensions
for element in dataset.take(1):
    print(f"Forme des entrées X: {element[0].shape}")
    print(f"Forme des sorties y: {element[1].shape}")

#*------ Extraction et préparation des tenseurs d'entraînement ------*#
# Extrait les X,Y du dataset d'entraînement (décompresse tous les lots en une fois)
x, y = tuple(zip(*dataset))              # Transforme le dataset en tuples (x, y)

# Convertit les objets TensorFlow en tableaux numpy avec type float32
x = np.asarray(x, dtype=np.float32)      # Forme: (n_batches, batch_size, longueur_sequence, 1)
y = np.asarray(y, dtype=np.float32)      # Forme: (n_batches, batch_size, longueur_sortie, 1)

# Restructure pour obtenir un tableau 3D où chaque exemple est indépendant
# On "aplatit" les dimensions batch pour avoir toutes les séquences d'une traite
x_train = np.asarray(tf.reshape(x, shape=(x.shape[0]*x.shape[1], longueur_sequence, x.shape[3])))
y_train = np.asarray(tf.reshape(y, shape=(y.shape[0]*y.shape[1], longueur_sortie, y.shape[3])))

# Dimensions finales des données d'entraînement
print(f"Dimensions finales x_train: {x_train.shape}")  # (n_batches*batch_size, longueur_sequence, 1)
print(f"Dimensions finales y_train: {y_train.shape}")  # (n_batches*batch_size, longueur_sortie, 1)

#*------ Extraction et préparation des tenseurs de validation ------*#
# Même procédure pour les données de validation
x_val, y_val = tuple(zip(*dataset_val))

# Conversion en tableaux numpy
x_val = np.asarray(x_val, dtype=np.float32)
y_val = np.asarray(y_val, dtype=np.float32)

# Restructuration des données de validation
x_val = np.asarray(tf.reshape(x_val, shape=(x_val.shape[0]*x_val.shape[1], longueur_sequence, x_val.shape[3])))
y_val = np.asarray(tf.reshape(y_val, shape=(y_val.shape[0]*y_val.shape[1], longueur_sortie, y_val.shape[3])))

print(f"Dimensions finales x_val: {x_val.shape}")
print(f"Dimensions finales y_val: {y_val.shape}")

#*------ Sauvegarde des données préparées ------*#
# Création du dossier de sauvegarde s'il n'existe pas
os.makedirs('data/processed', exist_ok=True)

# Sauvegarde les tableaux numpy pour l'entraînement du modèle
np.save('data/processed/x_train.npy', x_train)
np.save('data/processed/y_train.npy', y_train)
np.save('data/processed/x_val.npy', x_val)
np.save('data/processed/y_val.npy', y_val)
