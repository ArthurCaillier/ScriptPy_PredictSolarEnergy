#!preambule! Chargement packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates


"""
Analyse Exploratoire des Données (EDA)
-------------------------------------
Ce script réalise une analyse exploratoire des données météorologiques et de production 
d'énergie solaire collectées dans quatre États américains (Arizona, Floride, New York, Montana).
Les graphiques générés sont sauvegardés dans le dossier output/eda/.
Parmi les variables : 
1 - Mesures du rayonnement solaire
- DNI (Direct Normal Irradiance) : Rayonnement solaire direct normal - mesure du rayonnement solaire reçu perpendiculairement aux rayons du soleil. Unité: W/m². Particulièrement important pour les systèmes solaires à concentration.
- DHI (Diffuse Horizontal Irradiance) : Rayonnement solaire diffus horizontal - rayonnement solaire diffusé par l'atmosphère et les nuages. Unité: W/m². Représente la lumière qui arrive de toutes les directions du ciel.
- GHI (Global Horizontal Irradiance) : Rayonnement global horizontal - somme du rayonnement direct et diffus sur une surface horizontale. Unité: W/m². C'est la mesure totale du rayonnement solaire et généralement la plus utilisée pour prédire la production des panneaux solaires photovoltaïques.
- Clearsky GHI/DHI/DNI : Valeurs estimées en condition de ciel clair (sans nuages) pour chaque type de rayonnement.
2 - Autres variables météorologiques
- Solar Zenith Angle : Angle zénithal solaire - angle entre la verticale (zénith) et la direction du soleil. En degrés.
- Cloud Type : Type de nuage (codé numériquement).
- AOD (Aerosol Optical Depth) : Épaisseur optique des aérosols - mesure de la quantité d'aérosols dans l'atmosphère.
- Alpha : Exposant d'Ångström - paramètre lié à la distribution des tailles des particules d'aérosol.
- Asymmetry : Paramètre d'asymétrie des aérosols - caractérise la direction préférentielle de diffusion de la lumière.
- SSA (Single Scattering Albedo) : Albédo de diffusion simple - rapport entre la diffusion et l'extinction totale par les aérosols.
- Surface Albedo : Fraction du rayonnement solaire réfléchi par la surface terrestre.
- Precipitable Water : Eau précipitable - quantité totale de vapeur d'eau dans une colonne d'atmosphère. Unité: cm.
- Fill Flag et Cloud Fill Flag : Indicateurs de données manquantes ou interpolées (0 généralement signifie que les données sont mesurées directement).
Wind Speed : Vitesse du vent, probablement en m/s
Wind Direction : Direction du vent en degrés (0-360°), où 0° ou 360° = Nord, 90° = Est, 180° = Sud, 270° = Ouest
Autres variables manquantes
Ozone : Concentration d'ozone dans l'atmosphère, généralement en unité Dobson
Pressure : Pression atmosphérique, probablement en hPa (hectopascals)
Temperature et Dew Point : Température et point de rosée, probablement en °C
Relative Humidity : Humidité relative en pourcentage (0-100%)
"""

# Chargement du dataset
df = pd.read_csv('data/usaWithWeather.csv')

# Configuration graphique 
# Création du dossier de sortie 
OUTPUT_DIR = os.path.join('output', 'eda')
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Fonction de sauvegarde
def save_fig(fig, filename):
    """Sauvegarde une figure dans le dossier output/eda/"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Figure sauvegardée: {filepath}")

#*------ Informations générales ------*#
print(f"Affichage des premières lignes:{df.head()}")
print(f"Informations sur le dataset:{df.info()}") # Vars numeric de type int/float64
print(f"Statistiques descriptives:{df.describe()}") 
# Conversion en int/float32 pour gagner en temps de calcul
float_cols = df.select_dtypes(include=['float64']).columns
df[float_cols] = df[float_cols].astype('float32')

int_cols = df.select_dtypes(include=['int64']).columns
df[int_cols] = df[int_cols].astype('int32')  

print(f"Dimensions du dataset: {df.shape}") # 70080 obs & 33 vars

print("\nVérification des valeurs manquantes & duppliquées :")
print(df.isnull().sum()) 
print(df.duplicated().sum())
#!Obs! Aucun NAs ou valeurs duppliquées
print(df.columns)

#*------ Analyse Power - Modèle Univarié - ------*#
# Affichage de la série avec la librairie plotly
fig = go.Figure()
# Ouverture du graphique sur navigateur
pio.renderers.default = "browser"

#!Preambule! Conversion de la colonne LocalTime en datetime 
df['LocalTime'] = pd.to_datetime(df['LocalTime'])
#!Remarque! Cycle journalier de 48 observations (relevé tous les 30mn)

fig.add_trace(go.Scatter(x=df['LocalTime'], y=df['Power(MW)'], 
                         line=dict(color='blue', width=1),
                         name="Puissance (MW)"))

fig.update_xaxes(rangeslider_visible=True,
                 title="Date et heure locale")
                 
fig.update_yaxes(title="Puissance (MW)",
                autorange=True,
                fixedrange=False)

fig.update_layout(title="Production d'énergie solaire au fil du temps",
                  template="plotly_white")

# Ajout d'une barre de sélection temporelle
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1j", step="day", stepmode="backward"),
                dict(count=7, label="1sem", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all", label="Tout")
            ])
        )
    )
)
fig.show()
#!Obs! 4 tendances qui correspondent très probablement aux différents Etats (Floride, Arizona, New York et Montana)

#*------ Analyse par État ------*#
# Identification des états basée sur les coordonnées géographiques
# On va assigner un état à chaque point de données en fonction de lat/long

# Coordonnées uniques 
unique_locations = df[['lat', 'long']].drop_duplicates().sort_values('lat')
print(f"Coordonnées uniques par État:{unique_locations}")

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

# Analyse comparative de la production par état
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.flatten()

# 1. Boxplot de la production par état
sns.boxplot(x='State', y='Power(MW)', data=df[df['Power(MW)'] > 0], ax=axes[0])
axes[0].set_title('Distribution de Power(MW) par État (valeurs > 0)')
axes[0].set_xlabel('État')
axes[0].set_ylabel('Power(MW)')
axes[0].tick_params(axis='x', rotation=45)

# 2. Production moyenne par heure et par état
hourly_by_state = df.groupby(['State', 'Hour'])['Power(MW)'].mean().reset_index()
for state in hourly_by_state['State'].unique():
    subset = hourly_by_state[hourly_by_state['State'] == state]
    axes[1].plot(subset['Hour'], subset['Power(MW)'], marker='o', label=state)
axes[1].set_title('Production moyenne par heure selon l\'État')
axes[1].set_xlabel('Heure de la journée')
axes[1].set_ylabel('Power(MW) moyen')
axes[1].set_xticks(range(0, 24, 2))
axes[1].legend()
axes[1].grid(True)

# 3. Production moyenne mensuelle par état
monthly_by_state = df.groupby(['State', 'Month'])['Power(MW)'].mean().reset_index()
for state in monthly_by_state['State'].unique():
    subset = monthly_by_state[monthly_by_state['State'] == state]
    axes[2].plot(subset['Month'], subset['Power(MW)'], marker='o', label=state)
axes[2].set_title('Production moyenne par mois selon l\'État')
axes[2].set_xlabel('Mois')
axes[2].set_ylabel('Power(MW) moyen')
axes[2].set_xticks(range(1, 13))
axes[2].set_xticklabels(['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sept', 'Oct', 'Nov', 'Déc'])
axes[2].legend()
axes[2].grid(True)

# 4. Heatmap des corrélations par état
df_numeric = df.select_dtypes(include=['float32', 'int32'])
corr_by_state = {}
for state in df['State'].unique():
    # if state != 'Unknown':
    state_df = df[df['State'] == state]
    corr_by_state[state] = state_df[df_numeric.columns].corr()['Power(MW)'].abs().sort_values(ascending=False)

# Comparaison des principales correlations entre états
corr_comparison = pd.DataFrame(corr_by_state)

# Affichage des 10 principales variables (en excluant Power(MW))
top_vars = set()
for state in corr_by_state:
    top_vars.update(corr_by_state[state].index[1:6])  # Top 5 par état, excluant Power(MW)
top_vars = list(top_vars)[:10]  # Limiter à 10 variables

sns.heatmap(corr_comparison.loc[top_vars], annot=True, cmap='coolwarm', ax=axes[3])
axes[3].set_title('Corrélations avec Power(MW) par État')
axes[3].tick_params(axis='x', rotation=45)
axes[3].tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.show()
save_fig(fig, 'power_comparison_by_state.png')

#!Obs! Il existe des différences importantes dans la production solaire entre les états, 
# ~ justifiant des modèles spécifiques pour chaque région afin d'obtenir des prédictions optimales.

#*------ Focus État de Floride ------*#
# Filtrer les données pour n'afficher que la Floride
florida_df = df[df['State'] == 'Florida']

# Affichage de la série avec la librairie plotly
fig = go.Figure()
# Ouverture du graphique sur navigateur
pio.renderers.default = "browser"

fig.add_trace(go.Scatter(x=florida_df['LocalTime'], y=florida_df['Power(MW)'], 
                         line=dict(color='blue', width=1),
                         name="Puissance (MW)"))

fig.update_xaxes(rangeslider_visible=True,
                 title="Date et heure locale")
                 
fig.update_yaxes(title="Puissance (MW)",
                autorange=True,
                fixedrange=False)

fig.update_layout(title="Production d'énergie solaire au fil du temps",
                  template="plotly_white")

# Ajout d'une barre de sélection temporelle
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1j", step="day", stepmode="backward"),
                dict(count=7, label="1sem", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all", label="Tout")
            ])
        )
    )
)
fig.show()

#!Obs! Tendances journalieres
# ~ Motif en cloche chaque jour, correspondant au cycle solaire
# ~ La production commence & s'arrete a des heures variables selon les saisons
# Tendances saisonnières
# ~ Sur l'image annuelle, légère variation saisonnière dans l'amplitude maximale des pics
# ~ Les pics de production semblent légèrement plus élevés pendant les mois d'été (mai à septembre)
# ~ La durée quotidienne de production est également plus longue en été qu'en hiver

serie = florida_df['Power(MW)'].values

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
f1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
f1.subplots_adjust(hspace=0.3,wspace=0.2)

plot_acf(serie, ax=ax1, lags = range(0,500))
ax1.set_title("Autocorrélation")
plot_pacf(serie, ax=ax2, lags = range(0, 500))
ax2.set_title("Autocorrélation partielle")
plt.show()

#!Obs! ACF (Autocorrélation)
# ~ Forte saisonnalité : Le motif répétitif très marqué indique une saisonnalité journalière très forte, ce qui est logique pour l'énergie solaire
# ~ Décroissance lente : L'autocorrélation reste significative même à des décalages élevés, ce qui indique une forte persistance (mémoire longue) dans la série
# ~ Structure sinusoïdale caractéristique d'un cycle saisonnier régulier

# PACF (Autocorrélation partielle)
# ~ Forte dépendance au lag 1 : Le pic très élevé au lag 1 indique qu'une grande partie de la dépendance peut être expliquée par la valeur précédente
# ~ Pics significatifs aux multiples de 48 : Des pics plus petits mais significatifs aux décalages correspondant à des périodes journalières
# ~ Cette périodicité est typique des données d'énergie solaire, où la production à un moment donné est fortement corrélée avec celle du même moment les jours suivants
# ~ Décroissance rapide : Une fois ces effets pris en compte, les autres lags deviennent majoritairement non significatifs

# Implications pour votre modèle LSTM :
# Forte structure temporelle : Votre série présente une structure temporelle très forte qui se prête bien à la modélisation par réseaux de neurones récurrents comme LSTM
# Fenêtre d'entrée optimale : Vos graphiques suggèrent qu'une fenêtre d'entrée d'au moins 24-48 heures serait idéale pour capturer les motifs journaliers

