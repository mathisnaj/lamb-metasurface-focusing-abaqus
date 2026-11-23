"""Tous les paramètres sont en unités SI, sauf indication contraire."""

from pathlib import Path

# Plaque et maillage commun aux deux modèles.
LENGTH = 0.500
WIDTH = 0.400
THICKNESS = 0.0015
NX = 120
NY = 100
PLATE_LAYERS = 2
YOUNG = 70e9
POISSON = 0.33
DENSITY = 2700.0
STUB_YOUNG = 70e9
STUB_POISSON = 0.33
STUB_DENSITY = 2700.0
STUB_DZ = 0.0005
MAX_HEIGHT = 4  # hauteur entière en couches : chaque plot touche la plaque

# Zone optimisable en indices d'éléments [début, fin[. Position fixe.
LENS_X = (40, 60)
LENS_Y = (10, 90)
SYMMETRY_X = False  # miroir x -> -x autour du centre de la zone
SYMMETRY_Y = True   # miroir y -> -y autour du centre de la zone
INITIAL_HEIGHT = 2
# Anciens paramètres conservés pour référence ; sans effet dans la méthode actuelle,
# où la répétition est imposée exactement par CELL_PERIOD_X.
REPETITION_PERIOD_X = 1
REPETITION_WEIGHT = 1.0
# Nouvelle méthode : cellule de coupe x-z répétée exactement sur les deux faces.
CELL_PERIOD_X = 10             # nombre d'éléments x par cellule, doit diviser LENS_X #10
BAND_WIDTH_Y = 2              # largeur d'un élément de métasurface, en cellules y.  #3
BAND_GAP_Y = 2                # espacement sans plots entre éléments, en cellules y #1
MIN_TRANSMISSION_A = 0.85     # seuil du rapport d'amplitude à atteindre
PHASE_TOLERANCE = 0.5235987755982988  # pi/6, radians : normalisation de l'erreur
STRIP_PADDING_X = 24          # éléments de plaque saine de chaque côté de la lentille
STRIP_ABSORB_X = 12           # épaisseur de la couche amortie aux extrémités
STRIP_DAMPING = 3000000.0     # amortissement massique maximal, en s^-1
ITERATIONS = 800              # conservé de l'ancien optimiseur, non utilisé par PyTorch
CANDIDATES_PER_ITERATION = 8  # conservé de l'ancien optimiseur, non utilisé par PyTorch
MUTATION_PIXELS = 4
SEED = 42

# Apprentissage actif du modèle de substitution PyTorch.
NN_DEVICE = "auto"            # "auto", "cpu", "cuda" ou "mps"
NN_INITIAL_SAMPLES = 64       # cellules FEM initiales (y compris les cellules uniformes)
NN_ROUNDS = 12                # cycles entraînement / sélection / calcul FEM
NN_EVALUATIONS_PER_ROUND = 16 # nouvelles cellules FEM au maximum par cycle
NN_RANDOM_PER_ROUND = 4       # part exploratoire de ces évaluations
NN_POOL_SIZE = 512            # candidats prédits par le réseau à chaque cycle
NN_EPOCHS = 100               # époques d'entraînement par cycle et au contrôle final
NN_HIDDEN = 64                # largeur des deux couches cachées
NN_LEARNING_RATE = 0.003
NN_VALIDATION_FRACTION = 0.2

# Réseau de géométrie hérité de V2 ; seules les périodes et enveloppes ci-dessous
# restent utilisées en V4. Les réglages d'optimisation V2 sont conservés sans effet.
# Les anciens paramètres NN_* restent inutilisés sauf NN_DEVICE et NN_HIDDEN.
V2_STEPS = 120
V2_LEARNING_RATE = 0.01
V2_PERIOD_X_RANGE = (3.0, 20.0)  # périodes nominales en éléments du maillage
V2_PERIOD_Y_RANGE = (3.0, 20.0)
V2_INITIAL_PERIOD_X = 10.0
V2_INITIAL_PERIOD_Y = 4.0
V2_GATE_SHARPNESS = 5.0
V2_FILL_FRACTION = 0.6        # largeur relative initiale des motifs dans chaque direction
V2_INITIAL_FILL = 0.5         # fraction initiale approximative de hauteur maximale
V2_HEIGHT_PENALTY = 0.02      # coût moyen de matière, sans dimension
V2_DISCRETE_EVERY = 5         # évaluation FEM après arrondi des hauteurs

# V4 : adjoint élastique transitoire. Le coût augmente fortement avec le maillage.
V4_STEPS = 20
V4_LEARNING_RATE = 0.003
V4_DISCRETE_EVERY = 5
V4_HEIGHT_PENALTY = 0.02
V4_VOID_FRACTION = 1e-4       # rigidité et masse résiduelles des éléments vides
V4_CHECKPOINT_INTERVAL = 64   # compromis mémoire / recalcul de l'adjoint
ABAQUS_HISTORY_DT = 0.25e-6   # intervalle des traces sur la ligne (secondes)

# La source à droite est une source équivalente à une onde revenant du défaut.
# Le foyer est dans la zone de l'émetteur réel, à gauche de la lentille.
FREQUENCY = 50_000.0
RETURN_SOURCE = (0.35, 0.2)
FOCUS = (0.1, 0.2)
# V3 : ligne focale sur toute la largeur à x = FOCUS[0].
# Gaussienne centrée en WIDTH/2 ; FOCUS[1] reste un point de mesure.
FOCAL_SIGMA = 0.050           # écart-type de la gaussienne, en mètres
SENSORS = [(0.1, 0.1), (0.1, 0.2), (0.1, 0.3)]

# Vérification 3D : "top" sur la face supérieure, "A" efforts verticaux
# de même signe sur les deux faces, "S" efforts de signes opposés.
# Ces parités favorisent un mode, sans garantir une excitation modale pure.
SOURCE_MODE = "A"
SOURCE_FORCE = 1.0
CYCLES = 5
DURATION = 300e-6
FOCUS_WINDOW = (0.0, DURATION)  # fenêtre d'énergie pour comparer au foyer
FRAMES = 50
FPS = 15
SHOW = False
GEOMETRY_Z_SCALE = 10.0       # amplification visuelle de la hauteur, sans effet sur la physique

# Créer au besoin, dans un emplacement où l'utilisateur peut écrire.
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "resultats"
