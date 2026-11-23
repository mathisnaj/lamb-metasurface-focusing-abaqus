# Métasurface A0 : préparation Abaqus 2023

Ce dépôt reprend les fichiers de la V4 **sans son historique Git** et ajoute
une première chaîne Abaqus/Explicit pour une plaque nue et, éventuellement,
une carte de plots issue de la V4. **L'optimisation Abaqus n'est pas encore
implémentée** : sa licence et la compatibilité de l'objectif transitoire
pondéré restent à vérifier. Les fichiers V4 ci-dessous restent disponibles
séparément.

## Premier calcul sur Euler

La génération des fichiers `.inp` utilise le Python externe avec NumPy,
SciPy, Matplotlib, Pillow et PyTorch déjà requis par la V4. Le lecteur ODB
utilise le Python fourni avec Abaqus. Installer ces dépendances dans un
environnement Linux sur Euler, par exemple `.venv-euler` ; ne pas utiliser un
environnement copié depuis Windows. Charger le même module Python dans le
script Slurm si l'environnement en dépend.

1. Modifier si nécessaire `config.py` ; les valeurs de V4 sont conservées.
   `ABAQUS_HISTORY_DT` est l'intervalle de sortie des signaux sur la ligne,
   distinct du pas de temps d'intégration choisi par Abaqus.
2. Vérifier les directives et les modules dans `lancer_abaqus.sh`, puis :

   ```bash
   sbatch lancer_abaqus.sh
   squeue -u "$USER"
   ```

   Par défaut, seul le cas **plaque nue** est lancé. Le script demande
   quatre CPU sur un nœud, 4 Go par CPU, quatre heures et utilise
   `mp_mode=threads`. La disponibilité des licences et cette configuration
   doivent être vérifiées sur Euler. Pour un autre Python, soumettre par
   exemple avec `sbatch --export=ALL,PYTHON=/chemin/vers/python lancer_abaqus.sh`.
3. Pour comparer une carte produite par la V4 avec la plaque nue :

   ```bash
   sbatch --export=ALL,DESIGN_PATH=/chemin/absolu/vers/design.npz lancer_abaqus.sh
   ```

   La signature du fichier V4 doit correspondre aux paramètres physiques,
   géométriques et temporels actuels de `config.py`. Une plaque nue retenue
   par la V4 produira logiquement deux cas identiques.
4. Suivre `tail -f abaqus-NUMERO.out` après le démarrage du job. Les sorties
   sont isolées dans `abaqus_runs/NUMERO/` : `.inp`, `.odb`, `.dat`, `.msg`,
   `.sta`, `manifest.json`, historiques JSON et, si une carte est fournie,
   `comparaison.json`. Vérifier l'état final du job et les journaux Abaqus ;
   un message d'échec n'est pas un résultat utilisable.

Le calcul utilise des hexaèdres `C3D8R`, un chargement ponctuel vertical
fenêtré, des bords libres, et des plots pleins symétriques sur les deux faces.
La V4 utilise sa propre intégration élastique et le schéma temporel Abaqus
choisit son propre pas stable : **ce n'est pas une reproduction numérique
identique**. Les deux résultats doivent être comparés, puis faire l'objet
d'études de convergence du maillage, de l'intervalle de sortie, du pas et
de l'énergie de hourglass. La comparaison Abaqus intègre le carré du signal
linéaire par morceaux dans la fenêtre ; elle n'est pas une énergie mécanique.

Cette chaîne n'a pas encore été exécutée dans Abaqus/Euler. Commencer par une
plaque nue et vérifier les logs, les unités SI, la propagation et les traces
avant de lancer une recherche. Les ODB peuvent occuper beaucoup d'espace.

## Programme V4 hérité

La V4 optimise la réponse d'un **FEM élastique 3D transitoire**, et non plus le
FEM scalaire de la V3. Le réseau PyTorch produit les hauteurs et deux périodes
nominales. La propagation et l'adjoint discret sont calculés sur CPU ; le
réseau peut utiliser CPU, CUDA ou MPS. Ce n'est pas un PINN classique : le
réseau décrit la géométrie, pas le champ.

## Utilisation

Installer NumPy, SciPy, Matplotlib, Pillow et PyTorch pour le Python utilisé.
Tous les fichiers sont dans ce dossier. Modifier `config.py`, puis lancer :

```bash
python3 main.py
python3 verification_3d.py
```

Les paramètres physiques, la source, la ligne focale et le maillage de V3
sont conservés. Les résultats sont écrits dans `OUTPUT_DIRECTORY` et écrasés.
La vérification refuse une carte dont les paramètres physiques ou temporels
ne correspondent pas à la configuration courante.

**Coût :** chaque mise à jour nécessite une propagation 3D, un adjoint et des
recalculs ; chaque contrôle discret exige une simulation 3D supplémentaire.
La configuration fournie peut prendre beaucoup de temps et de mémoire,
même avec un GPU, car les solveurs restent sur CPU. Commencer éventuellement
avec un maillage réduit et peu d'étapes, puis remettre les valeurs souhaitées
et relancer. Modifier les indices de `LENS_X/Y` si le maillage change.
À titre indicatif, sur l'environnement CPU de développement, une mise à jour
et deux évaluations entières sur le maillage fourni ont pris environ
2 min 30 s ; ce temps dépend fortement de la machine.

## Objectif

À \(x_f\), on mesure \(u_z\) sur toute la largeur de la face supérieure, en
pondérant les nœuds par une gaussienne centrée en `WIDTH/2`, de largeur
`FOCAL_SIGMA = 0.050` m, avec demi-poids trapézoïdaux aux bords.
La somme des poids vaut un. Avec la fenêtre `FOCUS_WINDOW`, l'objectif est :

\[
G(h)=
\frac{\sum_jw_j\sum_{t_n\in W}u_z(x_f,y_j,t_n;h)^2\Delta t}
{\sum_jw_j\sum_{t_n\in W}u_z(x_f,y_j,t_n;0)^2\Delta t},
\quad J(h)=G(h)-\lambda\operatorname{moyenne}(h)/H_{\max}.
\]

C'est une intégrale de déplacement au carré, **pas une énergie mécanique
totale**. La gaussienne privilégie le centre ; elle n'impose ni uniformité le
long de la ligne ni largeur de focalisation selon x. La source représente une
onde retour supposée, pas un défaut simulé ; `SOURCE_MODE` favorise une parité
sans garantir un mode pur. Les bords sont libres et les réflexions comptent.

## Hauteurs continues et carte réalisable

Les plots sont des colonnes pleines, identiques sur les deux faces. Le
réseau produit une hauteur continue par élément. Pour la différentiation,
chaque couche de plot est remplie partiellement selon
`clip(hauteur - indice_couche, 0, 1)`. Masse et rigidité sont interpolées
ensemble, avec une fraction résiduelle `V4_VOID_FRACTION` dans les zones
vides pour conserver un maillage fixe. Ce matériau résiduel et les couches
partiellement remplies sont une **relaxation numérique**, pas une géométrie
fabriquable. Ils peuvent introduire une erreur ou des modes parasites.

Tous les `V4_DISCRETE_EVERY` pas, la carte est arrondie en couches entières,
puis recalculée par le FEM 3D à géométrie réelle, sans matériau résiduel.
La meilleure carte entière selon l'objectif pénalisé est retenue ; si aucune
ne dépasse la plaque nue, c'est celle-ci qui est conservée. L'adjoint agit
sur la relaxation ; il ne garantit pas une amélioration après arrondi ni un
optimum global. Comparer les gains relaxé et entier dans `convergence.png`.

La mise à jour du réseau utilise l'adjoint du schéma explicite de différences
centrales. Des checkpoints évitent de conserver tous les états, au prix de
recalculs. Un pas de temps conservateur commun est utilisé pour les deux
plaques et la vérification, permettant de comparer les mêmes mesures.

## Réglages et fichiers

* `V4_STEPS`, `V4_LEARNING_RATE`, `V4_DISCRETE_EVERY` et
  `V4_HEIGHT_PENALTY` règlent la recherche. `V4_CHECKPOINT_INTERVAL` règle le
  compromis mémoire/recalcul ; `V4_VOID_FRACTION` règle le matériau résiduel.
* Les paramètres `V2_PERIOD_*`, `V2_INITIAL_PERIOD_*`, `V2_FILL_FRACTION`,
  `V2_GATE_SHARPNESS` et `V2_INITIAL_FILL` définissent encore le réseau et
  ses enveloppes. Les anciens `V2_STEPS`, `V2_LEARNING_RATE`,
  `V2_DISCRETE_EVERY` et `V2_HEIGHT_PENALTY` ne pilotent plus l'optimisation.
  Les périodes apprises sont nominales, sans garantie d'espacement exact ni
  de fabricabilité.
* `design.npz`, `optimisation.json`, `convergence.png`, `metasurface_3d.png`
  et éventuellement `reseau.pt` décrivent le résultat retenu et la recherche.
  Le JSON rapporte le gain de la géométrie **entière**, non le gain relaxé.
* `verification_3d.npz`, `verification_3d.json`, `ligne_focale_3d.png`,
  `geometrie.png`, `capteurs.png` et `ondes_3d.gif` donnent la vérification,
  les profils, les signaux et l'animation. `methode_lentille.tex` détaille
  les équations et peut être importé dans Overleaf.

La validation utilise le même solveur, la même source et la même fenêtre que
l'optimisation : ce n'est pas une validation physique indépendante. Le maillage
actuel représente environ quatre éléments par longueur d'onde A0 théorique
selon x ; une étude de convergence spatiale et temporelle, une comparaison
avec un autre solveur et une validation expérimentale restent nécessaires.
