
import pandas as pd  # Manipulation des tableaux de données
import numpy as np   # Génération de valeurs numériques aléatoires


# ==================================================
# 1. Initialisation
# ==================================================

# Fixer la graine aléatoire pour garantir la reproductibilité
np.random.seed(42)

# Nombre de clients fictifs à générer
n = 500


# ==================================================
# 2. Génération des données (FEATURES)
# ==================================================

# Construire le DataFrame colonne par colonne
data = pd.DataFrame({
    "age": np.random.randint(18, 70, n),
    # Âge entre 18 et 69 ans

    "salaire": np.random.randint(1200, 8000, n),
    # Salaire mensuel simulé

    "abonnement": np.random.choice(
        ["Basique", "Standard", "Premium"], n
    ),
    # Type d'offre

    "anciennete": np.random.randint(0, 15, n),
    # Ancienneté en années, de 0 à 14

    "consommation": np.random.randint(0, 100, n),
    # Score d'utilisation du service, de 0 à 99
})


# ==================================================
# 3. Création de la TARGET : churn
# ==================================================

# Un client est considéré comme à risque si :
# - sa consommation est inférieure à 30
# - ET son ancienneté est inférieure à 2 ans

data["churn"] = (
    (data["consommation"] < 30) &
    (data["anciennete"] < 2)
).astype(int)

# True devient 1 et False devient 0


# ==================================================
# 4. Sauvegarde des données
# ==================================================

# Exporter le DataFrame dans un fichier CSV
# index=False évite d'ajouter une colonne d'index au fichier
data.to_csv("churn_data.csv", index=False)


# ==================================================
# 5. Vérification des résultats
# ==================================================

# Afficher les 5 premières lignes
print(data.head())

# Calculer et afficher le taux de churn
print(f"Taux de churn : {data['churn'].mean():.1%}")
