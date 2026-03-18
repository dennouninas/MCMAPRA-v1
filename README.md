# MCMAPRA — Modèle de Conciliation Multi-Agent

**Un modèle multi-agent de conciliation enrichi par ontologie et IoT de localisation pour atténuer le problème du cold start dans les systèmes de recommandation de POI dans les LBSN**

[![IEEE Access](https://img.shields.io/badge/IEEE_Access-2025-blue)](https://ieeeaccess.ieee.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green)](https://python.org)
[![Licence](https://img.shields.io/badge/Licence-MIT-yellow)](LICENSE)

---

## 📌 Résumé

MCMAPRA propose une architecture à 5 couches pour la recommandation de Points d'Intérêt (POI) dans les réseaux sociaux basés sur la localisation (LBSN), en ciblant spécifiquement le problème du **cold start** (δ < 0,1 % de densité matricielle).

**Résultats clés sur Foursquare NYC :**
| Modèle | Précision@10 | Rappel@10 | F1@10 | MSE |
|--------|:---:|:---:|:---:|:---:|
| FC-Pearson (baseline) | 0,312 | 0,178 | 0,227 | 1,42 |
| OWLREC [2024] | 0,371 | 0,218 | 0,275 | 1,16 |
| **MCMAPRA-Borda (ours) ★** | **0,411** | **0,260** | **0,322** | **0,94** |

En cold start strict (0 interaction) : **précision × 3,9** grâce à l'enrichissement ontologique.

---

## 🏗️ Architecture

```
MCMAPRA (5 couches)
│
├── ① DONNÉES         — Matrice R, Graphe T⁰, Flux IoT, Ontologie OWL 2
├── ② PRÉ-TRAITEMENT  — Normalisation (Éq.22), SPARQL 1.1, Kalman
├── ③ AGENTS          — Rec1Ag (Pearson), Rec2Ag (Jaccard), Rec3Ag (Confiance)
│     ├── Éq.(3-4b)  sim_P + lissage β=25
│     ├── Éq.(5-6b)  sim_J + Jaccard TF-IDF
│     └── Éq.(7-8c)  T^(k) marche aléatoire, convergence garantie
├── ④ CONCILIATION    — BordaAg (Éq.9), CondorcetAg (Éq.10-11)
└── ⑤ FUSION          — Score_final = λ·Score_vote + (1−λ)·Score_geo (Éq.12)
                       Score_geo   = exp(−d²/2σ²) × Pop_IoT × w_cat (Éq.13)
                       Score_onto  = 0,35·S_cat + 0,25·S_pop + 0,40·S_trust (Éq.18)
```

---

## 📁 Structure du projet

```
mcmapra/
│
├── mcmapra/
│   ├── mcmapra.py              # Algorithme 1 : Orchestration principale
│   ├── agents/
│   │   ├── rec1_pearson.py     # Éq. (3)-(4b) : Similarité de Pearson
│   │   ├── rec2_jaccard.py     # Éq. (5)-(6b) : Jaccard TF-IDF
│   │   └── rec3_confiance.py   # Éq. (7)-(8c) : Propagation de confiance
│   ├── conciliation/
│   │   ├── borda.py            # Algorithme 2A : Vote de Borda
│   │   └── condorcet.py        # Algorithme 2B : Vote de Condorcet + Copeland
│   ├── iot/
│   │   └── module_iot.py       # Algorithmes 3A-3C : Module IoT
│   ├── ontologie/
│   │   └── infereur.py         # Algorithmes 4A-4C : Ontologie OWL 2
│   └── evaluation/
│       └── metriques.py        # Éq. (14)-(17b) : MSE, Préc@k, Rappel@k, F1, ILD
│
├── main.py                     # Script de démonstration et d'évaluation
├── requirements.txt
├── tests/
│   └── test_mcmapra.py
├── notebooks/
│   └── demo_mcmapra.ipynb
└── data/
    └── README_data.md
```

---

## 🚀 Installation rapide

```bash
# Cloner le dépôt
git clone https://github.com/[votre_compte]/mcmapra.git
cd mcmapra

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate    # Linux/macOS
# venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

---

## ▶️ Utilisation

### Démonstration rapide
```bash
python main.py --k 10 --agent borda --protocole warm
```

### Cold start strict (0 interaction)
```bash
python main.py --k 10 --agent borda --protocole CS-0
```

### Étude d'ablation complète
```bash
python main.py --ablation
```

### Utilisation dans votre code Python
```python
import numpy as np
from mcmapra import MCMAPRA

# Charger vos données
R = np.load("data/matrice_interactions.npy")   # (n_users, n_poi)
T = np.load("data/graphe_confiance.npy", allow_pickle=True).item()

# Initialiser MCMAPRA
modele = MCMAPRA(
    matrice_R=R,
    graphe_confiance=T,
    agent_conciliation="borda",   # "borda" (optimal) ou "condorcet"
    lambda_fusion=0.30,           # Éq. (12) — valeur optimale
)

# Recommander Top-10 pour l'utilisateur 42
reco = modele.recommander(u_id=42, k=10, statut_cs="warm")
print(reco)   # [(poi_id, score), ...]

# Évaluer
metriques = modele.evaluer(R_test, k=10, protocole_cs="warm")
print(metriques)
```

---

## 📊 Paramètres Clés

### Paramètres IoT (Module 3)
| Paramètre | Symbole | Valeur Optimale | Description |
|-----------|---------|:---:|-------------|
| Rayon d'influence | σ | 500 m | Décroissance gaussienne Éq.(13) |
| Poids IoT/CF | λ | 0,30 | Fusion Score_vote vs Score_geo Éq.(12) |
| Décroissance temporelle | τ | 0,85 | Vieillissement Pop_IoT Éq.(21b) |
| Poids temps réel | w_real | 0,60 | Balance live/historique Éq.(21) |

### Paramètres Ontologiques (Module 4)
| Paramètre | Symbole | Valeur Optimale | Description |
|-----------|---------|:---:|-------------|
| Poids catégoriel | w_cat | 0,35 | Affinité FastText d=128 Éq.(18) |
| Poids popularité | w_pop | 0,25 | Popularité locale/globale Éq.(19) |
| Poids confiance | w_trust | 0,40 | Réseau social Éq.(20) |
| Facteur propagation | α | 0,70 | Marche aléatoire Éq.(7) |
| Profondeur | k | 3 | trust_depth optimal |

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 📂 Données

Les jeux de données utilisés dans l'article sont disponibles publiquement :

- **Foursquare NYC** : https://sites.google.com/site/yangdingqi/home/foursquare-dataset
- **Gowalla (USA)** : https://snap.stanford.edu/data/loc-gowalla.html

Placer les fichiers dans le dossier `data/` et adapter les chemins dans `main.py`.

---

## 📖 Citation

Si vous utilisez ce code, merci de citer l'article :

```bibtex
@article{auteur2025mcmapra,
  title   = {Un modèle multi-agent de conciliation enrichi par ontologie et IoT de localisation
             pour atténuer le problème du cold start dans les systèmes de recommandation de POI},
  author  = {Auteur Un and Auteur Deux and Auteur Trois},
  journal = {IEEE Access},
  year    = {2025},
  doi     = {10.1109/ACCESS.2025.XXXXXXX}
}
```

---

## 📋 Algorithmes Formels

| N° | Algorithme | Équations | Complexité |
|----|-----------|-----------|-----------|
| 1 | MCMAPRA (orchestration) | — | O(\|N(u)\|·\|C\|+\|C\|²·\|A\|) |
| 2A | BordaAg | Éq. (9-9b) | O(\|A\|·\|C\|+\|C\|log\|C\|) |
| 2B | CondorcetAg | Éq. (10-11) | O(\|A\|·\|C\|²) |
| 3A | CollecterContexteIoT | Éq. (21-21b) | O(\|POI\|) < 10 ms |
| 3B | ModuleIoT.Calculer | Éq. (13-13c) | O(\|C\|) |
| 3C | FusionFinale | Éq. (12) | O(\|C\|) |
| 4A | InfererOntologie | Éq. (7-8c) | O(\|V(u)\|·d) |
| 4B | Ontologie.Calculer | Éq. (18-20b) | O(\|C\|·d), d=128 |
| 4C | OntologieExpansion | — | O(N·log N), N=5k garanti |

---

## 📄 Licence

MIT © 2025 — Auteur Un, Auteur Deux, Auteur Trois
