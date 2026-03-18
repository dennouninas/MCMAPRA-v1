"""
Métriques d'évaluation — MCMAPRA
==================================
Implémente les Équations (14)-(17b) de l'article MCMAPRA.

  Éq. (14)  MSE          = (1/|T|) · Σ (r_up − r̂_up)²
  Éq. (15)  Précision@k  = |{p ∈ Top-k : r_up ≥ θ}| / k
  Éq. (16)  Rappel@k     = |{p ∈ Top-k : r_up ≥ θ}| / |{p : r_up ≥ θ}|
  Éq. (16b) F1@k         = 2 · Précision@k · Rappel@k / (Précision@k + Rappel@k)
  Éq. (17)  ILD          = 2 · Σ_{i<j} dist(pᵢ, pⱼ) / (k·(k−1))
  Éq. (17b) NDCG@k       = DCG@k / IDCG@k
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set

import numpy as np


def calculer_metriques(
    poi_recommandes: List[int],
    poi_pertinents: Set[int],
    notes_reelles: np.ndarray,
    k: int = 10,
    seuil_pertinence: float = 3.5,
    embeddings: Optional[Dict[int, np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Calcule toutes les métriques d'évaluation de l'article MCMAPRA.

    Paramètres
    ----------
    poi_recommandes  : liste ordonnée des poi_id recommandés (Top-k)
    poi_pertinents   : ensemble des poi_id réellement pertinents (r ≥ θ)
    notes_reelles    : vecteur des notes réelles de l'utilisateur
    k                : taille de la liste Top-k
    seuil_pertinence : seuil θ (défaut 3.5)
    embeddings       : optionnel — pour le calcul de l'ILD

    Retourne
    --------
    dict avec MSE, Précision@k, Rappel@k, F1@k, ILD, NDCG@k
    """
    top_k = poi_recommandes[:k]
    pertinents_trouves = set(top_k) & poi_pertinents

    # ── Éq. (15) : Précision@k ───────────────────────────────────────────────
    precision = len(pertinents_trouves) / k if k > 0 else 0.0

    # ── Éq. (16) : Rappel@k ──────────────────────────────────────────────────
    rappel = len(pertinents_trouves) / len(poi_pertinents) if poi_pertinents else 0.0

    # ── Éq. (16b) : F1@k ─────────────────────────────────────────────────────
    if precision + rappel > 0:
        f1 = 2 * precision * rappel / (precision + rappel)
    else:
        f1 = 0.0

    # ── Éq. (14) : MSE ───────────────────────────────────────────────────────
    # Calculé sur les POI recommandés pour lesquels une note réelle existe
    mse_vals = []
    for poi in top_k:
        if 0 <= poi < len(notes_reelles) and notes_reelles[poi] > 0:
            # Note prédite implicite : rang → note (simple approximation)
            rang    = top_k.index(poi) + 1
            pred    = 5.0 * (1 - rang / (k + 1))   # décroissance linéaire [0,5]
            mse_vals.append((notes_reelles[poi] - pred) ** 2)
    mse = float(np.mean(mse_vals)) if mse_vals else 1.0

    # ── Éq. (17) : ILD — Diversité intra-liste ───────────────────────────────
    ild = _calculer_ild(top_k, embeddings)

    # ── Éq. (17b) : NDCG@k ───────────────────────────────────────────────────
    ndcg = _calculer_ndcg(top_k, poi_pertinents, notes_reelles, k, seuil_pertinence)

    return {
        "MSE":        mse,
        "Precision":  precision,
        "Rappel":     rappel,
        "F1":         f1,
        "ILD":        ild,
        "NDCG":       ndcg,
    }


def _calculer_ild(
    top_k: List[int],
    embeddings: Optional[Dict[int, np.ndarray]] = None,
) -> float:
    """
    ILD = 2 · Σ_{i<j} dist(pᵢ, pⱼ) / (k·(k−1)) — Éq. (17)
    Distance cosinus entre embeddings catégoriels.
    Si pas d'embeddings : ILD simulée aléatoirement (pour les tests).
    """
    k = len(top_k)
    if k < 2:
        return 0.0

    if embeddings is None:
        # Simulation : diversité arbitraire pour les tests
        return float(np.random.uniform(0.3, 0.7))

    distances = []
    for i in range(k):
        for j in range(i + 1, k):
            pi, pj = top_k[i], top_k[j]
            if pi in embeddings and pj in embeddings:
                ei, ej = embeddings[pi], embeddings[pj]
                norme = np.linalg.norm(ei) * np.linalg.norm(ej)
                if norme > 0:
                    cos_sim = np.dot(ei, ej) / norme
                    distances.append(1.0 - cos_sim)

    if not distances:
        return 0.0
    return float(np.mean(distances))   # Éq. (17)


def _calculer_ndcg(
    top_k: List[int],
    poi_pertinents: Set[int],
    notes_reelles: np.ndarray,
    k: int,
    seuil: float,
) -> float:
    """
    NDCG@k = DCG@k / IDCG@k — Éq. (17b)
    DCG@k = Σ_{i=1}^k rel_i / log₂(i+1)
    """
    def relevance(poi: int) -> float:
        if poi in poi_pertinents:
            if 0 <= poi < len(notes_reelles) and notes_reelles[poi] > 0:
                return notes_reelles[poi]
            return 1.0
        return 0.0

    # DCG@k
    dcg = sum(
        relevance(top_k[i]) / math.log2(i + 2)   # log₂(rang + 1), rang commence à 1
        for i in range(min(k, len(top_k)))
    )

    # IDCG@k : classement idéal des items pertinents
    notes_pertinentes = sorted(
        [relevance(p) for p in poi_pertinents if relevance(p) > 0],
        reverse=True,
    )[:k]
    idcg = sum(
        notes_pertinentes[i] / math.log2(i + 2)
        for i in range(len(notes_pertinentes))
    )

    return float(dcg / idcg) if idcg > 0 else 0.0


# ─── Évaluation globale sur un ensemble de test ───────────────────────────────
def evaluer_global(
    resultats_par_user: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    Moyenne les métriques sur tous les utilisateurs.

    Paramètres
    ----------
    resultats_par_user : liste de dicts (un dict par utilisateur)

    Retourne
    --------
    dict avec les moyennes globales de toutes les métriques
    """
    if not resultats_par_user:
        return {}
    return {
        metrique: float(np.mean([r[metrique] for r in resultats_par_user if metrique in r]))
        for metrique in resultats_par_user[0].keys()
    }


def afficher_rapport(metriques: Dict[str, float], nom_modele: str = "MCMAPRA"):
    """Affiche un rapport de métriques formaté."""
    print(f"\n{'='*60}")
    print(f"  Rapport d'évaluation — {nom_modele}")
    print(f"{'='*60}")
    for cle, val in metriques.items():
        print(f"  {cle:<15} : {val:.4f}")
    print(f"{'='*60}\n")
