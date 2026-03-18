"""
Rec2Ag — Agent de Recommandation basé sur la Similarité de Jaccard
====================================================================
Implémente les Équations (5) et (6b) de l'article MCMAPRA.

  Éq. (5)  sim_J(u,v)  = |V(u) ∩ V(v)| / |V(u) ∪ V(v)|
  Éq. (6)  score_J(u,p)= Σ_{v∈N(u)} sim_J(u,v)·𝟙[r_vp>0] / |N(u)|
  Éq. (6b) sim_JW(u,v) = Σ_{p∈V(u)∩V(v)} idf(p) / (||V(u)||_idf + ||V(v)||_idf - Σ idf(p))
           avec idf(p) = log(m / df(p) + 1)
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


class Rec2AgJaccard:
    """
    Agent de filtrage collaboratif basé sur la similarité de Jaccard.

    Capture les ensembles binaires de visites, indépendamment des amplitudes
    de notes — plus robuste au bruit de notation en cold start.

    Paramètres
    ----------
    R          : ndarray (m, n) — matrice d'interactions
    k_voisins  : taille du voisinage (défaut 20)
    utiliser_idf : si True, utilise Jaccard pondéré TF-IDF (Éq. 6b)
    """

    def __init__(
        self,
        R: np.ndarray,
        k_voisins: int = 20,
        utiliser_idf: bool = True,
    ):
        self.R           = R.astype(float)
        self.m, self.n   = R.shape
        self.k_voisins   = k_voisins
        self.utiliser_idf = utiliser_idf

        # Ensembles de visites binaires pour chaque utilisateur
        self._visites: List[frozenset] = [
            frozenset(np.where(R[u] > 0)[0]) for u in range(self.m)
        ]

        # Pré-calculer les vecteurs IDF si nécessaire
        if utiliser_idf:
            self._idf = self._calculer_idf()
        else:
            self._idf = np.ones(self.n)

        # Cache des voisins
        self._cache_voisins: Dict[int, List[int]] = {}

    # ─── Éq. (5) : Jaccard standard ──────────────────────────────────────────
    def _jaccard_standard(self, u: int, v: int) -> float:
        """sim_J(u,v) = |V(u) ∩ V(v)| / |V(u) ∪ V(v)| — Éq. (5)"""
        Vu, Vv = self._visites[u], self._visites[v]
        inter = len(Vu & Vv)
        union = len(Vu | Vv)
        return inter / union if union > 0 else 0.0

    # ─── Éq. (6b) : Jaccard pondéré TF-IDF ───────────────────────────────────
    def _calculer_idf(self) -> np.ndarray:
        """
        idf(p) = log( m / df(p) + 1 )
        Réduit le biais vers les lieux très populaires (longue traîne).
        """
        df = np.sum(self.R > 0, axis=0).astype(float)   # fréquence documentaire
        return np.log(self.m / (df + 1) + 1)

    def _jaccard_idf(self, u: int, v: int) -> float:
        """
        sim_JW(u,v) — Éq. (6b) : Jaccard pondéré par l'IDF des POI.
        Atténue le poids des lieux populaires, valorise les co-visites rares.
        """
        Vu, Vv = self._visites[u], self._visites[v]
        inter  = list(Vu & Vv)
        if not inter:
            return 0.0

        sum_idf_inter = sum(self._idf[p] for p in inter)
        norm_u = sum(self._idf[p] for p in Vu)
        norm_v = sum(self._idf[p] for p in Vv)
        denom  = norm_u + norm_v - sum_idf_inter
        return sum_idf_inter / denom if denom > 0 else 0.0

    # ─── Similarité (dispatch standard / IDF) ────────────────────────────────
    def similarite(self, u: int, v: int) -> float:
        """Retourne la similarité de Jaccard (standard ou pondérée selon config)."""
        if self.utiliser_idf:
            return self._jaccard_idf(u, v)
        return self._jaccard_standard(u, v)

    # ─── k plus proches voisins ───────────────────────────────────────────────
    def k_plus_proches_voisins(self, u: int) -> List[int]:
        """Retourne les k voisins les plus similaires (Jaccard)."""
        if u not in self._cache_voisins:
            sims = {
                v: self.similarite(u, v)
                for v in range(self.m) if v != u
            }
            voisins = sorted(sims, key=sims.get, reverse=True)[:self.k_voisins]
            self._cache_voisins[u] = voisins
        return self._cache_voisins[u]

    # ─── Éq. (6) : Score binaire d'intérêt ───────────────────────────────────
    def score_interet(self, u: int, poi: int) -> float:
        """
        score_J(u,p) = Σ sim_J(u,v)·𝟙[r_vp>0] / |N(u)| — Éq. (6)
        """
        voisins = self.k_plus_proches_voisins(u)
        if not voisins:
            return 0.0

        total = sum(
            self.similarite(u, v)
            for v in voisins if poi in self._visites[v]
        )
        return total / len(voisins)

    # ─── Classement des candidats ─────────────────────────────────────────────
    def classer(self, u: int, candidats: List[int]) -> List[int]:
        """
        Classe les POI candidats par score d'intérêt Jaccard décroissant (Rec2Ag).
        """
        scores = {poi: self.score_interet(u, poi) for poi in candidats}
        return sorted(scores.keys(), key=lambda p: scores[p], reverse=True)
