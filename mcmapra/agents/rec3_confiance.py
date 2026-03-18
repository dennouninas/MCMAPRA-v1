"""
Rec3Ag — Agent de Recommandation basé sur la Propagation de Confiance
=======================================================================
Implémente les Équations (7), (8) et (8b-8c) de l'article MCMAPRA.

  Éq. (7)  T^(k) = α·T·T^(k−1) + (1−α)·T⁰,   T^(0) = T⁰
  Éq. (8)  score_T(u,p) = Σ_v T^(k)[u][v]·r_vp / Σ_v T^(k)[u][v]·𝟙[r_vp>0]
  Éq. (8b) T_norm[u][v] = T⁰[u][v] / Σ_w T⁰[u][w]   (normalisation stochastique)
  Éq. (8c) ||α·T_norm||_spectral < 1  ⟺  α < 1/ρ(T_norm)  (convergence garantie)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


class Rec3AgConfiance:
    """
    Agent de recommandation basé sur la propagation de confiance sociale.

    La confiance est diffusée dans le graphe social via un Random Walk
    itératif (Éq. 7). La normalisation stochastique (Éq. 8b) garantit
    la convergence (Éq. 8c).

    Paramètres
    ----------
    R           : ndarray (m, n) — matrice d'interactions
    T0          : dict {u: {v: w}} — matrice de confiance directe T⁰[u][v] ∈ [0,1]
    alpha       : facteur de propagation (défaut 0.7)
    trust_depth : nombre de sauts k (défaut 3 — optimal gain/latence)
    conf_min    : seuil minimal de confiance pour élagage (défaut 0.1)
    """

    def __init__(
        self,
        R: np.ndarray,
        T0: Dict[int, Dict[int, float]],
        alpha: float = 0.70,
        trust_depth: int = 3,
        conf_min: float = 0.10,
    ):
        self.R           = R.astype(float)
        self.m, self.n   = R.shape
        self.alpha       = alpha
        self.trust_depth = trust_depth
        self.conf_min    = conf_min

        # Construire et normaliser la matrice T⁰
        self._T0_dict    = T0
        self._T0_matrix  = self._construire_matrice(T0)
        self._T0_norm    = self._normaliser_stochastique(self._T0_matrix)

        # Vérifier la condition de convergence — Éq. (8c)
        self._verifier_convergence()

        # Calculer la confiance propagée T^(k)
        self._T_prop = self._propager(self.trust_depth)

        # Cache des classements
        self._cache_scores: Dict[int, Dict[int, float]] = {}

    # ─── Construction de la matrice de confiance ──────────────────────────────
    def _construire_matrice(self, T0_dict: Dict) -> np.ndarray:
        """Convertit le dictionnaire de confiance en matrice ndarray."""
        T = np.zeros((self.m, self.m))
        for u, voisins in T0_dict.items():
            for v, w in voisins.items():
                if 0 <= u < self.m and 0 <= v < self.m:
                    T[u, v] = max(0.0, min(1.0, float(w)))
        return T

    # ─── Éq. (8b) : Normalisation stochastique ───────────────────────────────
    def _normaliser_stochastique(self, T: np.ndarray) -> np.ndarray:
        """
        T_norm[u][v] = T⁰[u][v] / Σ_w T⁰[u][w] — Éq. (8b)
        Chaque ligne somme à 1 (ou reste à 0 si aucune confiance).
        """
        T_norm = T.copy()
        sommes = T_norm.sum(axis=1, keepdims=True)
        sommes[sommes == 0] = 1.0   # éviter la division par zéro
        return T_norm / sommes

    # ─── Éq. (8c) : Vérification de la convergence ───────────────────────────
    def _verifier_convergence(self):
        """
        Vérifie ||α·T_norm||_spectral < 1 — Éq. (8c).
        Garantit la convergence de la série T^(k).
        """
        if self.m == 0:
            return
        try:
            # Rayon spectral approximatif (puissance itérée)
            rho = np.max(np.abs(np.linalg.eigvals(self._T0_norm)))
            seuil_alpha = 1.0 / max(rho, 1e-9)
            if self.alpha >= seuil_alpha:
                logger.warning(
                    f"[Rec3Ag] α={self.alpha} ≥ 1/ρ(T_norm)={seuil_alpha:.3f} "
                    f"— convergence non garantie. Réduction automatique."
                )
                self.alpha = min(self.alpha, seuil_alpha * 0.95)
        except np.linalg.LinAlgError:
            logger.warning("[Rec3Ag] Impossible de calculer le rayon spectral.")

    # ─── Éq. (7) : Propagation par Random Walk itératif ──────────────────────
    def _propager(self, k: int) -> np.ndarray:
        """
        T^(k) = α·T_norm·T^(k−1) + (1−α)·T⁰_norm — Éq. (7)

        Itère k fois à partir de T^(0) = T⁰_norm.
        Complexité : O(k · m²) — acceptable pour m ≤ 100 000 avec sparse.
        """
        T_k = self._T0_norm.copy()   # T^(0) = T⁰_norm

        for iteration in range(1, k + 1):
            T_k_nouveau = (
                self.alpha * (self._T0_norm @ T_k)
                + (1 - self.alpha) * self._T0_norm
            )
            # Test de convergence anticipée
            delta = np.max(np.abs(T_k_nouveau - T_k))
            T_k = T_k_nouveau
            if delta < 1e-6:
                logger.debug(f"[Rec3Ag] Convergence à l'itération {iteration} (delta={delta:.2e})")
                break

        # Élagage des arêtes sous le seuil de confiance minimale
        T_k[T_k < self.conf_min] = 0.0

        return T_k

    # ─── Éq. (8) : Score de recommandation ───────────────────────────────────
    def score_confiance(self, u: int, poi: int) -> float:
        """
        score_T(u,p) = Σ_v T^(k)[u][v]·r_vp / Σ_v T^(k)[u][v]·𝟙[r_vp>0] — Éq. (8)
        """
        confiances = self._T_prop[u]   # vecteur ligne de taille m

        voisins_avec_visite = np.where((confiances > 0) & (self.R[:, poi] > 0))[0]
        if len(voisins_avec_visite) == 0:
            return 0.0

        numerateur   = np.sum(confiances[voisins_avec_visite] * self.R[voisins_avec_visite, poi])
        denominateur = np.sum(confiances[voisins_avec_visite])

        return float(numerateur / denominateur) if denominateur > 0 else 0.0

    # ─── Classement des candidats ─────────────────────────────────────────────
    def classer(self, u: int, candidats: List[int]) -> List[int]:
        """
        Classe les POI candidats par score de confiance décroissant (Rec3Ag).
        """
        scores = {poi: self.score_confiance(u, poi) for poi in candidats}
        return sorted(scores.keys(), key=lambda p: scores[p], reverse=True)

    # ─── Accesseurs ───────────────────────────────────────────────────────────
    @property
    def matrice_confiance_propagee(self) -> np.ndarray:
        """Retourne T^(k) — la matrice de confiance propagée."""
        return self._T_prop.copy()

    def confiance(self, u: int, v: int) -> float:
        """Retourne T^(k)[u][v] — confiance propagée de u vers v."""
        if 0 <= u < self.m and 0 <= v < self.m:
            return float(self._T_prop[u, v])
        return 0.0
