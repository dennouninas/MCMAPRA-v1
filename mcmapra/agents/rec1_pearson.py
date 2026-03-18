"""
Rec1Ag — Agent de Recommandation basé sur la Similarité de Pearson
====================================================================
Implémente les Équations (3) et (4b) de l'article MCMAPRA.

  Éq. (3)  sim_P(u,v) = Σ(r_ui−r̄_u)(r_vi−r̄_v) / √[Σ(r_ui−r̄_u)²·Σ(r_vi−r̄_v)²]
  Éq. (4)  r̂_up = r̄_u + Σ_{v∈N(u)} sim_P(u,v)·(r_vp−r̄_v) / Σ|sim_P(u,v)|
  Éq. (4b) sim_P_shrink(u,v) = sim_P(u,v) · |I(u,v)| / (|I(u,v)| + β)
"""

from __future__ import annotations

from typing import Dict, List, Set

import numpy as np


class Rec1AgPearson:
    """
    Agent de filtrage collaboratif basé sur la corrélation de Pearson.

    Paramètres
    ----------
    R             : ndarray (m, n) — matrice d'interactions normalisée
    k_voisins     : taille du voisinage (défaut 20)
    beta_lissage  : paramètre de lissage de Pearson (défaut 25, Éq. 4b)
    """

    def __init__(
        self,
        R: np.ndarray,
        k_voisins: int = 20,
        beta_lissage: float = 25.0,
    ):
        self.R            = R.astype(float)
        self.m, self.n    = R.shape
        self.k_voisins    = k_voisins
        self.beta_lissage = beta_lissage

        # Pré-calculer les moyennes par utilisateur (ignorer les zéros)
        self._moyennes = np.zeros(self.m)
        for u in range(self.m):
            interactions = R[u][R[u] > 0]
            self._moyennes[u] = interactions.mean() if len(interactions) > 0 else 0.0

        # Cache des similarités (calculé à la demande)
        self._cache_sim: Dict[int, Dict[int, float]] = {}

    # ─── Éq. (3) + (4b) : Pearson avec lissage ───────────────────────────────
    def similarite_pearson(self, u: int, v: int) -> float:
        """
        Calcule sim_P_shrink(u, v) — Éq. (3) et (4b).

        Le facteur de lissage réduit les similarités calculées sur peu
        de co-visites (atténuation directe du cold start).
        """
        # POI co-visités (non nuls pour les deux utilisateurs)
        masque = (self.R[u] > 0) & (self.R[v] > 0)
        I_uv = np.where(masque)[0]

        if len(I_uv) == 0:
            return 0.0

        # Vecteurs centrés — Éq. (3)
        r_u = self.R[u, I_uv] - self._moyennes[u]
        r_v = self.R[v, I_uv] - self._moyennes[v]

        norme = np.sqrt(np.sum(r_u**2) * np.sum(r_v**2))
        if norme == 0:
            return 0.0

        sim_raw = float(np.dot(r_u, r_v) / norme)  # Éq. (3)

        # Lissage de Pearson — Éq. (4b) : sim_shrink = sim · |I| / (|I| + β)
        sim_shrink = sim_raw * len(I_uv) / (len(I_uv) + self.beta_lissage)

        return sim_shrink

    # ─── k plus proches voisins ───────────────────────────────────────────────
    def k_plus_proches_voisins(self, u: int) -> List[int]:
        """
        Retourne les k voisins les plus similaires à u (similarité Pearson).
        Utilise un cache pour éviter les recalculs.
        """
        if u not in self._cache_sim:
            sims = {}
            for v in range(self.m):
                if v == u:
                    continue
                s = self.similarite_pearson(u, v)
                if s > 0:
                    sims[v] = s
            # Trier par similarité décroissante et garder les k meilleurs
            self._cache_sim[u] = dict(
                sorted(sims.items(), key=lambda x: x[1], reverse=True)[:self.k_voisins]
            )
        return list(self._cache_sim[u].keys())

    # ─── Éq. (4) : Prédiction de note ────────────────────────────────────────
    def predire_note(self, u: int, poi: int) -> float:
        """
        Prédit la note de l'utilisateur u pour le POI p — Éq. (4).
        r̂_up = r̄_u + Σ sim_P(u,v)·(r_vp−r̄_v) / Σ|sim_P(u,v)|
        """
        voisins = self.k_plus_proches_voisins(u)
        sims    = self._cache_sim.get(u, {})

        numerateur   = 0.0
        denominateur = 0.0

        for v in voisins:
            if self.R[v, poi] > 0:
                s = sims.get(v, 0.0)
                numerateur   += s * (self.R[v, poi] - self._moyennes[v])
                denominateur += abs(s)

        if denominateur == 0:
            return self._moyennes[u]

        return self._moyennes[u] + numerateur / denominateur

    # ─── Classement des candidats ─────────────────────────────────────────────
    def classer(self, u: int, candidats: List[int]) -> List[int]:
        """
        Classe les POI candidats par note prédite décroissante (Rec1Ag).

        Retourne
        --------
        liste ordonnée des poi_id du meilleur au moins bon
        """
        scores = {poi: self.predire_note(u, poi) for poi in candidats}
        return sorted(scores.keys(), key=lambda p: scores[p], reverse=True)
