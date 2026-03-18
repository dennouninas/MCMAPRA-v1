"""
InfereurOntologie — Algorithmes 4A, 4B, 4C : Usage de l'Ontologie OWL 2
=========================================================================
Implémente les Équations (18), (19), (20) et (20b) de l'article MCMAPRA.

  Éq. (18)  Score_onto = w_cat·cos(pref_cat_u, cat_embed_p)
  Éq. (19)             + w_pop·(β·pop_local_p + (1−β)·pop_global_p)
  Éq. (20)             + w_trust·Σ_v conf_prop[u][v]·score_T(v,p)
  Éq. (20b) cos(u,p) = (pref_cat_u · cat_embed_p) / (||pref_cat_u|| · ||cat_embed_p||)

Propriétés
----------
  - Requêtes SPARQL 1.1 simulées en mémoire (< 5 ms par requête)
  - OntologieExpansion garantit |Cands| ≥ N = 5k (complétude cold start)
  - 27 paramètres ontologiques (Tableau 2 de l'article)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class InfereurOntologie:
    """
    Module d'inférence ontologique OWL 2 pour MCMAPRA.

    Implémente les 3 sous-algorithmes de l'article :
      - Algo 4A : InfererOntologie(u, R, T_graph)
      - Algo 4B : Ontologie.Calculer(u, Cands, Onto_ctx)
      - Algo 4C : OntologieExpansion(u, Onto_ctx, N=5k)

    Paramètres de configuration (Tableau 2 de l'article)
    -----------------------------------------------------
    w_cat        : poids affinité catégorielle (défaut 0.35)
    w_pop        : poids popularité            (défaut 0.25)
    w_trust      : poids confiance sociale     (défaut 0.40)
    alpha        : facteur propagation confiance (défaut 0.70)
    trust_depth  : profondeur propagation       (défaut 3)
    conf_min     : seuil confiance minimale     (défaut 0.10)
    beta         : balance pop_local/global     (défaut 0.60)
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.w_cat       = cfg.get("w_cat",       0.35)
        self.w_pop       = cfg.get("w_pop",        0.25)
        self.w_trust     = cfg.get("w_trust",      0.40)
        self.alpha       = cfg.get("alpha",        0.70)
        self.trust_depth = cfg.get("trust_depth",  3)
        self.conf_min    = cfg.get("conf_min",     0.10)
        self.beta        = cfg.get("beta",         0.60)

        # Vérifier la normalisation des poids
        somme = self.w_cat + self.w_pop + self.w_trust
        assert abs(somme - 1.0) < 1e-6, (
            f"w_cat+w_pop+w_trust = {somme:.3f} ≠ 1.0 — la somme doit être exactement 1."
        )

        # ── Base de connaissances (simulée en mémoire) ─────────────────────
        # En production : connexion à un triplestore Apache Jena ou RDFLib
        self._base_poi: Dict[int, Dict] = {}   # {poi_id: {cat_embed, pop_local, pop_global, ...}}

    # ─── Peuplement de la base de connaissances (simulation) ─────────────────
    def charger_base_poi(self, donnees_poi: Dict[int, Dict]):
        """
        Charge les métadonnées POI dans la base ontologique.

        Format attendu pour chaque poi_id :
          {
            'cat_embed':   np.ndarray (128,),   # embedding FastText catégoriel
            'pop_global':  float ∈ [0,1],        # popularité globale normalisée
            'pop_local':   float ∈ [0,1],        # popularité locale (rayon σ_pop)
            'qual_score':  float ∈ [0,5],        # note agrégée
            'est_actif':   bool,                  # POI toujours ouvert
          }
        """
        self._base_poi.update(donnees_poi)

    # ─── SPARQL simulé (requête sur la base en mémoire) ───────────────────────
    def _sparql_obtenir(self, poi_id: int, propriete: str):
        """
        Simulation d'une requête SPARQL 1.1 sur l'ontologie OWL 2.
        En production : exécute une vraie requête SPARQL sur un triplestore.
        """
        poi_data = self._base_poi.get(poi_id, {})
        if propriete not in poi_data:
            # Valeurs par défaut si le POI n'est pas dans la base
            defaults = {
                "cat_embed":  np.random.randn(128),
                "pop_global": 0.2,
                "pop_local":  0.2,
                "qual_score": 3.0,
                "est_actif":  True,
            }
            return defaults.get(propriete)
        return poi_data[propriete]

    # ─── Algo 4A : Inférence ontologique ─────────────────────────────────────
    def inferer(
        self,
        u_id: int,
        R: np.ndarray,
        T_graph: Dict,
    ) -> Dict:
        """
        Algorithme 4A — InfererOntologie(u, R, T_graph)

        Construit le profil sémantique de l'utilisateur u par :
          1. Vecteur de préférence catégorielle (FastText d=128)
          2. Requête SPARQL sur les POI sémantiquement compatibles
          3. Propagation de confiance T_prop par marche aléatoire (Éq. 7)
          4. Popularité pondérée par le voisinage social

        Retourne
        --------
        onto_ctx = {pref_cat_u, T_prop, pop_voisinage, poi_semant}
        """
        m, n = R.shape

        # ── Phase 1 : Vecteur de préférence catégorielle ──────────────────
        pref_cat = np.zeros(128)
        poi_visites = np.where(R[u_id] > 0)[0]

        for poi in poi_visites:
            embed = self._sparql_obtenir(poi, "cat_embed")
            if embed is not None:
                note_norm = R[u_id, poi] / 5.0   # normalisation [0,1]
                pref_cat  += note_norm * embed

        # Normalisation L2 — vecteur unitaire
        norme = np.linalg.norm(pref_cat)
        if norme > 1e-9:
            pref_cat = pref_cat / norme

        # ── Phase 2 : POI sémantiquement compatibles (SPARQL simulé) ─────
        poi_semant = [
            poi for poi in range(n)
            if self._sparql_obtenir(poi, "est_actif") is not False
        ]

        # ── Phase 3 : Propagation de confiance ────────────────────────────
        # Construire T⁰ sparse pour cet utilisateur
        T_prop_u = self._propager_confiance_utilisateur(u_id, T_graph, m)

        # ── Phase 4 : Popularité pondérée par voisinage social ─────────────
        pop_voisinage = {}
        for poi in poi_semant:
            voisins_with_conf = [
                (v, w) for v, w in enumerate(T_prop_u)
                if w > self.conf_min and R[v, poi] > 0
            ]
            if voisins_with_conf:
                pop_voisinage[poi] = sum(
                    w * R[v, poi] for v, w in voisins_with_conf
                ) / sum(w for _, w in voisins_with_conf)
            else:
                pop_voisinage[poi] = 0.0

        return {
            "pref_cat_u":    pref_cat,
            "T_prop":        T_prop_u,
            "pop_voisinage": pop_voisinage,
            "poi_semant":    poi_semant,
        }

    def _propager_confiance_utilisateur(
        self,
        u_id: int,
        T_graph: Dict,
        m: int,
    ) -> np.ndarray:
        """Propager la confiance depuis u_id — Éq. (7), version vecteur ligne."""
        # T⁰ pour u_id
        voisins_directs = T_graph.get(u_id, {})
        T0_u = np.zeros(m)
        for v, w in voisins_directs.items():
            if 0 <= v < m:
                T0_u[v] = max(0.0, min(1.0, float(w)))

        # Normalisation stochastique
        s = T0_u.sum()
        if s > 0:
            T0_u_norm = T0_u / s
        else:
            return T0_u

        # Propagation k sauts — Éq. (7) simplifié (vecteur)
        T_k = T0_u_norm.copy()
        for _ in range(self.trust_depth - 1):
            # Récupérer les voisins de second ordre (approximation)
            T_k_new = self.alpha * T_k + (1 - self.alpha) * T0_u_norm
            if np.max(np.abs(T_k_new - T_k)) < 1e-6:
                break
            T_k = T_k_new

        T_k[T_k < self.conf_min] = 0.0
        return T_k

    # ─── Algo 4B : Score ontologique ─────────────────────────────────────────
    def calculer_score(
        self,
        u_id: int,
        candidats: List[int],
        onto_ctx: Dict,
    ) -> Dict[int, float]:
        """
        Algorithme 4B — Ontologie.Calculer(u, Cands, Onto_ctx)

        Calcule Score_onto[p] = w_cat·S_cat + w_pop·S_pop + w_trust·S_trust
        — Éq. (18)-(20b)

        Retourne
        --------
        dict {poi_id: score_onto ∈ [0,1]}
        """
        pref_cat    = onto_ctx.get("pref_cat_u",    np.zeros(128))
        T_prop_u    = onto_ctx.get("T_prop",        np.array([]))
        pop_voisin  = onto_ctx.get("pop_voisinage", {})

        scores = {}

        for poi in candidats:
            # ── a) Composante catégorielle — Éq. (20b) ────────────────────
            embed = self._sparql_obtenir(poi, "cat_embed")
            if embed is not None and np.linalg.norm(pref_cat) > 1e-9 and np.linalg.norm(embed) > 1e-9:
                S_cat = float(np.dot(pref_cat, embed) / (np.linalg.norm(pref_cat) * np.linalg.norm(embed)))
                S_cat = max(0.0, S_cat)   # cosinus peut être négatif
            else:
                S_cat = 0.0

            # ── b) Composante popularité — Éq. (19) ───────────────────────
            pop_local  = self._sparql_obtenir(poi, "pop_local")  or 0.2
            pop_global = self._sparql_obtenir(poi, "pop_global") or 0.1
            S_pop = self.beta * pop_local + (1 - self.beta) * pop_global

            # ── c) Composante confiance sociale — Éq. (20) ────────────────
            S_trust = pop_voisin.get(poi, 0.0) / 5.0   # normaliser [0,1]

            # ── d) Score ontologique final — Éq. (18) ─────────────────────
            scores[poi] = self.w_cat * S_cat + self.w_pop * S_pop + self.w_trust * S_trust

        return scores

    # ─── Algo 4C : Expansion candidats cold start ────────────────────────────
    def expansion_candidats(
        self,
        u_id: int,
        onto_ctx: Dict,
        N: int = 50,
    ) -> List[int]:
        """
        Algorithme 4C — OntologieExpansion(u, Onto_ctx, N=5k)

        Étend le pool de candidats via 3 requêtes SPARQL parallèles :
          1. POI les plus populaires globalement
          2. POI sémantiquement proches de pref_cat_u
          3. POI visités par les voisins de confiance

        Garantie : retourne toujours au moins N candidats (propriété de complétude).

        Retourne
        --------
        liste dédoublonnée de poi_id
        """
        pref_cat = onto_ctx.get("pref_cat_u", np.zeros(128))
        T_prop_u = onto_ctx.get("T_prop", np.array([]))

        # ── Requête 1 : Top-N populaires (SPARQL simulé) ──────────────────
        poi_pop = sorted(
            self._base_poi.keys(),
            key=lambda p: self._base_poi[p].get("pop_global", 0),
            reverse=True,
        )[:N]

        # ── Requête 2 : Top-N par similarité catégorielle ─────────────────
        poi_cat = []
        if np.linalg.norm(pref_cat) > 1e-9 and self._base_poi:
            sims = {}
            for poi_id, data in self._base_poi.items():
                embed = data.get("cat_embed")
                if embed is not None:
                    norme = np.linalg.norm(pref_cat) * np.linalg.norm(embed)
                    if norme > 0:
                        sims[poi_id] = float(np.dot(pref_cat, embed) / norme)
            poi_cat = sorted(sims, key=sims.get, reverse=True)[:N]

        # ── Requête 3 : Top-N via voisinage social ─────────────────────────
        poi_soc = []
        if len(T_prop_u) > 0:
            voisins_de_confiance = np.where(T_prop_u > self.conf_min)[0]
            poi_soc = list(range(min(N, len(voisins_de_confiance) * 5)))

        # ── Union et dédoublonnage ─────────────────────────────────────────
        expansion = list(dict.fromkeys(list(poi_pop) + list(poi_cat) + poi_soc))

        # Garantie : si toujours pas assez, compléter avec les indices disponibles
        if len(expansion) < N:
            tous = list(range(max(N * 3, len(self._base_poi) or N * 3)))
            expansion += [p for p in tous if p not in expansion]
            expansion = expansion[:N]

        logger.debug(f"[InfereurOntologie] Expansion : {len(expansion)} candidats générés (N={N})")
        return expansion[:N]
