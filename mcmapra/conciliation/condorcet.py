"""
CondorcetAg — Algorithme 2B : Agent de Conciliation par Vote de Condorcet
==========================================================================
Implémente les Équations (10) et (11) de l'article MCMAPRA.

  Éq. (10) pᵢ ≻ pⱼ ⟺ |{a∈A : rankₐ(pᵢ) < rankₐ(pⱼ)}| > |A|/2
  Éq. (11) Copeland(p) = |{q : p≻q}| − |{q : q≻p}|    [résolution cycles]

Propriétés formelles
--------------------
  ✓ Critère de Pareto          : satisfait si acyclique
  ✓ Indépendance alternatives  : satisfait si acyclique
  ✗ Robustesse cycles          : Copeland garantit la terminaison (8,3% des cas)
  ✓ Complexité                 : O(|A|·|C|²)  comparaisons + O(|C|²) Copeland
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CondorcetAg:
    """
    Agent de conciliation par vote de Condorcet avec résolution Copeland.

    Construit la matrice de domination par paires (Éq. 10), puis produit
    un classement par tri topologique. En cas de cycle (paradoxe de Condorcet),
    bascule automatiquement sur le score de Copeland (Éq. 11).
    """

    # ─── Éq. (10) : Matrice de domination ────────────────────────────────────
    def _construire_matrice_domination(
        self,
        listes: List[List[int]],
    ) -> Tuple[List[int], Dict[Tuple[int,int], bool]]:
        """
        Construit Dom[pᵢ][pⱼ] = True si pᵢ ≻ pⱼ — Éq. (10).

        pᵢ ≻ pⱼ ssi la majorité stricte des agents préfère pᵢ à pⱼ
        (c.-à-d. au moins 2 agents sur 3).
        """
        # Union de tous les candidats
        tous = list(dict.fromkeys(p for L in listes for p in L))
        n_agents = len(listes)

        # Construire les rangs par agent
        rangs = {}
        for a, L_a in enumerate(listes):
            for rang, poi in enumerate(L_a, start=1):
                rangs[(a, poi)] = rang
            # Les POI absents reçoivent un rang maximal
            for poi in tous:
                if (a, poi) not in rangs:
                    rangs[(a, poi)] = len(L_a) + 1

        # Matrice de domination
        dom: Dict[Tuple[int,int], bool] = {}
        for pi in tous:
            for pj in tous:
                if pi == pj:
                    continue
                # Nombre d'agents préférant pi à pj
                votes_pi = sum(
                    1 for a in range(n_agents)
                    if rangs[(a, pi)] < rangs[(a, pj)]
                )
                dom[(pi, pj)] = votes_pi > n_agents / 2   # Éq. (10)

        return tous, dom

    # ─── Détection de cycle (DFS) ─────────────────────────────────────────────
    def _detecter_cycle(
        self,
        candidats: List[int],
        dom: Dict[Tuple[int,int], bool],
    ) -> bool:
        """
        Détecte un cycle dans le graphe de domination par DFS.
        Retourne True si un cycle est détecté (paradoxe de Condorcet).
        """
        visites  = set()
        en_cours = set()

        def dfs(noeud: int) -> bool:
            visites.add(noeud)
            en_cours.add(noeud)
            for autre in candidats:
                if autre == noeud:
                    continue
                if dom.get((noeud, autre), False):
                    if autre in en_cours:
                        return True   # cycle trouvé
                    if autre not in visites:
                        if dfs(autre):
                            return True
            en_cours.discard(noeud)
            return False

        for c in candidats:
            if c not in visites:
                if dfs(c):
                    return True
        return False

    # ─── Tri topologique ─────────────────────────────────────────────────────
    def _tri_topologique(
        self,
        candidats: List[int],
        dom: Dict[Tuple[int,int], bool],
    ) -> List[int]:
        """
        Ordonne les candidats par tri topologique sur le DAG de domination.
        Utilisé quand il n'y a pas de cycle.
        """
        # Degré entrant (nombre de candidats qui dominent p)
        degre_entrant = {p: 0 for p in candidats}
        for pi in candidats:
            for pj in candidats:
                if pi != pj and dom.get((pi, pj), False):
                    degre_entrant[pj] = degre_entrant.get(pj, 0)
                    # pi ≻ pj → pj a un "dominateur" de plus
                    degre_entrant[pj] += 0  # on veut les sources

        # En fait on veut trier les "gagnants" (peu dominés) en premier
        score_dom = {
            p: sum(1 for q in candidats if q != p and dom.get((p, q), False))
            for p in candidats
        }
        return sorted(candidats, key=lambda p: score_dom[p], reverse=True)

    # ─── Éq. (11) : Score de Copeland ────────────────────────────────────────
    def _calculer_copeland(
        self,
        candidats: List[int],
        dom: Dict[Tuple[int,int], bool],
    ) -> Dict[int, int]:
        """
        Copeland(p) = |{q : p≻q}| − |{q : q≻p}| — Éq. (11)
        Résout les cycles du paradoxe de Condorcet.
        """
        scores = {}
        for p in candidats:
            victoires = sum(1 for q in candidats if q != p and dom.get((p, q), False))
            defaites  = sum(1 for q in candidats if q != p and dom.get((q, p), False))
            scores[p] = victoires - defaites
        return scores

    # ─── Fusion principale ────────────────────────────────────────────────────
    def fusionner(
        self,
        L1: List[int],
        L2: List[int],
        L3: List[int],
    ) -> Dict[int, int]:
        """
        Algorithme 2B — CondorcetAg.Fusionner(L₁, L₂, L₃)

        Fusionne trois classements par vote de Condorcet.
        Bascule sur Copeland si un cycle est détecté (Éq. 11).

        Retourne
        --------
        dict {poi_id: rang_final}  (1 = meilleur)
        """
        listes = [L1, L2, L3]
        candidats, dom = self._construire_matrice_domination(listes)

        # Détection du paradoxe de Condorcet
        if self._detecter_cycle(candidats, dom):
            logger.debug("[CondorcetAg] Paradoxe de Condorcet détecté — basculement sur Copeland (Éq. 11)")
            scores_copeland = self._calculer_copeland(candidats, dom)
            classement = sorted(candidats, key=lambda p: scores_copeland[p], reverse=True)
        else:
            classement = self._tri_topologique(candidats, dom)

        return {poi: rang + 1 for rang, poi in enumerate(classement)}
