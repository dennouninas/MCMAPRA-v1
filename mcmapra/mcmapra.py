"""
MCMAPRA — Algorithme 1 : Orchestration Principale
==================================================
Modèle de Conciliation Multi-Agent des Algorithmes de Recommandation de POI

Auteurs : [Auteur Un, Auteur Deux, Auteur Trois]
Article  : IEEE Access 2025 — «Un modèle multi-agent de conciliation enrichi par
           ontologie et IoT de localisation pour atténuer le problème du cold start
           dans les systèmes de recommandation de POI dans les LBSN»
GitHub   : https://github.com/[votre_compte]/mcmapra

Algorithme 1 : MCMAPRA(u, k, statut_cs)
-----------------------------------------
Entrée  : u          — identifiant de l'utilisateur cible
          k          — nombre de recommandations souhaité (Top-k)
          statut_cs  — statut cold start ∈ {'CS-0','CS-3','CS-10','warm'}
Sortie  : liste_finale — liste de k POI recommandés triés par score décroissant
Complexité : O(|N(u)| · |Cands| + |Cands|² · |A|)  avec |A| = 3 agents
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from mcmapra.agents.rec1_pearson   import Rec1AgPearson
from mcmapra.agents.rec2_jaccard   import Rec2AgJaccard
from mcmapra.agents.rec3_confiance import Rec3AgConfiance
from mcmapra.conciliation.borda      import BordaAg
from mcmapra.conciliation.condorcet  import CondorcetAg
from mcmapra.iot.module_iot          import ModuleIoT
from mcmapra.ontologie.infereur      import InfereurOntologie
from mcmapra.evaluation.metriques    import calculer_metriques

logger = logging.getLogger(__name__)


# ─── Constantes ──────────────────────────────────────────────────────────────
STATUTS_CS = {"CS-0", "CS-3", "CS-10", "warm"}
LAMBDA_OPTIMAL = 0.30          # Éq. (12) — poids fusion IoT vs vote CF
K_VOISINS_CF   = 20            # taille du voisinage k-NN


# ─── Classe principale ────────────────────────────────────────────────────────
class MCMAPRA:
    """
    Modèle de Conciliation Multi-Agent pour la Recommandation de POI.

    Architecture à 5 couches :
      ① Données       : matrice R, graphe T, IoT, Ontologie
      ② Pré-traitement: normalisation, SPARQL, fusion capteurs
      ③ Agents        : Rec1Ag (Pearson), Rec2Ag (Jaccard), Rec3Ag (Confiance)
      ④ Conciliation  : BordaAg, CondorcetAg
      ⑤ Fusion        : Score_geo + Score_onto → Score_final adaptatif
    """

    def __init__(
        self,
        matrice_R: np.ndarray,
        graphe_confiance: Dict,
        config_iot: Optional[Dict] = None,
        config_onto: Optional[Dict] = None,
        lambda_fusion: float = LAMBDA_OPTIMAL,
        k_voisins: int = K_VOISINS_CF,
        agent_conciliation: str = "borda",   # "borda" | "condorcet"
        verbeux: bool = False,
    ):
        """
        Paramètres
        ----------
        matrice_R          : ndarray (m, n) — notes normalisées [0..5], 0=non visité
        graphe_confiance   : dict {u_id: {v_id: poids}} — T⁰[u][v] ∈ [0,1]
        config_iot         : dict de paramètres IoT (sigma, tau_decay, etc.)
        config_onto        : dict de paramètres ontologiques (w_cat, w_pop, w_trust)
        lambda_fusion      : λ ∈ [0,1] — pondération Score_vote vs Score_geo (Éq. 12)
        k_voisins          : taille du voisinage CF (défaut 20)
        agent_conciliation : "borda" (optimal) ou "condorcet"
        verbeux            : afficher les logs de progression
        """
        self.R              = matrice_R
        self.T_graph        = graphe_confiance
        self.lambda_fusion  = lambda_fusion
        self.k_voisins      = k_voisins
        self.verbeux        = verbeux

        # ── Paramètres IoT par défaut (Tableau 3 de l'article) ──
        self.config_iot = config_iot or {
            "sigma":       500,    # rayon d'influence [m]
            "d_min":        50,    # distance minimale filtre [m]
            "d_max":      5000,    # distance maximale filtre [m]
            "tau_decay":   0.85,   # taux de décroissance temporelle
            "w_realtime":  0.60,   # poids signal temps réel
            "rho_min":     0.10,   # densité minimale
            "Delta_t":      30,    # fenêtre temporelle [min]
        }

        # ── Paramètres ontologiques par défaut (Tableau 2 de l'article) ──
        self.config_onto = config_onto or {
            "w_cat":    0.35,   # poids affinité catégorielle
            "w_pop":    0.25,   # poids popularité
            "w_trust":  0.40,   # poids confiance sociale
            "alpha":    0.70,   # facteur de propagation de confiance
            "trust_depth": 3,   # profondeur de propagation
            "conf_min": 0.10,   # seuil minimum de confiance
            "beta":     0.60,   # balance pop_local / pop_global
        }

        # ── Instanciation des agents ──────────────────────────────────────
        self.rec1 = Rec1AgPearson(self.R, k_voisins=self.k_voisins)
        self.rec2 = Rec2AgJaccard(self.R, k_voisins=self.k_voisins)
        self.rec3 = Rec3AgConfiance(self.R, self.T_graph, **{
            k: self.config_onto[k] for k in ["alpha","trust_depth","conf_min"]
        })

        # ── Agents de conciliation ────────────────────────────────────────
        if agent_conciliation == "borda":
            self.agent_vote = BordaAg()
        elif agent_conciliation == "condorcet":
            self.agent_vote = CondorcetAg()
        else:
            raise ValueError(f"agent_conciliation doit être 'borda' ou 'condorcet', reçu : {agent_conciliation}")

        # ── Modules enrichissants ─────────────────────────────────────────
        self.module_iot  = ModuleIoT(**self.config_iot)
        self.infereur    = InfereurOntologie(self.config_onto)

        if verbeux:
            logging.basicConfig(level=logging.INFO)

    # ─────────────────────────────────────────────────────────────────────────
    def recommander(
        self,
        u_id: int,
        k: int = 10,
        statut_cs: str = "warm",
        contexte_iot: Optional[Dict] = None,
    ) -> List[Tuple[int, float]]:
        """
        Algorithme 1 — MCMAPRA(u, k, statut_cs)

        Étapes
        ------
        1. Collecte des données contextuelles
        2. Construction du pool de candidats (avec expansion cold start)
        3. Exécution parallèle des 3 agents de recommandation
        4. Conciliation par vote (Borda ou Condorcet)
        5. Calcul des scores enrichissants (IoT + Ontologie)
        6. Fusion adaptative et classement Top-k

        Retourne
        --------
        liste_finale : [(poi_id, score_final), ...] triée par score décroissant
        """
        assert statut_cs in STATUTS_CS, f"statut_cs invalide : {statut_cs}"
        t0 = time.perf_counter()

        # ── Étape 1 : Collecte des contextes ─────────────────────────────
        logger.info("[MCMAPRA] Étape 1 — Collecte des données contextuelles")
        iot_ctx   = self.module_iot.collecter_contexte(u_id, contexte_iot)
        onto_ctx  = self.infereur.inferer(u_id, self.R, self.T_graph)

        # ── Étape 2 : Pool de candidats ───────────────────────────────────
        logger.info("[MCMAPRA] Étape 2 — Construction des candidats")
        voisins   = self.rec1.k_plus_proches_voisins(u_id)
        poi_visites = set(np.where(self.R[u_id] > 0)[0])
        cands = set()
        for v in voisins:
            cands.update(set(np.where(self.R[v] > 0)[0]) - poi_visites)

        # Expansion cold start si nécessaire
        if len(cands) < k:
            logger.info(f"[MCMAPRA] Cold start détecté (|Cands|={len(cands)} < k={k}) — expansion ontologique + IoT")
            cands_onto = self.infereur.expansion_candidats(u_id, onto_ctx, N=5*k)
            cands_iot  = self.module_iot.proximite_candidats(u_id, iot_ctx, sigma=self.config_iot["sigma"])
            cands = cands | set(cands_onto) | set(cands_iot)

        cands = list(cands)
        logger.info(f"[MCMAPRA] Pool de candidats : {len(cands)} POI")

        # ── Étape 3 : Exécution des 3 agents en parallèle ─────────────────
        logger.info("[MCMAPRA] Étape 3 — Agents de recommandation (Pearson, Jaccard, Confiance)")
        L1 = self.rec1.classer(u_id, cands)   # Éq. (3) et (4b) — Pearson + lissage
        L2 = self.rec2.classer(u_id, cands)   # Éq. (5) et (6b) — Jaccard TF-IDF
        L3 = self.rec3.classer(u_id, cands)   # Éq. (7) et (8c) — Confiance propagée

        # ── Étape 4 : Conciliation par vote ───────────────────────────────
        logger.info(f"[MCMAPRA] Étape 4 — Conciliation ({self.agent_vote.__class__.__name__})")
        rang_vote = self.agent_vote.fusionner(L1, L2, L3)   # Éq. (9) Borda ou (10-11) Condorcet

        # ── Étape 5 : Scores enrichissants ───────────────────────────────
        logger.info("[MCMAPRA] Étape 5 — Score_geo et Score_onto")
        score_geo  = self.module_iot.calculer(u_id, cands, iot_ctx)    # Éq. (13)
        score_onto = self.infereur.calculer_score(u_id, cands, onto_ctx)  # Éq. (18-20b)

        # Score de vote normalisé (rang → score décroissant)
        n = len(cands)
        score_vote = {poi: (n - rang) / max(n, 1) for poi, rang in rang_vote.items()}

        # ── Étape 6 : Fusion adaptative ───────────────────────────────────
        logger.info(f"[MCMAPRA] Étape 6 — Fusion adaptative (statut={statut_cs}, λ={self.lambda_fusion})")
        score_final = {}
        for poi in cands:
            sv = score_vote.get(poi, 0.0)
            sg = score_geo.get(poi, 0.0)
            so = score_onto.get(poi, 0.0)

            # Éq. (12) adaptée au statut cold start
            if statut_cs == "CS-0":
                score_final[poi] = so                                            # 100 % ontologie
            elif statut_cs == "CS-3":
                score_final[poi] = 0.4 * so + 0.6 * sg                          # mixte onto / géo
            elif statut_cs == "CS-10":
                score_final[poi] = 0.3 * so + 0.7 * sv                          # léger onto
            else:  # warm
                score_final[poi] = self.lambda_fusion * sv + (1 - self.lambda_fusion) * sg  # Éq. (12)

        # Tri décroissant et sélection Top-k
        liste_finale = sorted(score_final.items(), key=lambda x: x[1], reverse=True)[:k]

        elapsed = time.perf_counter() - t0
        logger.info(f"[MCMAPRA] Recommandation terminée en {elapsed*1000:.1f} ms — Top-{k} produit")

        return liste_finale

    # ─────────────────────────────────────────────────────────────────────────
    def evaluer(
        self,
        R_test: np.ndarray,
        k: int = 10,
        seuil_pertinence: float = 3.5,
        protocole_cs: str = "warm",
    ) -> Dict[str, float]:
        """
        Évalue MCMAPRA sur un ensemble de test selon les métriques de l'article.

        Métriques calculées
        -------------------
        - MSE          : Éq. (14)
        - Précision@k  : Éq. (15)
        - Rappel@k     : Éq. (16)
        - F1@k         : Éq. (16b)
        - ILD          : Éq. (17) — diversité intra-liste
        """
        resultats_par_user = []

        for u_id in range(self.R.shape[0]):
            poi_reels = set(np.where(R_test[u_id] >= seuil_pertinence)[0])
            if not poi_reels:
                continue

            reco = self.recommander(u_id, k=k, statut_cs=protocole_cs)
            poi_reco = [p for p, _ in reco]

            resultats_par_user.append(
                calculer_metriques(poi_reco, poi_reels, R_test[u_id], k, seuil_pertinence)
            )

        # Moyennage sur tous les utilisateurs
        if not resultats_par_user:
            return {}
        return {
            cle: float(np.mean([r[cle] for r in resultats_par_user]))
            for cle in resultats_par_user[0]
        }
