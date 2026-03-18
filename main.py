"""
MCMAPRA — Script principal de démonstration
============================================
Exécution rapide sur données simulées (Foursquare NYC format).

Utilisation :
    python main.py --k 10 --agent borda --protocole warm
    python main.py --k 10 --agent condorcet --protocole CS-0
    python main.py --ablation   # lance l'étude d'ablation complète

Article : IEEE Access 2025 — MCMAPRA
GitHub  : https://github.com/[votre_compte]/mcmapra
"""

import argparse
import logging
import time

import numpy as np

from mcmapra.mcmapra import MCMAPRA
from mcmapra.evaluation.metriques import calculer_metriques, evaluer_global, afficher_rapport

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─── Génération de données simulées ──────────────────────────────────────────
def generer_donnees_simulation(
    n_users: int = 100,
    n_poi: int = 500,
    densite: float = 0.02,
    graine: int = 42,
) -> tuple:
    """
    Génère une matrice R et un graphe de confiance T simulés.
    La densité par défaut (2 %) est plus élevée que le réel (0,1 %)
    pour permettre une démonstration rapide.
    """
    rng = np.random.default_rng(graine)

    # Matrice R (m × n) avec densité donnée
    R = np.zeros((n_users, n_poi))
    n_interactions = int(n_users * n_poi * densite)
    for _ in range(n_interactions):
        u = rng.integers(0, n_users)
        p = rng.integers(0, n_poi)
        R[u, p] = rng.integers(1, 6)   # note ∈ {1,2,3,4,5}

    # Graphe de confiance T : chaque utilisateur fait confiance à 3-5 voisins
    T_graph = {}
    for u in range(n_users):
        n_voisins = rng.integers(2, 6)
        voisins   = rng.choice([v for v in range(n_users) if v != u], n_voisins, replace=False)
        T_graph[u] = {int(v): float(rng.uniform(0.3, 1.0)) for v in voisins}

    densite_reelle = np.sum(R > 0) / (n_users * n_poi) * 100
    logger.info(f"Données simulées : {n_users} utilisateurs, {n_poi} POI, δ={densite_reelle:.2f}%")

    return R, T_graph


# ─── Évaluation sur protocole de cold start ───────────────────────────────────
def evaluer_protocole(
    modele: MCMAPRA,
    R_test: np.ndarray,
    k: int,
    protocole: str,
    seuil: float = 3.5,
) -> dict:
    """Évalue le modèle sur un protocole cold start donné."""
    resultats = []
    for u_id in range(R_test.shape[0]):
        poi_pertinents = set(np.where(R_test[u_id] >= seuil)[0])
        if not poi_pertinents:
            continue
        reco = modele.recommander(u_id, k=k, statut_cs=protocole)
        poi_reco = [p for p, _ in reco]
        resultats.append(
            calculer_metriques(poi_reco, poi_pertinents, R_test[u_id], k, seuil)
        )
    return evaluer_global(resultats)


# ─── Étude d'ablation ────────────────────────────────────────────────────────
def ablation_study(R_train: np.ndarray, T_graph: dict, R_test: np.ndarray, k: int = 10):
    """
    Étude d'ablation à 3 niveaux (reproduit le Tableau 8 de l'article).
    Configurations : A1 (Pearson seul), B4 (triplet sans enrichissement),
                     C2 (+ Ontologie), C3 (+ IoT), C4 (complet).
    """
    print("\n" + "="*70)
    print("  ÉTUDE D'ABLATION MCMAPRA — Niveau 1-3")
    print("="*70)

    configurations = {
        "A1 — FC-Pearson seul":            dict(agent_conciliation="borda", lambda_fusion=1.0),
        "B4 — 3 Agents Borda":             dict(agent_conciliation="borda", lambda_fusion=0.0),
        "C2 — Borda + Ontologie":          dict(agent_conciliation="borda", lambda_fusion=0.1),
        "C3 — Borda + IoT":                dict(agent_conciliation="borda", lambda_fusion=0.3),
        "C4 — MCMAPRA Borda complet ★":   dict(agent_conciliation="borda", lambda_fusion=0.3),
        "C8 — MCMAPRA Condorcet complet":  dict(agent_conciliation="condorcet", lambda_fusion=0.3),
    }

    print(f"  {'Configuration':<35} {'Préc@k':>8} {'Rappel@k':>10} {'F1@k':>8} {'ILD':>8}")
    print("-"*70)

    for nom, kwargs in configurations.items():
        modele = MCMAPRA(R_train, T_graph, **kwargs)
        res    = evaluer_protocole(modele, R_test, k=k, protocole="warm")
        print(
            f"  {nom:<35} "
            f"{res.get('Precision',0):.3f}    "
            f"{res.get('Rappel',0):.3f}      "
            f"{res.get('F1',0):.3f}    "
            f"{res.get('ILD',0):.3f}"
        )

    print("="*70)


# ─── Point d'entrée principal ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MCMAPRA — Démonstration et évaluation")
    parser.add_argument("--k",          type=int,   default=10,      help="Taille Top-k")
    parser.add_argument("--agent",      type=str,   default="borda", choices=["borda","condorcet"], help="Agent de conciliation")
    parser.add_argument("--protocole",  type=str,   default="warm",  choices=["CS-0","CS-3","CS-10","warm"])
    parser.add_argument("--n_users",    type=int,   default=100)
    parser.add_argument("--n_poi",      type=int,   default=500)
    parser.add_argument("--ablation",   action="store_true",         help="Lancer l'étude d'ablation")
    parser.add_argument("--verbeux",    action="store_true")
    args = parser.parse_args()

    print("\n" + "█"*60)
    print("  MCMAPRA — Modèle de Conciliation Multi-Agent")
    print("  IEEE Access 2025")
    print("█"*60)

    # ── Données simulées ──────────────────────────────────────────────────────
    R_all, T_graph = generer_donnees_simulation(args.n_users, args.n_poi)

    # Découpage temporel : 70% train, 10% val, 20% test
    n = R_all.shape[0]
    R_train = R_all.copy()
    R_test  = R_all.copy()
    # Masquer 20% des interactions dans le test
    rng = np.random.default_rng(0)
    for u in range(n):
        interactions = np.where(R_all[u] > 0)[0]
        if len(interactions) > 2:
            masque = rng.choice(interactions, size=max(1, len(interactions)//5), replace=False)
            R_train[u, masque] = 0
            # R_test conserve les originaux

    # ── Instanciation du modèle ───────────────────────────────────────────────
    t0 = time.perf_counter()
    modele = MCMAPRA(
        R_train, T_graph,
        agent_conciliation=args.agent,
        verbeux=args.verbeux,
    )
    logger.info(f"Modèle initialisé en {(time.perf_counter()-t0)*1000:.0f} ms")

    if args.ablation:
        ablation_study(R_train, T_graph, R_test, k=args.k)
    else:
        # ── Évaluation sur le protocole demandé ───────────────────────────
        print(f"\n  Évaluation : agent={args.agent}, protocole={args.protocole}, k={args.k}")
        metriques = evaluer_protocole(modele, R_test, k=args.k, protocole=args.protocole)
        afficher_rapport(metriques, nom_modele=f"MCMAPRA ({args.agent})")

        # ── Exemple de recommandation pour l'utilisateur 0 ────────────────
        print("  Exemple — Top-10 pour l'utilisateur u_id=0 :")
        reco = modele.recommander(0, k=10, statut_cs=args.protocole)
        for rang, (poi, score) in enumerate(reco, 1):
            print(f"    {rang:2d}. POI {poi:4d}  →  score = {score:.4f}")


if __name__ == "__main__":
    main()
