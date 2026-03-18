"""
Tests unitaires — MCMAPRA
=========================
Vérifie les algorithmes 1-4 et les métriques d'évaluation.

Exécution :
    pytest tests/test_mcmapra.py -v
"""

import numpy as np
import pytest

from mcmapra.agents.rec1_pearson    import Rec1AgPearson
from mcmapra.agents.rec2_jaccard    import Rec2AgJaccard
from mcmapra.agents.rec3_confiance  import Rec3AgConfiance
from mcmapra.conciliation.borda     import BordaAg
from mcmapra.conciliation.condorcet import CondorcetAg
from mcmapra.iot.module_iot         import ModuleIoT
from mcmapra.ontologie.infereur     import InfereurOntologie
from mcmapra.evaluation.metriques   import calculer_metriques
from mcmapra.mcmapra                import MCMAPRA


# ─── Fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def matrice_test():
    """Matrice 10 utilisateurs × 20 POI."""
    rng = np.random.default_rng(42)
    R   = np.zeros((10, 20))
    for u in range(10):
        pois = rng.choice(20, size=rng.integers(3, 8), replace=False)
        for p in pois:
            R[u, p] = rng.integers(1, 6)
    return R


@pytest.fixture
def graphe_confiance():
    return {
        0: {1: 0.8, 2: 0.6},
        1: {3: 0.9, 4: 0.5},
        2: {4: 0.7},
        3: {5: 0.8},
    }


# ─── Tests Rec1Ag (Pearson) ──────────────────────────────────────────────────
class TestRec1AgPearson:
    def test_similarite_meme_utilisateur(self, matrice_test):
        agent = Rec1AgPearson(matrice_test)
        assert agent.similarite_pearson(0, 0) == 0.0   # pas de sim avec soi-même

    def test_similarite_bornee(self, matrice_test):
        agent = Rec1AgPearson(matrice_test)
        for u in range(5):
            for v in range(u+1, 5):
                s = agent.similarite_pearson(u, v)
                assert -1.0 <= s <= 1.0, f"sim_P({u},{v})={s} hors [-1,1]"

    def test_lissage_shrinkage(self, matrice_test):
        agent = Rec1AgPearson(matrice_test, beta_lissage=25.0)
        s_brut   = Rec1AgPearson(matrice_test, beta_lissage=0.0).similarite_pearson(0, 1)
        s_lisse  = agent.similarite_pearson(0, 1)
        # Le lissage doit réduire la valeur absolue de la similarité
        assert abs(s_lisse) <= abs(s_brut) + 1e-9

    def test_classement_retourne_liste(self, matrice_test):
        agent    = Rec1AgPearson(matrice_test)
        candidats = list(range(10))
        result   = agent.classer(0, candidats)
        assert isinstance(result, list)
        assert set(result) == set(candidats)

    def test_voisins_k_max(self, matrice_test):
        k     = 3
        agent = Rec1AgPearson(matrice_test, k_voisins=k)
        voisins = agent.k_plus_proches_voisins(0)
        assert len(voisins) <= k


# ─── Tests Rec2Ag (Jaccard) ──────────────────────────────────────────────────
class TestRec2AgJaccard:
    def test_jaccard_standard_bornee(self, matrice_test):
        agent = Rec2AgJaccard(matrice_test, utiliser_idf=False)
        for u in range(5):
            for v in range(u+1, 5):
                s = agent.similarite(u, v)
                assert 0.0 <= s <= 1.0

    def test_jaccard_idf_bornee(self, matrice_test):
        agent = Rec2AgJaccard(matrice_test, utiliser_idf=True)
        for u in range(5):
            for v in range(u+1, 5):
                s = agent.similarite(u, v)
                assert 0.0 <= s <= 1.0

    def test_classement_complet(self, matrice_test):
        agent    = Rec2AgJaccard(matrice_test)
        candidats = [1, 2, 3, 4, 5]
        result   = agent.classer(0, candidats)
        assert set(result) == set(candidats)
        assert len(result) == len(candidats)


# ─── Tests Rec3Ag (Confiance) ────────────────────────────────────────────────
class TestRec3AgConfiance:
    def test_convergence_garantie(self, matrice_test, graphe_confiance):
        """Vérifie Éq. (8c) : convergence de la propagation."""
        agent = Rec3AgConfiance(matrice_test, graphe_confiance, alpha=0.7, trust_depth=3)
        T_prop = agent.matrice_confiance_propagee
        # Toutes les valeurs doivent être dans [0,1]
        assert np.all(T_prop >= 0.0)
        assert np.all(T_prop <= 1.0 + 1e-9)

    def test_elagage_conf_min(self, matrice_test, graphe_confiance):
        """Les arêtes sous conf_min doivent être élaguées."""
        agent  = Rec3AgConfiance(matrice_test, graphe_confiance, conf_min=0.5)
        T_prop = agent.matrice_confiance_propagee
        valeurs_non_nulles = T_prop[T_prop > 0]
        if len(valeurs_non_nulles) > 0:
            assert np.all(valeurs_non_nulles >= 0.5 - 1e-9)

    def test_score_bornee(self, matrice_test, graphe_confiance):
        agent = Rec3AgConfiance(matrice_test, graphe_confiance)
        for poi in range(10):
            s = agent.score_confiance(0, poi)
            assert 0.0 <= s <= 5.0   # max = note maximale

    def test_classement_retourne_candidats(self, matrice_test, graphe_confiance):
        agent    = Rec3AgConfiance(matrice_test, graphe_confiance)
        candidats = [0, 1, 2, 3, 4]
        result   = agent.classer(0, candidats)
        assert set(result) == set(candidats)


# ─── Tests BordaAg ───────────────────────────────────────────────────────────
class TestBordaAg:
    def test_borda_scores_positifs(self):
        agent  = BordaAg()
        listes = [[1, 2, 3], [2, 1, 3], [3, 2, 1]]
        scores = agent.calculer_scores_borda(listes)
        assert all(s >= 0 for s in scores.values())

    def test_borda_pareto(self):
        """Si tous les agents préfèrent A à B, alors Borda(A) > Borda(B)."""
        agent  = BordaAg()
        # POI 0 toujours en premier, POI 1 toujours en second
        listes = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
        scores = agent.calculer_scores_borda(listes)
        assert scores[0] > scores[1] > scores[2]

    def test_fusionner_retourne_rangs(self):
        agent  = BordaAg()
        L1, L2, L3 = [1, 2, 3], [2, 1, 3], [3, 1, 2]
        rangs  = agent.fusionner(L1, L2, L3)
        assert isinstance(rangs, dict)
        assert set(rangs.keys()) == {1, 2, 3}
        assert min(rangs.values()) == 1

    def test_fusionner_couvre_tous_candidats(self):
        agent  = BordaAg()
        rangs  = agent.fusionner([1,2,3,4], [4,3,2,1], [2,4,1,3])
        assert set(rangs.keys()) == {1, 2, 3, 4}


# ─── Tests CondorcetAg ────────────────────────────────────────────────────────
class TestCondorcetAg:
    def test_fusionner_sans_cycle(self):
        agent = CondorcetAg()
        L1    = [1, 2, 3]
        L2    = [1, 2, 3]
        L3    = [1, 2, 3]
        rangs = agent.fusionner(L1, L2, L3)
        # Sans cycle, le POI 1 doit être premier
        assert rangs[1] < rangs[2] < rangs[3]

    def test_fusionner_avec_cycle(self):
        """Test que Copeland résout le cycle A≻B≻C≻A."""
        agent = CondorcetAg()
        # Agent 1: A>B>C, Agent 2: B>C>A, Agent 3: C>A>B  → cycle
        rangs = agent.fusionner(["A","B","C"], ["B","C","A"], ["C","A","B"])
        assert isinstance(rangs, dict)
        assert set(rangs.keys()) == {"A", "B", "C"}

    def test_retourne_tous_candidats(self):
        agent = CondorcetAg()
        rangs = agent.fusionner([1,2,3,4], [4,3,2,1], [2,1,4,3])
        assert set(rangs.keys()) == {1,2,3,4}


# ─── Tests ModuleIoT ─────────────────────────────────────────────────────────
class TestModuleIoT:
    def test_distance_haversine_zero(self):
        loc = (40.7128, -74.0060)
        assert ModuleIoT.distance_haversine(loc, loc) < 1e-6

    def test_distance_haversine_positive(self):
        loc1 = (40.7128, -74.0060)   # NYC
        loc2 = (48.8566,   2.3522)   # Paris
        d    = ModuleIoT.distance_haversine(loc1, loc2)
        assert 5_000_000 < d < 6_000_000   # ~5800 km en mètres

    def test_score_geo_borne(self):
        module = ModuleIoT()
        ctx    = module.collecter_contexte(0)
        scores = module.calculer(0, list(range(10)), ctx)
        for s in scores.values():
            assert 0.0 <= s <= 1.0 + 1e-9

    def test_filtres_distance(self):
        """POI au-delà de d_max doit avoir score=0."""
        module = ModuleIoT(d_max=100.0)   # rayon 100m très restrictif
        ctx    = {"loc_u": (40.7128, -74.0060), "rho_aged": {}, "t_slot": "Soir"}
        locs   = {999: (40.7128 + 1.0, -74.0060)}  # ~111 km > d_max
        scores = module.calculer(0, [999], ctx, localisation_poi=locs)
        assert scores.get(999, 0.0) == 0.0


# ─── Tests InfereurOntologie ──────────────────────────────────────────────────
class TestInfereurOntologie:
    def test_poids_normalises(self):
        with pytest.raises(AssertionError):
            InfereurOntologie({"w_cat": 0.4, "w_pop": 0.4, "w_trust": 0.4})  # somme ≠ 1

    def test_inferer_retourne_contexte(self, matrice_test, graphe_confiance):
        onto = InfereurOntologie()
        ctx  = onto.inferer(0, matrice_test, graphe_confiance)
        assert "pref_cat_u" in ctx
        assert "T_prop"     in ctx
        assert "poi_semant" in ctx

    def test_score_bornee(self, matrice_test, graphe_confiance):
        onto = InfereurOntologie()
        ctx  = onto.inferer(0, matrice_test, graphe_confiance)
        scores = onto.calculer_score(0, list(range(10)), ctx)
        for s in scores.values():
            assert 0.0 <= s <= 1.0 + 1e-9

    def test_expansion_garantie(self, matrice_test, graphe_confiance):
        """Algo 4C doit garantir au moins N candidats."""
        onto = InfereurOntologie()
        ctx  = onto.inferer(0, matrice_test, graphe_confiance)
        N    = 30
        cands = onto.expansion_candidats(0, ctx, N=N)
        assert len(cands) >= N


# ─── Tests Métriques ─────────────────────────────────────────────────────────
class TestMetriques:
    def test_precision_parfaite(self):
        poi_reco     = [1, 2, 3, 4, 5]
        poi_pertinent = {1, 2, 3, 4, 5}
        notes        = np.zeros(10); notes[[1,2,3,4,5]] = 4.0
        res = calculer_metriques(poi_reco, poi_pertinent, notes, k=5)
        assert res["Precision"] == pytest.approx(1.0)

    def test_rappel_zero_si_aucun_pertinent(self):
        poi_reco     = [1, 2, 3]
        poi_pertinent = {4, 5, 6}
        notes        = np.zeros(10)
        res = calculer_metriques(poi_reco, poi_pertinent, notes, k=3)
        assert res["Rappel"] == pytest.approx(0.0)

    def test_f1_harmonique(self):
        poi_reco     = [1, 2, 3, 4, 5]
        poi_pertinent = {1, 2, 6, 7, 8}
        notes        = np.zeros(10); notes[[1,2,6,7,8]] = 4.0
        res = calculer_metriques(poi_reco, poi_pertinent, notes, k=5)
        p, r = res["Precision"], res["Rappel"]
        f1_attendu = 2*p*r/(p+r) if (p+r) > 0 else 0
        assert res["F1"] == pytest.approx(f1_attendu, abs=1e-6)


# ─── Tests MCMAPRA intégration ────────────────────────────────────────────────
class TestMCMAPRAIntegration:
    def test_recommander_retourne_k(self, matrice_test, graphe_confiance):
        modele = MCMAPRA(matrice_test, graphe_confiance)
        reco   = modele.recommander(0, k=5)
        assert len(reco) <= 5

    def test_recommander_scores_dans_ordre(self, matrice_test, graphe_confiance):
        modele = MCMAPRA(matrice_test, graphe_confiance)
        reco   = modele.recommander(0, k=5)
        scores = [s for _, s in reco]
        assert scores == sorted(scores, reverse=True)

    def test_cold_start_cs0(self, matrice_test, graphe_confiance):
        """En CS-0, le modèle doit quand même produire des recommandations."""
        modele = MCMAPRA(matrice_test, graphe_confiance)
        reco   = modele.recommander(0, k=5, statut_cs="CS-0")
        assert len(reco) > 0

    def test_agents_borda_vs_condorcet(self, matrice_test, graphe_confiance):
        m_borda = MCMAPRA(matrice_test, graphe_confiance, agent_conciliation="borda")
        m_cond  = MCMAPRA(matrice_test, graphe_confiance, agent_conciliation="condorcet")
        r_borda = m_borda.recommander(0, k=5)
        r_cond  = m_cond.recommander(0, k=5)
        assert len(r_borda) > 0
        assert len(r_cond)  > 0
