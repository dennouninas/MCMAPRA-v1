"""
ModuleIoT — Algorithmes 3A, 3B, 3C : Géolocalisation IoT Temps Réel
======================================================================
Implémente les Équations (13), (13b), (13c), (21) et (21b) de l'article MCMAPRA.

  Éq. (13)  Score_geo = exp(−d²/2σ²) × Pop_IoT(p,t) × w_cat(cat_p, u)
  Éq. (13b) d_Hav = 2·R_T·arcsin(sqrt(sin²(Δlat/2) + cos(lat_u)·cos(lat_p)·sin²(Δlon/2)))
  Éq. (13c) filtres : d_min=50m, d_max=5km
  Éq. (21)  Pop_IoT = w_real·ρ_live(p,t) + (1−w_real)·ρ_hist(p,t_slot)
  Éq. (21b) ρ_live_aged = ρ_live(p,t₀)·τ_decay^(Δt/T_fenêtre)
"""

from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Constantes géographiques ─────────────────────────────────────────────────
R_TERRE_KM = 6371.0   # rayon moyen de la Terre [km]

# Créneaux horaires contextuels (t_slot) — 6 créneaux de 4h
CRENEAUX = {
    "Nuit_matin": (0, 4),
    "Matin":      (4, 8),
    "Midi":       (8, 12),
    "Apres_midi": (12, 16),
    "Soir":       (16, 20),
    "Nuit_soir":  (20, 24),
}


class ModuleIoT:
    """
    Module de géolocalisation IoT pour MCMAPRA.

    Implémente les 3 sous-algorithmes de l'article :
      - Algo 3A : CollecterContexteIoT(u, t)
      - Algo 3B : ModuleIoT.Calculer(u, Cands, IOT_ctx)
      - Algo 3C : FusionFinale (intégrée dans MCMAPRA principal)

    Paramètres (Tableau 3 de l'article)
    ------------------------------------
    sigma       : rayon d'influence gaussien [m] (défaut 500m — optimal)
    d_min       : distance minimale filtre [m]   (défaut 50m)
    d_max       : distance maximale filtre [m]   (défaut 5000m)
    tau_decay   : taux de décroissance temporelle (défaut 0.85)
    w_realtime  : poids du signal en direct [0,1] (défaut 0.60)
    rho_min     : densité minimale (défaut 0.10)
    Delta_t     : fenêtre temporelle [min]        (défaut 30)
    theta_outlier_sigma : multiplicateur z-score pour outliers (défaut 3.0)
    """

    def __init__(
        self,
        sigma: float = 500.0,
        d_min: float = 50.0,
        d_max: float = 5000.0,
        tau_decay: float = 0.85,
        w_realtime: float = 0.60,
        rho_min: float = 0.10,
        Delta_t: int = 30,
        theta_outlier_sigma: float = 3.0,
    ):
        self.sigma              = sigma
        self.d_min              = d_min
        self.d_max              = d_max
        self.tau_decay          = tau_decay
        self.w_realtime         = w_realtime
        self.rho_min            = rho_min
        self.Delta_t            = Delta_t
        self.theta_outlier_sigma = theta_outlier_sigma

        # Base de données de popularité historique simulée
        # Format : {poi_id: {t_slot: rho_historique}}
        self._historique_pop: Dict[int, Dict[str, float]] = {}

    # ─── Algo 3A : Collecte et fusion multi-capteurs ──────────────────────────
    def collecter_contexte(
        self,
        u_id: int,
        contexte_externe: Optional[Dict] = None,
    ) -> Dict:
        """
        Algorithme 3A — CollecterContexteIoT(u, t)

        En production : fusionne GPS, BLE beacons et WiFi RSSI par filtre de Kalman.
        En simulation  : utilise les données du contexte_externe.

        Paramètres
        ----------
        u_id              : identifiant utilisateur
        contexte_externe  : dict optionnel contenant {lat, lon, t, rho_live}

        Retourne
        --------
        iot_ctx = {loc_u: (lat, lon), rho_aged: dict, t_slot: str, t: datetime}
        """
        now = datetime.now()

        if contexte_externe is not None:
            lat  = contexte_externe.get("lat", 40.7128)
            lon  = contexte_externe.get("lon", -74.0060)
            rho_live_data = contexte_externe.get("rho_live", {})
        else:
            # Simulation : position par défaut (NYC)
            lat, lon = 40.7128, -74.0060
            rho_live_data = {}

        # Validation outlier — seuil theta = μ + 3σ (z-score)
        # En production : comparaison avec historique de position
        # Ici : accepter directement (pas d'historique disponible)
        loc_u = (lat, lon)

        # Créneaux horaire contextuel (t_slot)
        heure = now.hour
        t_slot = self._determiner_creneau(heure)

        # Popularité temps réel avec décroissance temporelle — Éq. (21b)
        rho_aged = {}
        for poi_id, rho_val in rho_live_data.items():
            elapsed_min = contexte_externe.get("elapsed_min", 0)
            rho_a = rho_val * (self.tau_decay ** (elapsed_min / self.Delta_t))
            rho_aged[poi_id] = max(0.0, rho_a) if rho_a >= self.rho_min else 0.0

        return {
            "loc_u":    loc_u,
            "rho_aged": rho_aged,
            "t_slot":   t_slot,
            "t":        now,
        }

    # ─── Éq. (13b) : Distance de Haversine ───────────────────────────────────
    @staticmethod
    def distance_haversine(
        loc1: Tuple[float, float],
        loc2: Tuple[float, float],
    ) -> float:
        """
        d_Hav = 2·R_T·arcsin(√(sin²(Δlat/2) + cos(lat_u)·cos(lat_p)·sin²(Δlon/2)))
        — Éq. (13b)

        Paramètres
        ----------
        loc1, loc2 : (lat [°], lon [°]) en degrés décimaux WGS84

        Retourne
        --------
        distance en mètres
        """
        lat1, lon1 = math.radians(loc1[0]), math.radians(loc1[1])
        lat2, lon2 = math.radians(loc2[0]), math.radians(loc2[1])

        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1

        a = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))
        return R_TERRE_KM * c * 1000   # en mètres

    # ─── Éq. (21) : Popularité IoT temps réel ────────────────────────────────
    def pop_iot(
        self,
        poi_id: int,
        rho_aged: Dict[int, float],
        t_slot: str,
    ) -> float:
        """
        Pop_IoT(p,t) = w_real·ρ_live(p,t) + (1−w_real)·ρ_hist(p,t_slot) — Éq. (21)
        """
        rho_live = rho_aged.get(poi_id, 0.0)
        rho_hist = self._historique_pop.get(poi_id, {}).get(t_slot, 0.1)
        return self.w_realtime * rho_live + (1 - self.w_realtime) * rho_hist

    # ─── Algo 3B : Score géographique ────────────────────────────────────────
    def calculer(
        self,
        u_id: int,
        candidats: List[int],
        iot_ctx: Dict,
        localisation_poi: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> Dict[int, float]:
        """
        Algorithme 3B — ModuleIoT.Calculer(u, Cands, IOT_ctx)

        Calcule Score_geo[p] pour chaque POI candidat — Éq. (13).

        Paramètres
        ----------
        u_id             : identifiant utilisateur
        candidats        : liste de poi_id
        iot_ctx          : contexte IoT produit par Algo 3A
        localisation_poi : dict {poi_id: (lat, lon)} — positions des POI

        Retourne
        --------
        dict {poi_id: score_geo ∈ [0,1]}
        """
        loc_u     = iot_ctx.get("loc_u",    (40.7128, -74.0060))
        rho_aged  = iot_ctx.get("rho_aged", {})
        t_slot    = iot_ctx.get("t_slot",   "Soir")

        scores = {}

        for poi in candidats:
            # Localisation du POI (simulée si non fournie)
            if localisation_poi and poi in localisation_poi:
                loc_p = localisation_poi[poi]
            else:
                # Simulation : position légèrement décalée du user
                loc_p = (loc_u[0] + 0.002 * (poi % 10 - 5), loc_u[1] + 0.002 * (poi % 7 - 3))

            # a) Distance Haversine — Éq. (13b)
            d = self.distance_haversine(loc_u, loc_p)

            # b) Filtres de distance — Éq. (13c)
            if d < self.d_min or d > self.d_max:
                scores[poi] = 0.0
                continue

            # c) Décroissance spatiale gaussienne — Éq. (13)
            f_decay = math.exp(-(d ** 2) / (2 * self.sigma ** 2))

            # d) Popularité temps réel — Éq. (21)
            pop = self.pop_iot(poi, rho_aged, t_slot)

            # e) Pondération catégorielle (w_cat = 1.0 si non disponible)
            w_cat_t = 1.0

            # f) Score géographique final — Éq. (13)
            scores[poi] = f_decay * pop * w_cat_t

        return scores

    # ─── Candidats par proximité (Algo 3A — fallback cold start) ─────────────
    def proximite_candidats(
        self,
        u_id: int,
        iot_ctx: Dict,
        sigma: float = 500.0,
        n_max: int = 50,
    ) -> List[int]:
        """
        Retourne une liste de poi_id proches géographiquement.
        Utilisé dans l'expansion cold start (Algo 1, Étape 2).
        Simulation : retourne des indices fictifs.
        """
        # En production : requête sur une base de données géolocalisée (PostGIS, etc.)
        # Ici : simulation avec des indices quelconques
        return list(range(min(n_max, 1000)))

    # ─── Utilitaires ─────────────────────────────────────────────────────────
    def _determiner_creneau(self, heure: int) -> str:
        """Retourne le nom du créneau horaire pour une heure donnée."""
        for nom, (debut, fin) in CRENEAUX.items():
            if debut <= heure < fin:
                return nom
        return "Nuit_soir"

    def mettre_a_jour_historique(
        self,
        poi_id: int,
        t_slot: str,
        rho: float,
    ):
        """Met à jour l'historique de popularité d'un POI pour un créneau donné."""
        if poi_id not in self._historique_pop:
            self._historique_pop[poi_id] = {}
        self._historique_pop[poi_id][t_slot] = max(0.0, min(1.0, rho))
