"""
Cessna 172 – Flugzeugparameter (Längsbewegung)

Quelle: UIUC Cessna-172-Modell (FlightGear), basierend auf Roskam.
Alle Werte in SI-Einheiten, Winkel in rad.
Für ein anderes Flugzeug nur diese Datei austauschen.
"""

import numpy as np

# ============================================================
# Umrechnungsfaktoren (Originaldaten in imperialen Einheiten)
# ============================================================
FT_TO_M = 0.3048                  # Fuß -> Meter
FT2_TO_M2 = FT_TO_M**2            # Quadratfuß -> Quadratmeter
LB_TO_KG = 0.453592               # Pfund (Masse) -> Kilogramm
SLUGFT2_TO_KGM2 = 1.35582         # slug·ft² -> kg·m²

# ============================================================
# Umgebung
# ============================================================
rho = 1.225                       # Luftdichte, Meereshöhe ISA [kg/m³]
g = 9.81                          # Erdbeschleunigung [m/s²]

# ============================================================
# Masse und Geometrie
# ============================================================
m = 2300 * LB_TO_KG               # Masse (MTOM) [kg]            ≈ 1043
S = 174 * FT2_TO_M2               # Flügelfläche [m²]            ≈ 16.17
c_bar = 4.9 * FT_TO_M             # mittlere Flügeltiefe [m]     ≈ 1.49
b = 35.8 * FT_TO_M                # Spannweite [m]               ≈ 10.91
I_y = 1346 * SLUGFT2_TO_KGM2      # Nickträgheitsmoment [kg·m²]  ≈ 1825

# ============================================================
# Auftrieb  CL = CL0 + CL_alpha*α + CL_delta_e*δe + (c̄/2V)*(CL_q*q + CL_alpha_dot*α̇)
# ============================================================
CL0 = 0.31                        # Auftrieb bei α = 0 [-]
CL_alpha = 5.143                  # Auftriebsanstieg [1/rad]
CL_delta_e = 0.43                 # Auftrieb durch Höhenruder [1/rad]
CL_q = 3.9                        # Auftrieb durch Nickrate [1/rad]
CL_alpha_dot = 1.7                # Auftrieb durch α-Änderung [1/rad]

# ============================================================
# Widerstand  CD = CD0 + k*CL²
# ============================================================
CD0 = 0.031                       # Nullwiderstand [-]
k = 0.054                         # Faktor induzierter Widerstand [-]

# ============================================================
# Nickmoment  Cm = Cm0 + Cm_alpha*α + Cm_delta_e*δe + (c̄/2V)*(Cm_q*q + Cm_alpha_dot*α̇)
# ============================================================
Cm0 = -0.015                      # Nickmoment bei α = 0 [-]
Cm_alpha = -0.89                  # statische Längsstabilität [1/rad]
Cm_delta_e = -1.28                # Höhenruderwirksamkeit [1/rad]
Cm_q = -12.4                      # Nickdämpfung [1/rad]
Cm_alpha_dot = -5.2               # Dämpfung durch α-Änderung [1/rad]

# ============================================================
# Stellgrößen-Grenzen
# ============================================================
delta_e_max = np.deg2rad(25.0)    # Höhenruder-Anschlag ± [rad] (real ca. +28°/-23°)
T_min = 0.0                       # minimaler Schub [N]
T_max = 2500.0                    # maximaler Schub [N] (Annahme, 160 PS, grob)