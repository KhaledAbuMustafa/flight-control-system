"""
Cessna 182 – Flugzeugparameter

Quelle: Roskam, "Airplane Flight Dynamics and Automatic Flight Controls", Part I,
        Appendix B, Table B1 "Airplane A" (Cessna 182), Spalte CRUISE
        (h = 5000 ft, M = 0.201, U1 = 220.1 ft/s, alpha1 = 0°, CG bei 26.4 % c̄).
Alle Werte in SI-Einheiten, Winkel in rad. Stabilitätsachsen, bei alpha1 = 0 ≈ Körperachsen.
Gleiche Variablennamen wie c172_params.py, damit die Datei austauschbar ist.
"""

import numpy as np

NAME = "Cessna 182 (Roskam, cruise)"

FT_TO_M = 0.3048
FT2_TO_M2 = FT_TO_M**2
LB_TO_KG = 0.453592
SLUGFT2_TO_KGM2 = 1.35582

# Umgebung (wie bei der C172 Meereshöhe, damit der Vergleich fair ist)
rho = 1.225                       # [kg/m³]
g = 9.81                          # [m/s²]

# Masse und Geometrie
m = 2650 * LB_TO_KG               # [kg]         ≈ 1202
S = 174 * FT2_TO_M2               # [m²]         ≈ 16.17
c_bar = 4.9 * FT_TO_M             # [m]          ≈ 1.49
b = 36.0 * FT_TO_M                # [m]          ≈ 10.97
I_y = 1346 * SLUGFT2_TO_KGM2      # [kg·m²]
I_x = 948 * SLUGFT2_TO_KGM2       # [kg·m²]
I_z = 1967 * SLUGFT2_TO_KGM2      # [kg·m²]
I_xz = 0.0                        # [kg·m²] (laut Tabelle 0)

# Schwerpunkt relativ zum Bezugspunkt der Daten (Bezug = 26.4 % c̄), + = hinten
dh_cg = 0.0
x_cg_ref_mac = 0.264              # Bezugsschwerpunkt als Anteil von c̄

# Auftrieb
CL0 = 0.307
CL_alpha = 4.41
CL_delta_e = 0.43
CL_q = 3.9
CL_alpha_dot = 1.7

# Widerstand: Roskam gibt CD_0 = 0.027 und CD_1 = 0.032 bei CL_1 = 0.307 an.
# Umrechnung auf CD = CD0 + k*CL^2 so, dass CD_1 im Reiseflug exakt getroffen wird:
CD0 = 0.027
k = (0.032 - 0.027) / 0.307**2    # ≈ 0.053
CD_delta_e = 0.0

# Nickmoment
Cm0 = 0.04
Cm_alpha = -0.613
Cm_delta_e = -1.122
Cm_q = -12.4
Cm_alpha_dot = -7.27

# Seitenbewegung (gleiche Vorzeichenkonvention wie c172_params.py:
#   +delta_a = rechtes Querruder runter -> Rollen nach links).
# Roskam definiert +delta_a umgekehrt (Cl_delta_a = +0.229, Cn_delta_a = -0.0216),
# deshalb sind beide Querruder-Beiwerte hier im Vorzeichen gedreht.
CY_beta = -0.393
CY_p = -0.075
CY_r = 0.214
CY_delta_a = 0.0
CY_delta_r = 0.187

Cl_beta = -0.0923
Cl_p = -0.484
Cl_r = 0.0798
Cl_delta_a = -0.229
Cl_delta_r = 0.0147

Cn_beta = 0.0587
Cn_p = -0.0278
Cn_r = -0.0937
Cn_delta_a = 0.0216
Cn_delta_r = -0.0645

# Stellgrößen-Grenzen – NICHT aus Roskam (dort nicht angegeben):
# Annahme: gleiche Ruderausschläge wie C172; Schub skaliert mit Leistung (230 PS statt 160 PS).
delta_e_min = np.deg2rad(-28.0)
delta_e_max_6dof = np.deg2rad(23.0)
delta_a_max = np.deg2rad(15.0)
delta_r_max = np.deg2rad(16.0)
delta_e_max = np.deg2rad(25.0)    # nur für Kompatibilität mit den 3-DOF-Dateien
T_min = 0.0
T_max = 2500.0 * 230 / 160        # ≈ 3600 N (grobe Annahme)