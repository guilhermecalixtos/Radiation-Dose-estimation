
"""
Estimador de dose ionizante

Base:
 - ICRU Report 85a (2011): fluencia (Eq. 3.1.3), fluencia de energia (Sec. 3.1.3),
   kerma (Eq. 5.1.1), kerma de colisao (Eq. 5.1.3), dose absorvida (Sec. 5.2.5),
   coef. de atenuacao (Sec. 4.2).
 - NIST XCOM / Hubbell & Seltzer (chap2, chap3, ElemTab/z14): mu/rho, mu_en/rho (Si).
 - De Paoli et al., RADECS 2023: feixe gaussiano + lei de Lambert + Si.
"""
import math
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1) Constantes e conversoes
# ----------------------------------------------------------------------
EV_TO_J = 1.602176487e-19          # ICRU 85, Tabela 2.2
GY_TO_KRAD = 0.1                   # 1 Gy = 100 rad = 0.1 krad
FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))   # 1/2.3548
SENSOR_MATERIAL = "Si"  # Opções: "Si", "GaAs" ou "CdTe"
SENSOR_DENSITIES_G_CM3 = {"Si": 2.33,"GaAs": 5.3176,"CdTe": 5.85} #g/cm3

# ----------------------------------------------------------------------
# 2) Dados NIST (Si, Z=14; O, Z=8) - energia [MeV], coeficientes [cm2/g]
# Fonte: physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/
# ----------------------------------------------------------------------
NIST_SI = {
    "density_g_cm3": 2.33,
    "E_MeV":  [2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 8e-3, 1e-2, 1.5e-2, 2e-2, 3e-2, 4e-2,
               5e-2, 6e-2, 8e-2, 1e-1, 1.5e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1,
               8e-1, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
    "mu_rho": [2.777e3, 9.784e2, 4.529e2, 2.450e2, 1.470e2, 6.468e1, 3.389e1, 1.034e1,
               4.464, 1.436, 7.012e-1, 4.385e-1, 3.207e-1, 2.228e-1, 1.835e-1,
               1.448e-1, 1.275e-1, 1.082e-1, 9.614e-2, 8.748e-2, 8.077e-2,
               7.082e-2, 6.361e-2, 5.688e-2, 5.183e-2, 4.480e-2, 3.678e-2,
               3.240e-2, 2.967e-2, 2.788e-2, 2.574e-2, 2.462e-2],
    "muen_rho": [2.669e3, 9.516e2, 4.427e2, 2.400e2, 1.439e2, 6.313e1, 3.289e1, 9.794,
                 4.076, 1.164, 4.782e-1, 2.430e-1, 1.434e-1, 6.896e-2, 4.513e-2,
                 3.086e-2, 2.905e-2, 2.932e-2, 2.968e-2, 2.971e-2, 2.951e-2,
                 2.875e-2, 2.778e-2, 2.652e-2, 2.535e-2, 2.345e-2, 2.101e-2,
                 1.963e-2, 1.878e-2, 1.827e-2, 1.773e-2, 1.753e-2],
}
#Dados NIST Fonte: physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/
NIST_Cd = {
    "E_MeV": [
        0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.060, 0.080, 0.100,
        0.150, 0.200, 0.300, 0.400, 0.500, 0.600, 0.800, 1.000, 1.250,
        1.500, 2.000, 3.000, 4.000, 5.000, 6.000, 8.000, 10.000,
    ],
    "mu_rho": [
        1.244e2, 4.178e1, 1.920e1, 3.765e1, 1.778e1, 9.779, 5.975,
        2.751, 1.524, 5.593e-1, 3.038e-1, 1.571e-1, 1.129e-1, 9.250e-2,
        8.064e-2, 6.670e-2, 5.826e-2, 5.129e-2, 4.673e-2, 4.139e-2,
        3.698e-2, 3.559e-2, 3.536e-2, 3.563e-2, 3.691e-2, 3.853e-2,
    ],
    "muen_rho": [
        1.197e2, 3.952e1, 1.776e1, 1.594e1, 9.812, 6.115, 4.001, 1.957,
        1.096, 3.753e-1, 1.813e-1, 7.580e-2, 4.868e-2, 3.838e-2,
        3.339e-2, 2.854e-2, 2.597e-2, 2.387e-2, 2.247e-2, 2.099e-2,
        2.051e-2, 2.114e-2, 2.206e-2, 2.300e-2, 2.477e-2, 2.625e-2,
    ],
}
#Dados NIST Fonte: physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/

NIST_Te = {
    "E_MeV": [
        0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.060, 0.080, 0.100,
        0.150, 0.200, 0.300, 0.400, 0.500, 0.600, 0.800, 1.000, 1.250,
        1.500, 2.000, 3.000, 4.000, 5.000, 6.000, 8.000, 10.000,
    ],
    "mu_rho": [
        1.501e2, 5.078e1, 2.341e1, 7.878, 2.064e1, 1.145e1, 7.041,
        3.255, 1.801, 6.492e-1, 3.429e-1, 1.679e-1, 1.163e-1, 9.328e-2,
        8.022e-2, 6.538e-2, 5.669e-2, 4.967e-2, 4.518e-2, 4.009e-2,
        3.606e-2, 3.494e-2, 3.492e-2, 3.534e-2, 3.683e-2, 3.860e-2,
    ],
    "muen_rho": [
        1.433e2, 4.802e1, 2.175e1, 6.997, 9.480, 6.330, 4.306, 2.195,
        1.256, 4.409e-1, 2.132e-1, 8.646e-2, 5.326e-2, 4.055e-2,
        3.440e-2, 2.854e-2, 2.559e-2, 2.327e-2, 2.179e-2, 2.033e-2,
        1.998e-2, 2.075e-2, 2.177e-2, 2.279e-2, 2.466e-2, 2.620e-2,
    ],
}
CDTE_CADMIUM_MASS_FRACTION = 112.414 / (112.414 + 127.60)
CDTE_TELLURIUM_MASS_FRACTION = 1.0 - CDTE_CADMIUM_MASS_FRACTION #Fração em massa do CdTe (Cd=112.414, Te=127.60) para regra de mistura
#Dados NIST Fonte: physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/

NIST_Ga = {
    "E_MeV": [
        0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010,
        0.015, 0.020, 0.030, 0.040, 0.050, 0.060, 0.080, 0.100, 0.150,
        0.200, 0.300, 0.400, 0.500, 0.600,
    ],
    "muen_rho": [
        1692, 5044, 2496, 878.2, 408.2, 222.9, 135.2, 60.82, 32.50,
        61.35, 30.59, 10.61, 4.810, 2.560, 1.518, 0.6613, 0.3497,
        0.1193, 0.06463, 0.03782, 0.03156, 0.02920, 0.02793,
    ],
    "mu_rho": [
        1.697e3, 5.087e3, 2.515e3, 8.857e2, 4.130e2, 2.266e2, 1.382e2,
        6.302e1, 3.421e1, 8.537e1, 3.928e1, 1.281e1, 5.726, 3.076, 1.868,
        8.823e-1, 5.197e-1, 2.387e-1, 1.619e-1, 1.123e-1, 9.325e-2,
        8.236e-2, 7.487e-2,
    ],
}
#Dados NIST Fonte: physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/

NIST_As = {
    "E_MeV": [
        0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010,
        0.015, 0.020, 0.030, 0.040, 0.050, 0.060, 0.080, 0.100, 0.150,
        0.200, 0.300, 0.400, 0.500, 0.600,
    ],
    "muen_rho": [
        2116, 5164, 2902, 1039, 486.2, 266.7, 162.2, 73.30, 39.28,
        63.71, 33.10, 11.98, 5.537, 2.983, 1.783, 0.7837, 0.4155,
        0.1399, 0.07350, 0.04043, 0.03261, 0.02966, 0.02813,
    ],
    "mu_rho": [
        2.121e3, 5.227e3, 2.931e3, 1.049e3, 4.920e2, 2.709e2, 1.656e2,
        7.573e1, 4.115e1, 9.856e1, 4.564e1, 1.506e1, 6.760, 3.635, 2.203,
        1.030, 5.971e-1, 2.622e-1, 1.719e-1, 1.150e-1, 9.414e-2,
        8.259e-2, 7.483e-2,
    ],
}
GAAS_GALLIUM_MASS_FRACTION = 69.723 / (69.723 + 74.921595)
GAAS_ARSENIC_MASS_FRACTION = 1.0 - GAAS_GALLIUM_MASS_FRACTION

# NIST XCOM: https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z08.html
NIST_O = {
    "E_MeV":  [2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 8e-3, 1e-2, 1.5e-2, 2e-2, 3e-2, 4e-2,
               5e-2, 6e-2, 8e-2, 1e-1, 1.5e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1,
               8e-1, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
    "mu_rho": [6.949e2, 2.171e2, 9.315e1, 4.790e1, 2.770e1, 1.163e1, 5.952,
               1.836, 8.651e-1, 3.779e-1, 2.585e-1, 2.132e-1, 1.907e-1,
               1.678e-1, 1.551e-1, 1.361e-1, 1.237e-1, 1.070e-1, 9.566e-2,
               8.729e-2, 8.070e-2, 7.087e-2, 6.372e-2, 5.697e-2, 5.185e-2,
               4.459e-2, 3.597e-2, 3.100e-2, 2.777e-2, 2.552e-2, 2.263e-2,
               2.089e-2],
    "muen_rho": [6.926e2, 2.158e2, 9.221e1, 4.715e1, 2.708e1, 1.116e1, 5.565,
                 1.545, 6.179e-1, 1.729e-1, 7.530e-2, 4.414e-2, 3.207e-2,
                 2.468e-2, 2.355e-2, 2.506e-2, 2.679e-2, 2.877e-2, 2.953e-2,
                 2.971e-2, 2.957e-2, 2.887e-2, 2.794e-2, 2.669e-2, 2.551e-2,
                 2.350e-2, 2.066e-2, 1.882e-2, 1.757e-2, 1.668e-2, 1.553e-2,
                 1.483e-2],
}
SIO2_DENSITY_G_CM3 = 2.2
SIO2_SI_MASS_FRACTION = 28.0855 / (28.0855 + 2 * 15.999)
SIO2_O_MASS_FRACTION = 1.0 - SIO2_SI_MASS_FRACTION

# Interpolacao log-log.
def loglog_interp(E_MeV, table, key):
    energies = np.asarray(table["E_MeV"], dtype=float)
    coefficients = np.asarray(table[key], dtype=float)
    if E_MeV < energies[0] or E_MeV > energies[-1]:
        raise ValueError(
            f"Energia {E_MeV:g} MeV fora da faixa tabelada "
            f"({energies[0]:g}–{energies[-1]:g} MeV)"
        )
    x = np.log(energies)
    y = np.log(coefficients)
    return float(np.exp(np.interp(np.log(E_MeV), x, y)))

#Coeficiente de massa do SiO2 por regra de mistura [cm2/g].
def sio2_mass_attenuation(E_keV, key="muen_rho"):
    E_MeV = E_keV / 1e3
    si = loglog_interp(E_MeV, NIST_SI, key)
    oxygen = loglog_interp(E_MeV, NIST_O, key)
    return SIO2_SI_MASS_FRACTION * si + SIO2_O_MASS_FRACTION * oxygen

#Calcula o mu_en/rho do CdTe pela regra das frações em massa [cm2/g].
def cdte_mass_absorption(E_keV):
    return sensor_mass_coefficient(E_keV, "CdTe", "muen_rho")

#Calcula o mu_en/rho do GaAs pela regra das frações em massa [cm2/g].
def gaas_mass_absorption(E_keV):
    return sensor_mass_coefficient(E_keV, "GaAs", "muen_rho")

#Retorna mu/rho ou mu_en/rho do material selecionado [cm2/g].
def sensor_mass_coefficient(E_keV, sensor_material, coefficient):
    if sensor_material == "Si":
        return loglog_interp(E_keV / 1e3, NIST_SI, coefficient)
    if sensor_material == "GaAs":
        ga = loglog_interp(E_keV / 1e3, NIST_Ga, coefficient)
        arsenic = loglog_interp(E_keV / 1e3, NIST_As, coefficient)
        return (GAAS_GALLIUM_MASS_FRACTION * ga+ GAAS_ARSENIC_MASS_FRACTION * arsenic)
    if sensor_material == "CdTe":
        cd = loglog_interp(E_keV / 1e3, NIST_Cd, coefficient)
        te = loglog_interp(E_keV / 1e3, NIST_Te, coefficient)
        return (CDTE_CADMIUM_MASS_FRACTION * cd + CDTE_TELLURIUM_MASS_FRACTION * te)
    raise ValueError("sensor_material deve ser 'Si', 'GaAs' ou 'CdTe'")

#Retorna as propriedades: densidade, mu/rho e mu_en/rho do sensor selecionado.
def sensor_properties(E_keV, sensor_material):
    if sensor_material not in SENSOR_DENSITIES_G_CM3:
        raise ValueError("SENSOR_MATERIAL deve ser 'Si', 'GaAs' ou 'CdTe'")
    return {
        "density": SENSOR_DENSITIES_G_CM3[sensor_material],
        "mu_rho": sensor_mass_coefficient(E_keV, sensor_material, "mu_rho"),
        "muen_rho": sensor_mass_coefficient(E_keV, sensor_material, "muen_rho"),
    }


# Dados experimentais de mu_en/rho para os materiais [cm2/g]. -- 
# COEFFICIENTS D'ATTENUATION MASSIQUE ET D'ABSORPTION MASSIQUE EN ENERGIE POUR LES PHOTONS DE 10 keV A 10 MeV 
# e https://physics.nist.gov/PhysRefData/XrayMassCoef/tab4.html
EXPERIMENTAL_SIO2_E_MEV = np.array([
    0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.060, 0.080,
    0.100, 0.150, 0.200, 0.500, 1.000, 2.000, 5.000, 10.000,
])
EXPERIMENTAL_SIO2_MUEN_RHO = np.array([
    18.2, 5.31, 2.20, 0.628, 0.246, 0.126, 0.0778, 0.0429,
    0.0329, 0.0277, 0.0281, 0.0299, 0.0278, 0.0235, 0.0180, 0.0158,
])

EXPERIMENTAL_CdTe_E_MEV = np.array([
    0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.060, 0.080,
    0.100, 0.150, 0.200, 0.500, 1.000, 2.000, 5.000, 10.000,
])

EXPERIMENTAL_CdTe_MUEN_RHO = np.array([1.322E+02, 4.404E+01, 1.988E+01, 1.119E+01 , 9.635E+00, 6.229E+00, 4.163E+00, 2.083E+00,
    1.181E+00, 4.101E-01, 1.982E-01, 3.952E-02 , 2.575E-02, 2.188E-02, 0.0180, 2.618E-02,
])

EXPERIMENTAL_GaAs_E_MEV = np.array([
    0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010,
    0.015, 0.020, 0.030, 0.040, 0.050, 0.060, 0.080, 0.100, 0.150,
    0.200, 0.300, 0.400, 0.500, 0.600,
])
EXPERIMENTAL_GaAs_MUEN_RHO = np.array([
    1.911e3, 5.106e3, 2.706e3, 9.614e2, 4.486e2, 2.456e2, 1.492e2,
    6.728e1, 3.601e1, 6.257e1, 3.189e1, 1.132e1, 5.187, 2.779, 1.655,
    0.7246, 0.3838, 0.1299, 0.06921, 0.03916, 0.03210, 0.02943, 0.02802,
])

#Compara mu_en/rho experimental e calculado para CdTe.
def plot_cdte_coefficient_comparison():
    
    experimental = EXPERIMENTAL_CdTe_MUEN_RHO
    calculated = np.array([
        cdte_mass_absorption(E_MeV * 1e3)
        for E_MeV in EXPERIMENTAL_CdTe_E_MEV
    ])
    relative_error_pct = (calculated - experimental) / experimental * 100
    max_error_index = np.argmax(np.abs(relative_error_pct))

    print("\nComparação de mu_en/rho do CdTe:")
    print("Energia (MeV) | Experimental | Calculado | Erro relativo (%)")
    for energy, measured, estimate, error in zip(
        EXPERIMENTAL_CdTe_E_MEV, experimental, calculated, relative_error_pct,
    ):
        print(
            f"{energy:12.3f} | {measured:12.5g} | {estimate:9.5g} | "
            f"{error:+16.2f}"
        )
    print(
        f"Erro percentual absoluto médio (MAPE): "
        f"{np.mean(np.abs(relative_error_pct)):.2f}%"
    )
    print(
        f"Maior erro percentual absoluto: "
        f"{abs(relative_error_pct[max_error_index]):.2f}% "
        f"em {EXPERIMENTAL_CdTe_E_MEV[max_error_index]:g} MeV"
    )
    print(
        f"Viés percentual médio (calculado - experimental): "
        f"{np.mean(relative_error_pct):+.2f}%"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(
        EXPERIMENTAL_CdTe_E_MEV, calculated, marker="o",
        label="Calculado (NIST XCOM + regra de mistura)",
    )
    ax.loglog(
        EXPERIMENTAL_CdTe_E_MEV, experimental,
        linestyle="none", marker="s", label="Experimental",
    )
    ax.set_xlabel("Energia (MeV)")
    ax.set_ylabel("Coeficiente de absorção de energia de massa, mu_en/rho (cm²/g)")
    ax.set_title("CdTe: comparação do coeficiente de absorção")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    fig.tight_layout()
    return fig, ax

#Compara mu_en/rho calculado pela mistura com a tabela NIST de GaAs.
def plot_gaas_coefficient_comparison():
    reference = EXPERIMENTAL_GaAs_MUEN_RHO
    calculated = np.array([
        gaas_mass_absorption(E_MeV * 1e3)
        for E_MeV in EXPERIMENTAL_GaAs_E_MEV
    ])
    relative_error_pct = (calculated - reference) / reference * 100
    max_error_index = np.argmax(np.abs(relative_error_pct))

    print("\nComparação de mu_en/rho do GaAs (referência NIST, não medição experimental):")
    print("Energia (MeV) | Referência NIST | Calculado | Diferença (%)")
    for energy, reference_value, estimate, error in zip(
        EXPERIMENTAL_GaAs_E_MEV, reference, calculated, relative_error_pct,
    ):
        print(
            f"{energy:12.5g} | {reference_value:15.6g} | "
            f"{estimate:9.5g} | {error:+13.2f}"
        )
    print(
        f"Erro percentual absoluto médio (MAPE): "
        f"{np.mean(np.abs(relative_error_pct)):.3f}%"
    )
    print(
        f"Maior diferença percentual absoluta: "
        f"{abs(relative_error_pct[max_error_index]):.2f}% "
        f"em {EXPERIMENTAL_GaAs_E_MEV[max_error_index]:g} MeV"
    )
    print(
        f"Viés percentual médio (calculado - referência): "
        f"{np.mean(relative_error_pct):+.3f}%"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(
        EXPERIMENTAL_GaAs_E_MEV, calculated, marker="o",
        label="Calculado (Ga + As, regra de mistura)",
    )
    ax.loglog(
        EXPERIMENTAL_GaAs_E_MEV, reference,
        linestyle="none", marker="s", label="Referência NIST para GaAs",
    )
    ax.set_xlabel("Energia (MeV)")
    ax.set_ylabel("Coeficiente de absorção de energia de massa, mu_en/rho (cm²/g)")
    ax.set_title("GaAs: coeficiente de absorção calculado x referência NIST")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    fig.tight_layout()
    return fig, ax


#Comparação do mu_en/rho experimental e calculado para SiO2. ---https://inis.iaea.org/records/cgraw-2pq15--- Não sabia se era valido
def plot_sio2_coefficient_comparison():
    calculated = np.array([
        sio2_mass_attenuation(E_MeV * 1e3, "muen_rho")
        for E_MeV in EXPERIMENTAL_SIO2_E_MEV
    ])
    experimental = EXPERIMENTAL_SIO2_MUEN_RHO
    relative_error_pct = (calculated - experimental) / experimental * 100
    max_error_index = np.argmax(np.abs(relative_error_pct))

    print("\nComparação de mu_en/rho do SiO2:")
    print("Energia (MeV) | Experimental | Calculado | Erro relativo (%)")
    for energy, measured, estimate, error in zip(
        EXPERIMENTAL_SIO2_E_MEV, experimental, calculated, relative_error_pct,
    ):
        print(
            f"{energy:12.3f} | {measured:12.5g} | {estimate:9.5g} | "
            f"{error:+16.2f}"
        )
    print(f"Erro percentual absoluto médio (MAPE): "
          f"{np.mean(np.abs(relative_error_pct)):.2f}%")
    print(
        f"Maior erro percentual absoluto: "
        f"{abs(relative_error_pct[max_error_index]):.2f}% "
        f"em {EXPERIMENTAL_SIO2_E_MEV[max_error_index]:g} MeV"
    )
    print(f"Viés percentual médio (calculado - experimental): "
          f"{np.mean(relative_error_pct):+.2f}%")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(
        EXPERIMENTAL_SIO2_E_MEV, calculated, marker="o",
        label="Calculado (NIST XCOM + regra de mistura)",
    )
    ax.loglog(
        EXPERIMENTAL_SIO2_E_MEV, EXPERIMENTAL_SIO2_MUEN_RHO,
        linestyle="none", marker="s", label="Experimental",
    )
    ax.set_xlabel("Energia (MeV)")
    ax.set_ylabel("Coeficiente de absorção de energia de massa, mu_en/rho (cm²/g)")
    ax.set_title("SiO₂: coeficiente de absorção experimental x calculado")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    fig.tight_layout()
    return fig, ax  


# ----------------------------------------------------------------------
# 3) Feixe: fluencia e fluencia de energia (ICRU 85, Sec. 3.1.3)
# ----------------------------------------------------------------------
# Fluencia no centro de um feixe gaussiano 2D [fotons/cm2].
def peak_fluence(n_photons, fwhm_x_um, fwhm_y_um=None): #Fwhm_x_um e fwhm_y_um em micrometros é a largura do feixe 
    fwhm_y_um = fwhm_x_um if fwhm_y_um is None else fwhm_y_um #Se não for fornecido, assume-se simetria do feixe
    sx = fwhm_x_um * FWHM_TO_SIGMA * 1e-4      # cm, seria o desvio padrao do feixe gaussiano em x (o quanto o feixe se espalha)
    sy = fwhm_y_um * FWHM_TO_SIGMA * 1e-4
    return n_photons / (2.0 * math.pi * sx * sy) #Retorna a fluencia de pico em fotons/cm2 (central)

def energy_fluence(fluence_cm2, E_keV): #Calculo de Fluencia de energia (Psi) a partir da fluencia de fotons (Phi) e energia do feixe (E)
    """Psi = Phi * E  [J/cm2]."""
    return fluence_cm2 * E_keV * 1e3 * EV_TO_J

# ----------------------------------------------------------------------
# 4) Dose (ICRU 85, Eq. 5.1.1 e 5.1.3; NIST chap3 eq. 13)
# ----------------------------------------------------------------------
def dose_Gy(psi_J_cm2, coef_cm2_g, depth_um=0.0, density=2.33, lambert=True): #Estimativa de dose em Gy a partir da fluencia de energia (Psi) e coeficiente de atenuacao (mu/rho ou mu_en/rho)
    """D = Psi * coef * exp(-mu*z)   [J/g -> Gy = 1000 J/kg]."""
    atten = math.exp(-coef_cm2_g * density * depth_um * 1e-4) if lambert else 1.0 #Fator de atenuacao de Beer-Lambert
    return psi_J_cm2 * coef_cm2_g * atten * 1e3


def asic_sio2_dose_Gy(
    psi_J_cm2, E_keV, silicon_depth_um, sensor_material="Si",
): #Dose estimada na superficie de SiO2 apos atravessar o sensor selecionado.
 
    if silicon_depth_um < 0:
        raise ValueError("profundidade do sensor deve ser não negativa")
    properties = sensor_properties(E_keV, sensor_material)
    transmission = math.exp(
        -properties["mu_rho"] * properties["density"] * silicon_depth_um * 1e-4
    )
    psi_at_asic = psi_J_cm2 * transmission
    muen_sio2 = sio2_mass_attenuation(E_keV, "muen_rho")
    return dose_Gy(psi_at_asic, muen_sio2, density=SIO2_DENSITY_G_CM3,lambert=False)

# ----------------------------------------------------------------------
# 5) Varredura (scan) e dose acumulada (coloquei numa estrutura retangular)
# ----------------------------------------------------------------------
# fluência de energia no pico de um pulso, coeficiente de atenuação (mu/rho ou mu_en/rho), largura do feixe (FWHM), posições dos pulsos, posição do feixe em y, largura e altura da estrutura, número de pontos em x e y, limites de x e y, e outros parâmetros opcionais. 
def _scan_gaussian_shape(fwhm_x_um, positions_um, y_beam_um, x_limits_um, y_limits_um, nx, ny, fwhm_y_um=None):
    fwhm_y_um = fwhm_x_um if fwhm_y_um is None else fwhm_y_um
    xs = np.linspace(*x_limits_um, nx)
    ys = np.linspace(*y_limits_um, ny)
    X, Y = np.meshgrid(xs, ys)
    shape = np.zeros_like(X)
    for xp in positions_um:
        exponent = -4.0 * math.log(2.0) * (
            (X - xp) ** 2 / fwhm_x_um ** 2
            + (Y - y_beam_um) ** 2 / fwhm_y_um ** 2
        )
        shape += np.exp(exponent)
    return xs, ys, shape

#Mapa de dose do feixe
def scan_dose_map(psi_peak, coef, fwhm_x_um, positions_um, y_beam_um, width_um, height_um, nx=400, ny=140, x_limits_um=None, y_limits_um=None, fwhm_y_um=None, **kw): 
    """Mapa de dose [Gy] de uma linha de pulsos sobre o retangulo W x H."""
    x_limits_um = (0, width_um) if x_limits_um is None else x_limits_um
    y_limits_um = (0, height_um) if y_limits_um is None else y_limits_um
    xs, ys, shape = _scan_gaussian_shape(
        fwhm_x_um, positions_um, y_beam_um,
        x_limits_um, y_limits_um, nx, ny,
        fwhm_y_um=fwhm_y_um,
    )
    return xs, ys, dose_Gy(psi_peak, coef, **kw) * shape

#Mapa de dose de um feixe no SiO2 apos atravessar o sensor de Si.
def asic_sio2_dose_map(
    psi_peak, E_keV, silicon_depth_um, fwhm_x_um, positions_um, y_beam_um,
    width_um, height_um, nx=400, ny=140, x_limits_um=None, y_limits_um=None,
    sensor_material="Si", fwhm_y_um=None,
):
    if silicon_depth_um < 0:
        raise ValueError("profundidade do sensor deve ser não negativa")
    properties = sensor_properties(E_keV, sensor_material)
    transmission = math.exp(
        -properties["mu_rho"] * properties["density"] * silicon_depth_um * 1e-4
    )
    return scan_dose_map(psi_peak * transmission,sio2_mass_attenuation(E_keV, "muen_rho"),
        fwhm_x_um, positions_um, y_beam_um, width_um, height_um, nx=nx, ny=ny,
        x_limits_um=x_limits_um, y_limits_um=y_limits_um,
        fwhm_y_um=fwhm_y_um,
        density=SIO2_DENSITY_G_CM3, lambert=False,)

#Mapa de fluência de fótons de um feixe gaussiano
def scan_fluence_map(fluence_peak_cm2, fwhm_x_um, positions_um, y_beam_um, x_limits_um, y_limits_um, nx=400, ny=140, fwhm_y_um=None):
    xs, ys, shape = _scan_gaussian_shape(
        fwhm_x_um, positions_um, y_beam_um, x_limits_um, y_limits_um, nx, ny,
        fwhm_y_um=fwhm_y_um,
    )
    return xs, ys, fluence_peak_cm2 * shape

# --------------------Resumo--------------------------------------------------
def summarize(map_Gy):
    return dict(mean_krad=map_Gy.mean() * GY_TO_KRAD,max_krad=map_Gy.max() * GY_TO_KRAD,min_krad=map_Gy.min() * GY_TO_KRAD)

# ----------------------------------------------------------------------
# 6) Substituição de parâmetros e execução do script ----- Pode utilizar aquele artigo ou coisa da Mogno
# ----------------------------------------------------------------------
if __name__ == "__main__":
    E_keV, N, FWHMX, FWHMY = 22, 7.6e5, 94, 90 #-- Mogno
    sensor = SENSOR_MATERIAL
    sensor_data = sensor_properties(E_keV, sensor)
    sensor_density = sensor_data["density"]
    mu = sensor_data["mu_rho"]
    men = sensor_data["muen_rho"]
    muen_sio2 = sio2_mass_attenuation(E_keV, "muen_rho")
    print(f"Coeficiente mu_en/rho do SiO2 = {muen_sio2:.4f} cm²/g")
    phi = peak_fluence(N, FWHMX, FWHMY) #Função de Fluencia de pico (Phi) em fotons/cm2
    psi = energy_fluence(phi, E_keV) #Função de Fluencia de energia (Psi) em J/cm2
    print("----------Parâmetros do feixe:----------")
    print(f"Material do sensor = {sensor} | densidade = {sensor_density:g} g/cm³")
    print(f"Energia Utilizada = {E_keV} keV | Contagem de fotons = {N:.3e} fótons por pulso | FWHM x = {FWHMX} um | FWHM y = {FWHMY} um")
    print(f"fluencia pico = {phi:.3e} fotons/cm2 | psi pico = {psi:.4e} J/cm2")
    print(f"livre caminho medio em {sensor} = {1e4/(mu*sensor_density):.0f} um")

    D_mu  = dose_Gy(psi, mu, density=sensor_density)  * GY_TO_KRAD
    D_men = dose_Gy(psi, men, density=sensor_density) * GY_TO_KRAD
    print(f"----------\nDose de pico por pulso (superficie):----------")
    print(f"  usando mu/rho    : {D_mu:.3f} krad({sensor})")
    print(f"  usando mu_en/rho : {D_men:.3f} krad({sensor})")
    for z in (1, 10, 300):
        print(f"  mu_en/rho a {z:>4} um de profundidade: "
              f"{dose_Gy(psi, men, depth_um=z, density=sensor_density)*GY_TO_KRAD:.5f} krad")
        print(f"  superficie de SiO2 apos {z:>4} um de {sensor}: "
              f"{asic_sio2_dose_Gy(psi, E_keV, z, sensor)*GY_TO_KRAD:.5f} krad(SiO2)")
# Intervalos de plots e região de interesse
    W, H = 55, 55.
    plot_half_range = 2 * max(FWHMX, FWHMY)
    plot_limits = (-plot_half_range, plot_half_range)
    for n_pulses, label in ((1, "1 pulso"),): #Ideia de analisar vários pulsos
        pos = [W / 2] * n_pulses
        xs, ys, M = scan_dose_map(
            psi, men, FWHMX, pos, H / 2, W, H, fwhm_y_um=FWHMY, density=sensor_density,
        )
        pos_centered = [x - W / 2 for x in pos]
        plot_xs, plot_ys, plot_M = scan_dose_map(
            psi, men, FWHMX, pos_centered, 0, W, H,
            x_limits_um=plot_limits, y_limits_um=plot_limits,
            fwhm_y_um=FWHMY,
            density=sensor_density,
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        imagem = ax.pcolormesh(plot_xs, plot_ys, plot_M * GY_TO_KRAD, shading="auto")
        fig.colorbar(imagem, ax=ax, label=f"Dose (krad({sensor}))")
        ax.set_xlabel("x relativo ao centro da estrutura (um)")
        ax.set_ylabel("y relativo ao centro do feixe (um)")
        ax.set_title(f"Mapa de dose no sensor {sensor} — {label}")
        ax.set_aspect("equal")
        fig.tight_layout()
 #plot de fluencia de fotons
        fluence_xs, fluence_ys, fluence = scan_fluence_map(phi, FWHMX, pos_centered, 0, plot_limits, plot_limits, fwhm_y_um=FWHMY)
        fig, ax = plt.subplots(figsize=(8, 6))
        imagem = ax.pcolormesh(fluence_xs, fluence_ys, fluence, shading="auto", cmap="viridis",)
        fig.colorbar(imagem, ax=ax, label="Fluencia de fótons (fotons/cm²)")
        ax.set_xlabel("x relativo ao centro da estrutura (um)")
        ax.set_ylabel("y relativo ao centro do feixe (um)")
        ax.set_title(f"Mapa de fluência — {label}")
        ax.set_aspect("equal")
        fig.tight_layout() 


        r = summarize(M)
        print(f"\nPulsos no centro ({label}) - mu_en/rho:")
        print(f"  média na superfície = {r['mean_krad']:.5f} krad | "
              f"max = {r['max_krad']:.5f} | min = {r['min_krad']:.5f}")  

    Sensor_Asic_depth_um = 675 #Profundidade do sensor até a superfície do SiO2 (ASIC) em micrômetros
    asic_xs, asic_ys, asic_map = asic_sio2_dose_map(psi, E_keV, Sensor_Asic_depth_um, FWHMX, [0.0], 0.0, W, H,
        x_limits_um=plot_limits, y_limits_um=plot_limits,
        sensor_material=sensor,
        fwhm_y_um=FWHMY,
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    imagem = ax.pcolormesh(asic_xs, asic_ys, asic_map * GY_TO_KRAD, shading="auto",)
    fig.colorbar(imagem, ax=ax, label="Dose (krad(SiO2))")
    ax.set_xlabel("x relativo ao centro da estrutura (um)")
    ax.set_ylabel("y relativo ao centro do feixe (um)")
    ax.set_title(f"Dose no ASIC (SiO2) — 1 pulso, após {Sensor_Asic_depth_um:g} um de {sensor}")
    ax.set_aspect("equal")
    fig.tight_layout()

    asic_metrics = summarize(asic_map)
    print(f"\nDose no ASIC (SiO2), 1 pulso após {Sensor_Asic_depth_um:g} um de {sensor}:")
    print(f"  média na superfície = {asic_metrics['mean_krad']:.5f} krad | "
          f"max = {asic_metrics['max_krad']:.5f} krad | "
          f"min = {asic_metrics['min_krad']:.5f} krad")
    plot_sio2_coefficient_comparison()
    if sensor == "GaAs":
        plot_gaas_coefficient_comparison()
    elif sensor == "CdTe":
        plot_cdte_coefficient_comparison()
    plt.show() 
