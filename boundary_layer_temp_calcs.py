import numpy as np

def stagnation_temperature(V_inf, T_inf, Cp=1000, dissociation_correction=True):
    """Calculate stagnation temperature considering real gas effects."""
    T0 = T_inf + (V_inf ** 2) / (2 * Cp)
    
    if dissociation_correction:
        # Apply empirical correction for air dissociation at high temperatures
        if T0 > 5000:
            T0 = 14000 - 5000 * np.exp(-T0 / 7000)  # Rough empirical correction
    
    return T0

def boundary_layer_edge_temperature(M, T0, gamma=1.4, real_gas=True):
    """Calculate temperature at the boundary layer edge considering real gas effects."""
    
    if real_gas:
        # For high Mach numbers, gamma decreases due to dissociation and ionization
        gamma_eff = max(1.2, 1.4 - 0.1 * np.log10(M))  # Approximate trend
    else:
        gamma_eff = gamma
    
    Te = T0 / (1 + ((gamma_eff - 1) / 2) * M ** 2)
    
    return Te

# Example: Apollo 10 reentry
V_inf = 11100  # m/s
T_inf = 220  # K (free-stream temperature at high altitude)
M = 36  # Approximate Mach number

# Compute stagnation temperature with real gas correction
T0_corrected = stagnation_temperature(V_inf, T_inf, dissociation_correction=True)

# Compute boundary layer edge temperature with real gas effects
Te_corrected = boundary_layer_edge_temperature(M, T0_corrected, real_gas=True)

print(f"Corrected Stagnation Temperature: {T0_corrected:.2f} K")
print(f"Boundary Layer Edge Temperature: {Te_corrected:.2f} K")
