import math

class MartianAtmosphere:
    """
    A simple model of the Martian atmosphere from -10 km to 100 km altitude.
    
    The model is split into two regions:
      1. A lower region (from -10 km to 40 km) where the temperature decreases
         linearly with altitude (lapse rate) and the pressure is computed via the
         barometric formula for a non-isothermal layer.
      2. An upper, isothermal region (from 40 km to 100 km) where the temperature 
         remains constant and the pressure decays exponentially.
    
    Constants (typical values):
      - Reference altitude (0 km): P0 = 610 Pa, T0 = 210 K.
      - Gravity on Mars: g = 3.71 m/s^2.
      - Specific gas constant for CO2 (Mars' main constituent): R = 189 J/(kg*K).
      - Lapse rate for lower region: L = 0.0013 K/m.
      
    Note: This is a simplified model and actual conditions on Mars may vary.
    """
    
    def __init__(self):
        # Reference conditions at 0 km altitude
        self.P0 = 610.0       # Pressure at 0 km in Pa
        self.T0 = 210.0       # Temperature at 0 km in K
        
        # Physical constants
        self.g = 3.71         # Gravitational acceleration on Mars in m/s^2
        self.R = 189.0        # Specific gas constant for Martian CO2 atmosphere (J/(kg*K))
        self.L = 0.0013       # Lapse rate (K/m) for altitudes up to 40 km
        
        # Transition altitude between the two regions (in meters)
        self.z_transition = 40000.0  # 40 km
        
        # Precompute the exponent for the lapse rate (non-isothermal) region:
        self.exponent = self.g / (self.R * self.L)
        
        # Temperature at the transition altitude (40 km)
        self.T_transition = self.T0 - self.L * self.z_transition  # ≈ 210 - 52 = 158 K
        
        # Pressure at the transition altitude using the lapse rate formula:
        self.P_transition = self.P0 * ((self.T_transition / self.T0) ** self.exponent)
    
    def properties(self, altitude_km):
        """
        Calculate the Martian atmospheric properties at a given altitude.
        
        Parameters:
            altitude_km (float): Altitude in kilometers (must be between -10 and 100 km)
        
        Returns:
            dict: A dictionary containing:
                - 'altitude_m': Altitude in meters.
                - 'temperature_K': Temperature at the altitude in Kelvin.
                - 'pressure_Pa': Pressure at the altitude in Pascals.
                - 'density_kg_m3': Density at the altitude in kg/m^3 (via the ideal gas law).
        """
        altitude_m = altitude_km * 1000.0
        
        # Check if the altitude is within the modeled range:
        if altitude_km < -10 or altitude_km > 100:
            raise ValueError("Altitude out of modeled range (-10 km to 100 km)")
        
        if altitude_m <= self.z_transition:
            # Lower region: use linear temperature profile (lapse rate)
            T = self.T0 - self.L * altitude_m
            # Pressure using the barometric formula for a layer with a constant lapse rate
            P = self.P0 * ((T / self.T0) ** self.exponent)
        else:
            # Upper region (isothermal): Temperature remains constant at T_transition
            T = self.T_transition
            delta_z = altitude_m - self.z_transition
            # Pressure decays exponentially from the transition altitude
            P = self.P_transition * math.exp(-self.g * delta_z / (self.R * T))
        
        # Density from the ideal gas law: ρ = P / (R * T)
        density = P / (self.R * T)
        
        return {
            "altitude_m": altitude_m,
            "temperature_K": T,
            "pressure_Pa": P,
            "density_kg_m3": density
        }
    
    def __call__(self, altitude_km):
        """Allow the class instance to be called directly with an altitude in km."""
        return self.properties(altitude_km)


# # Example usage:
# if __name__ == "__main__":
#     atmosphere = MartianAtmosphere()
#     test_altitudes = [-10, 0, 10, 40, 100]  # altitudes in km
#     for alt in test_altitudes:
#         props = atmosphere.properties(alt)
#         print(f"Altitude: {alt:>4} km | Temperature: {props['temperature_K']:6.2f} K | "
#               f"Pressure: {props['pressure_Pa']:8.2f} Pa | Density: {props['density_kg_m3']:7.5f} kg/m^3")
