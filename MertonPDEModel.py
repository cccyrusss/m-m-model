import numpy as np
from scipy.optimize import root
from scipy.interpolate import interp1d

# -------------------------------
# 1. Core Merton PDE Model (Finite Difference Solver)
# -------------------------------
class MertonPDEModel:
    """
    Solves Merton's General Equation (1977) for Contingent Claims.
    This uses an Explicit Finite Difference scheme.
    """
    def __init__(self, r, sigma, T, V_max, N_v, N_t):
        self.r = r           # Risk-free rate
        self.sigma = sigma   # Asset Volatility (sigma_V)
        self.T = T           # Time to maturity
        self.V_max = V_max   # Max Asset value for grid
        self.N_v = N_v       # Number of asset steps
        self.N_t = N_t       # Number of time steps
        
        self.dV = V_max / N_v
        self.dt = T / N_t
        self.V_grid = np.linspace(0, V_max, N_v+1)

    def solve_equity(self, debt_face):
        """
        Solves for Equity Value. 
        Merton (1977) treats Equity as a Call Option on Firm Assets (V).
        Boundary Condition at T: Max(0, V - Debt)
        """
        # 1. Set Terminal Boundary Condition (at maturity T)
        # Payoff = max(0, V - B) [cite: 94]
        F = np.maximum(0, self.V_grid - debt_face)
        
        # Pre-compute indices and squares for speed
        j = np.arange(1, self.N_v)
        j2 = j**2  # FIXED: Was j*2
        sigma2 = self.sigma**2 # FIXED: Was sigma*2
        
        # Explicit Finite Difference Coefficients
        # Derived from Merton's Eq (1) 
        # alpha = coeff for F[i-1], beta = coeff for F[i], gamma = coeff for F[i+1]
        alpha = 0.5 * self.dt * (sigma2 * j2 - self.r * j)
        beta  = 1.0 - self.dt * (sigma2 * j2 + self.r)
        gamma = 0.5 * self.dt * (sigma2 * j2 + self.r * j)
        
        # Stability Check: Beta must be positive for explicit scheme stability
        if np.any(beta < 0):
            raise ValueError(f"Unstable Scheme! Increase N_t (currently {self.N_t}) or decrease N_v.")

        # 2. Iterate Backwards in Time (from T to 0)
        for n in range(self.N_t):
            F_new = np.zeros_like(F)
            
            # Interior points calculation
            F_new[1:-1] = alpha * F[0:-2] + beta * F[1:-1] + gamma * F[2:]
            
            # Lower Boundary (V=0): Firm is bankrupt, Equity is worthless
            F_new[0] = 0 
            
            # Upper Boundary (V -> infinity): Linearity (Deep in the money)
            F_new[-1] = 2*F_new[-2] - F_new[-3]
            
            F = F_new
            
        return F

# -------------------------------
# 2. The Calibration Engine (Inverse Problem)
# -------------------------------
class BalanceSheetGuesser:
    def __init__(self, r, T, last_reported_debt):
        self.r = r
        self.T = T
        self.last_reported_debt = last_reported_debt

    def guess(self, obs_equity_val, obs_equity_vol):
        """
        Infers Market Asset Value (V) and Asset Vol (sigma_V) 
        from observable Equity Value (E) and Equity Vol (sigma_E).
        """
        def objective(params):
            V_guess, sigma_V_guess = params
            
            # Constraints to prevent solver crashing
            if V_guess <= 0 or sigma_V_guess <= 0.001: 
                return [1e5, 1e5] # Penalty
            
            # Setup Model with SAFE time steps (N_t=3000 to ensure stability)
            model = MertonPDEModel(self.r, sigma_V_guess, self.T, 
                                   V_max=V_guess*3, N_v=100, N_t=3000)
            
            # Solve for Equity Curve
            F_equity_curve = model.solve_equity(self.last_reported_debt)
            
            # Create interpolator to find exact value at V_guess
            interp = interp1d(model.V_grid, F_equity_curve, kind='cubic', fill_value="extrapolate")
            E_model = float(interp(V_guess))
            
            # Calculate Model Equity Volatility via Delta
            # Relation: sigma_E = (V / E) * Delta * sigma_V
            h = V_guess * 0.01
            delta = (interp(V_guess+h) - interp(V_guess-h)) / (2*h)
            vol_E_model = (V_guess / E_model) * delta * sigma_V_guess
            
            # We want difference to be 0
            return [E_model - obs_equity_val, vol_E_model - obs_equity_vol]

        # Initial Guess: V is roughly Equity + Debt, Vol is roughly half of Equity Vol
        x0 = [obs_equity_val + self.last_reported_debt, obs_equity_vol * 0.5]
        
        sol = root(objective, x0, method='hybr')
        
        if not sol.success:
            print("Warning: Calibration did not perfectly converge.")
            
        return {
            "Implied_V": sol.x[0],
            "Implied_Asset_Vol": sol.x[1],
            "Implied_Debt_Value_Market": sol.x[0] - obs_equity_val, 
            "Used_Debt_Face": self.last_reported_debt
        }

# -------------------------------
# 3. Execution (Example)
# -------------------------------
if __name__ == "__main__":
    # --- SCENARIO: Sep 30, 2025 ---
    # Market Data
    obs_market_cap = 85.0   # $85 Billion
    obs_iv = 0.32           # 32% Volatility
    r_rate = 0.04           # 4.0% Risk free
    T_maturity = 3.0        # Avg debt maturity
    debt_last_Q = 40.0      # Debt known from last quarter

    print(f"--- 📅 Calibration Step ---")
    guesser = BalanceSheetGuesser(r_rate, T_maturity, debt_last_Q)
    prediction = guesser.guess(obs_market_cap, obs_iv)

    print(f"Observed Equity: ${obs_market_cap}B, Vol: {obs_iv:.1%}")
    print(f"Implied Firm Assets (V): ${prediction['Implied_V']:.2f} B")
    print(f"Implied Asset Volatility: {prediction['Implied_Asset_Vol']:.2%}")
    
    # Validation
    actual_debt_book = 50.0 # Secretly borrowed 10B more
    print(f"\n--- 🔍 Reality Check ---")
    print(f"The model 'thought' debt was ${debt_last_Q}B (Face).")
    print(f"The actual debt is ${actual_debt_book}B.")
    print(f"Implied Market Value of Debt: ${prediction['Implied_Debt_Value_Market']:.2f}B")
    
    # If Implied Debt Value < Actual Face Value, the market is pricing in distress
    if prediction['Implied_Debt_Value_Market'] < actual_debt_book * 0.95:
         print("Result: The market is pricing this debt at a discount (Distress Risk).")
    else:
         print("Result: The market views this debt as safe.")