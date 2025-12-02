import numpy as np
from scipy.optimize import root
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# -------------------------------
# 1. Core Merton PDE Model (Optimized for Calibration)
# -------------------------------
class MertonPDEModel:
    def __init__(self, r, sigma, T, V_max, N_v, N_t):
        self.r = r
        self.sigma = sigma
        self.T = T
        self.V_max = V_max
        self.N_v = N_v
        self.N_t = N_t
        
        self.dV = V_max / N_v
        self.dt = T / N_t
        self.V_grid = np.linspace(0, V_max, N_v+1)

    def solve_equity(self, debt_face):
        """
        Solves specifically for Equity (Call on V).
        Returns the Equity Value vector at t=0.
        """
        # Payoff: max(0, V - D)
        F = np.maximum(0, self.V_grid - debt_face)
        
        # Pre-compute coefficients to speed up the loop
        j = np.arange(1, self.N_v)
        v_j = self.V_grid[j]
        
        # Coefficients for tridiagonal matrix (Explicit scheme for clarity)
        # Note: For production, Implicit (Crank-Nicolson) is more stable.
        alpha = 0.5 * self.dt * (self.sigma*2 * j*2 - self.r * j)
        beta  = 1 - self.dt * (self.sigma*2 * j*2 + self.r)
        gamma = 0.5 * self.dt * (self.sigma*2 * j*2 + self.r * j)
        
        for n in range(self.N_t):
            F_new = np.zeros_like(F)
            # Interior points
            F_new[1:-1] = alpha * F[0:-2] + beta * F[1:-1] + gamma * F[2:]
            
            # Boundary: V=0 -> Equity=0
            F_new[0] = 0
            
            # Boundary: V=Max -> Linear Extrapolation
            F_new[-1] = 2*F_new[-2] - F_new[-3]
            
            F = F_new
            
        return F

# -------------------------------
# 2. The "Guessing" Engine (Inverse Merton)
# -------------------------------
class BalanceSheetGuesser:
    def __init__(self, r, T, last_reported_debt):
        self.r = r
        self.T = T
        self.last_reported_debt = last_reported_debt  # Best proxy for current debt

    def guess(self, obs_equity_val, obs_equity_vol):
        """
        Infer Market Asset Value (V) and Asset Vol (sigma_V) 
        using CURRENT Market Data + LAST Known Debt.
        """
        def objective(params):
            V_guess, sigma_V_guess = params
            
            # Sanity constraints
            if V_guess <= 0 or sigma_V_guess <= 0.01: return [1e5, 1e5]
            
            # Setup Model
            # V_max must encompass the guess
            model = MertonPDEModel(self.r, sigma_V_guess, self.T, V_max=V_guess*3, N_v=100, N_t=100)
            F_equity = model.solve_equity(self.last_reported_debt)
            
            # Interpolate Equity Value
            interp = interp1d(model.V_grid, F_equity, kind='cubic', fill_value="extrapolate")
            E_model = float(interp(V_guess))
            
            # Calculate Model Volatility via Delta
            # sigma_E = (V/E) * Delta * sigma_V
            h = V_guess * 0.01
            delta = (interp(V_guess+h) - interp(V_guess-h)) / (2*h)
            vol_E_model = (V_guess / E_model) * delta * sigma_V_guess
            
            return [E_model - obs_equity_val, vol_E_model - obs_equity_vol]

        # Initial Guess: V ~ E + D, sigma_V ~ sigma_E * 0.5
        x0 = [obs_equity_val + self.last_reported_debt, obs_equity_vol * 0.5]
        
        sol = root(objective, x0, method='hybr')
        
        if not sol.success:
            raise ValueError("Calibration failed.")
            
        return {
            "Implied_V": sol.x[0],
            "Implied_Asset_Vol": sol.x[1],
            "Implied_Debt_Value_Market": sol.x[0] - obs_equity_val, # V - E
            "Used_Debt_Face": self.last_reported_debt
        }

# -------------------------------
# 3. The Validation Step (The Cross-Check)
# -------------------------------
def validate_prediction(prediction, actual_report):
    """
    Compares the 'Guessed' state (Sep 30) with the 'Actual' report (Released Nov).
    """
    print("\n=== 🔍 BALANCE SHEET CROSS-CHECK ===")
    
    # 1. Asset Validation
    # We compare Implied V (Market) vs Reported Assets (Book)
    # A ratio > 1.0 implies the market prices in growth/intangibles.
    market_to_book = prediction['Implied_V'] / actual_report['Total_Assets_Book']
    
    print(f"1. Asset Check:")
    print(f"   - Implied Market Assets (V): ${prediction['Implied_V']:.2f} B")
    print(f"   - Actual Book Assets (A):    ${actual_report['Total_Assets_Book']:.2f} B")
    print(f"   - Market-to-Book Ratio:      {market_to_book:.2f}x")
    if market_to_book < 1.0:
        print("   ⚠ WARNING: Market values assets LESS than book value (Distress signal?)")
        
    # 2. Debt Validation
    # Did the company borrow more than we thought?
    # We used 'last_reported_debt' to guess. Now we see the 'actual_debt'.
    debt_surprise = actual_report['Total_Debt_Book'] - prediction['Used_Debt_Face']
    
    print(f"\n2. Debt Surprise Check:")
    print(f"   - Debt Assumed (Jun 30):     ${prediction['Used_Debt_Face']:.2f} B")
    print(f"   - Debt Actual (Sep 30):      ${actual_report['Total_Debt_Book']:.2f} B")
    print(f"   - Net Borrowing/Repayment:   ${debt_surprise:.2f} B")
    
    # 3. Solvency Check
    # Compare Implied Net Worth (V - Actual Debt) vs Market Cap
    adjusted_equity = prediction['Implied_V'] - actual_report['Total_Debt_Book']
    
    print(f"\n3. Hidden Leverage Check:")
    print(f"   If we update Debt to Actuals, implied Equity should be: ${adjusted_equity:.2f} B")
    print(f"   Observed Market Cap was: ${obs_market_cap:.2f} B")
    
    diff = adjusted_equity - obs_market_cap
    if abs(diff) > obs_market_cap * 0.1:
         print(f"   ⚠ Mismatch! The market likely anticipated the debt change.")
    else:
         print(f"   ✅ Consistent. Market implicitly knew the debt level.")

# -------------------------------
# Example Execution Flow
# -------------------------------

# --- TIME: September 30, 2025 ---
# You are sitting at your desk. The Earnings Report is 2 months away.
# You gather data:
obs_market_cap = 85.0   # $85 Billion
obs_iv = 0.32           # 32% Volatility
r_rate = 0.045          # 4.5% Risk free
T_maturity = 3.0        # Avg debt maturity

# You only have JUNE 30 data for debt (Last Quarter)
debt_last_Q = 40.0      # $40 Billion

print(f"--- 📅 Date: Sep 30, 2025 (Prediction Time) ---")
guesser = BalanceSheetGuesser(r_rate, T_maturity, debt_last_Q)
prediction = guesser.guess(obs_market_cap, obs_iv)

print(f"Guessed State:")
print(f"  Implied Asset Value (V): ${prediction['Implied_V']:.2f} B")
print(f"  Implied Asset Vol:       {prediction['Implied_Asset_Vol']:.2%}")

# --- TIME: November 15, 2025 ---
# Earnings Released! Actual Balance Sheet for Sep 30 is revealed.
# Scenario: Company secretly borrowed $10B more to fund a project.
actual_report = {
    'Total_Assets_Book': 90.0,  # Book Value
    'Total_Debt_Book': 50.0     # Actually $50B, not $40B!
}

# --- RUN VALIDATION ---
validate_prediction(prediction, actual_report)