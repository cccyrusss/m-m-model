import numpy as np
from scipy.optimize import root
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# =============================================================================
# 1. THE CORE MODEL: MERTON'S PDE SOLVER
# =============================================================================
class MertonPDEModel:
    """
    Implements the pricing logic from Merton (1977).
    It solves the General PDE (Eq 1 in the paper) using Finite Differences.
    """
    def __init__(self, r, sigma, T, V_max, N_v, N_t):
        """
        :param r: Risk-free interest rate (e.g., 0.05)
        :param sigma: Volatility of the underlying Firm Assets (e.g., 0.20)
        :param T: Time to maturity in years
        :param V_max: Maximum asset value for the grid (boundary condition)
        :param N_v: Number of asset steps (grid density)
        :param N_t: Number of time steps (stability)
        """
        self.r = r
        self.sigma = sigma
        self.T = T
        self.V_max = V_max
        self.N_v = N_v
        self.N_t = N_t
        
        # Grid Setup
        self.dV = V_max / N_v
        self.dt = T / N_t
        self.V_grid = np.linspace(0, V_max, N_v+1)

    def solve(self, payoff_func):
        """
        Generic solver for ANY contingent claim.
        :param payoff_func: A function f(V) that defines value at time T.
        """
        # 1. Terminal Condition (at Maturity T)
        F = payoff_func(self.V_grid)
        
        # 2. Pre-compute Finite Difference Coefficients
        # Using indices j=1 to N_v-1 to avoid loop overhead
        j = np.arange(1, self.N_v)
        j2 = j**2            # Corrected: Squared (Python uses ** not *)
        sigma2 = self.sigma**2
        
        # Explicit Scheme Coefficients (derived from Merton Eq 1)
        # alpha corresponds to F[i-1], beta to F[i], gamma to F[i+1]
        alpha = 0.5 * self.dt * (sigma2 * j2 - self.r * j)
        beta  = 1.0 - self.dt * (sigma2 * j2 + self.r)
        gamma = 0.5 * self.dt * (sigma2 * j2 + self.r * j)

        # Stability Warning for Explicit Scheme
        if np.any(beta < 0):
            print("Warning: Stability condition violated. Increase N_t.")

        # 3. Iterate Backwards in Time (from T down to 0)
        for n in range(self.N_t):
            F_new = np.zeros_like(F)
            
            # Interior points
            F_new[1:-1] = alpha * F[0:-2] + beta * F[1:-1] + gamma * F[2:]
            
            # Boundary Condition: V = 0 (Bankruptcy)
            # If assets are 0, claim is usually 0 (discounted)
            F_new[0] = F[0] * (1 - self.r * self.dt)
            
            # Boundary Condition: V = Max (Deep in the money)
            # Linear extrapolation: F'' = 0
            F_new[-1] = 2*F_new[-2] - F_new[-3]
            
            F = F_new
            
        return F

# =============================================================================
# 2. THE ANALYTICAL ENGINE: BALANCE SHEET CALIBRATION
# =============================================================================
class BalanceSheetGuesser:
    """
    Reverse-engineers "Hidden" Firm Value (V) from Observable Stock Prices (E).
    This is the practical application of Merton's theory (KMV style).
    """
    def __init__(self, r, T, debt_face):
        self.r = r
        self.T = T
        self.debt_face = debt_face

    def guess(self, obs_equity_val, obs_equity_vol):
        """
        Solves for V and Sigma_V given E and Sigma_E.
        """
        def objective(params):
            V_guess, sigma_V_guess = params
            
            # Penalize invalid inputs to keep optimizer on track
            if V_guess <= 0 or sigma_V_guess <= 0.001: return [1e5, 1e5]
            
            # Setup Model (High N_t for stability during optimization)
            model = MertonPDEModel(self.r, sigma_V_guess, self.T, 
                                   V_max=V_guess*3, N_v=80, N_t=2000)
            
            # Solve for Equity Curve (Call Option)
            payoff = lambda V: np.maximum(0, V - self.debt_face)
            F_curve = model.solve(payoff)
            
            # Interpolate to find model value at V_guess
            interp = interp1d(model.V_grid, F_curve, kind='cubic', fill_value="extrapolate")
            E_model = float(interp(V_guess))
            
            # Calculate Model Volatility via Delta
            # Relation: sigma_E = (V / E) * Delta * sigma_V
            h = V_guess * 0.01
            delta = (interp(V_guess+h) - interp(V_guess-h)) / (2*h)
            vol_E_model = (V_guess / E_model) * delta * sigma_V_guess
            
            return [E_model - obs_equity_val, vol_E_model - obs_equity_vol]

        # Initial Guess
        x0 = [obs_equity_val + self.debt_face, obs_equity_vol * 0.5]
        
        sol = root(objective, x0, method='hybr')
        
        return {
            "Implied_Asset_Value": sol.x[0],
            "Implied_Asset_Vol": sol.x[1],
            "Success": sol.success
        }

# =============================================================================
# 3. VISUALIZATION: INSIGHTS FROM THE PAPER
# =============================================================================
def plot_merton_insights():
    print("Generating Plots...")
    
    # Parameters
    r = 0.05
    T = 5.0
    V_max = 200
    FaceValue = 80 # Debt Face Value (B)

    # Payoff Functions
    payoff_equity = lambda V: np.maximum(0, V - FaceValue)
    payoff_debt   = lambda V: np.minimum(V, FaceValue)

    # --- Scenario A: Low Volatility (20%) ---
    model_low = MertonPDEModel(r, 0.20, T, V_max, N_v=100, N_t=2000)
    S_low = model_low.solve(payoff_equity)
    D_low = model_low.solve(payoff_debt)

    # --- Scenario B: High Volatility (50%) ---
    model_high = MertonPDEModel(r, 0.50, T, V_max, N_v=100, N_t=2000)
    S_high = model_high.solve(payoff_equity)
    D_high = model_high.solve(payoff_debt)

    # --- Plotting ---
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Modigliani-Miller Proof
    # Show that S + D = V (The diagonal line)
    ax[0].plot(model_low.V_grid, S_low, label='Equity (Call Option)', color='blue')
    ax[0].plot(model_low.V_grid, D_low, label='Risky Debt', color='red')
    ax[0].plot(model_low.V_grid, S_low + D_low, 'k--', linewidth=2, label='Total Firm Value (S+D)')
    ax[0].plot(model_low.V_grid, model_low.V_grid, 'g:', linewidth=1, label='Unlevered Asset V')
    
    ax[0].set_title("Insight 1: The Modigliani-Miller Theorem\n(Total Value is invariant to leverage)", fontsize=12)
    ax[0].set_xlabel("Firm Asset Value (V)")
    ax[0].set_ylabel("Security Value")
    ax[0].axvline(FaceValue, color='gray', linestyle='--', alpha=0.5, label='Default Barrier')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Plot 2: Risk Shifting (Agency Cost)
    # Show how Volatility transfers wealth from Debt to Equity
    ax[1].plot(model_low.V_grid, D_low, color='red', label='Debt (Vol=20%)')
    ax[1].plot(model_high.V_grid, D_high, color='darkred', linestyle='--', label='Debt (Vol=50%)')
    
    ax[1].plot(model_low.V_grid, S_low, color='blue', label='Equity (Vol=20%)')
    ax[1].plot(model_high.V_grid, S_high, color='cyan', linestyle='--', label='Equity (Vol=50%)')

    ax[1].set_title("Insight 2: Risk Shifting / Agency Costs\n(Higher Volatility hurts Debt, helps Equity)", fontsize=12)
    ax[1].set_xlabel("Firm Asset Value (V)")
    ax[1].set_ylabel("Value")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # 1. Run the Calibration Example (Numerical)
    print("--- PART 1: CALIBRATION (The Inverse Problem) ---")
    market_cap = 85.0   # Observed Stock Value
    market_vol = 0.32   # Observed Stock Volatility (32%)
    debt_face  = 40.0   # Known Debt Face Value
    
    guesser = BalanceSheetGuesser(r=0.04, T=3.0, debt_face=debt_face)
    result = guesser.guess(market_cap, market_vol)
    
    print(f"Inputs: Equity=${market_cap}B, Vol={market_vol:.1%}")
    if result['Success']:
        print(f"Output: Implied Asset Value (V) = ${result['Implied_Asset_Value']:.2f}B")
        print(f"Output: Implied Asset Volatility = {result['Implied_Asset_Vol']:.2%}")
    else:
        print("Calibration failed to converge.")

    print("\n--- PART 2: VISUALIZATION (Theory) ---")
    # 2. Run the Plotting to show M-M and Risk Shifting
    plot_merton_insights()