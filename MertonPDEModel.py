import numpy as np
import matplotlib.pyplot as plt

class MertonPDEModel:
    """
    Implements the General Contingent Claim PDE from Merton (1977).
    Equation (1): 0 = 0.5*sigma^2*V^2*F_vv + (rV - D1)*F_v - rF + F_t + D2
    
    We solve this using an Explicit Finite Difference scheme backwards in time.
    """
    
    def __init__(self, r, sigma, T, V_max, N_v, N_t, D1_func=None, D2_func=None):
        """
        Parameters:
        r      : Risk-free rate
        sigma  : Volatility of the underlying asset
        T      : Time to maturity (in years)
        V_max  : Maximum value of V to model (the "infinity" boundary)
        N_v    : Number of grid points for V (Asset Value)
        N_t    : Number of time steps
        D1_func: Function D1(V, t) -> Payout of Underlying Asset (default 0)
        D2_func: Function D2(V, t) -> Payout of Contingent Claim (default 0)
        """
        self.r = r
        self.sigma = sigma
        self.T = T
        self.V_max = V_max
        self.N_v = N_v
        self.N_t = N_t
        
        # Grid setup
        self.dV = V_max / N_v
        self.dt = T / N_t
        self.V_grid = np.linspace(0, V_max, N_v+1)
        self.time_grid = np.linspace(0, T, N_t+1)
        
        # Default payouts to zero if not provided
        self.D1 = D1_func if D1_func else lambda V, t: 0.0
        self.D2 = D2_func if D2_func else lambda V, t: 0.0

    def solve(self, payoff_func, boundary_lower=None, boundary_upper=None):
        """
        Solves for F(V, t=0).
        
        payoff_func: Function h(V) -> Value at T (Boundary condition at maturity) 
        boundary_lower: Condition at V=0 (default 0)
        boundary_upper: Condition at V=V_max (default linear extrapolation)
        """
        # Initialize price grid F[time_step, V_level]
        # We step BACKWARDS from t=T to t=0
        F = np.zeros((self.N_t + 1, self.N_v + 1))
        
        # 1. Set Terminal Condition (at t=T) 
        # F[V, T] = h(V)
        F[self.N_t, :] = [payoff_func(v) for v in self.V_grid]
        
        # Coefficients for the Explicit Method
        # To adapt Merton's Eq(1) for backward stepping (tau = T - t):
        # F_tau = 0.5*sigma^2*V^2*F_vv + (rV - D1)*F_v - rF + D2
        
        # Iterate backwards from T-1 to 0
        for i in range(self.N_t - 1, -1, -1):
            t_current = i * self.dt
            
            for j in range(1, self.N_v): # Skip boundaries 0 and V_max for now
                V = self.V_grid[j]
                
                # Current Payouts
                d1_val = self.D1(V, t_current)
                d2_val = self.D2(V, t_current)
                
                # Finite Difference Approximations
                # Delta (First Derivative F_v)
                delta = (F[i+1, j+1] - F[i+1, j-1]) / (2 * self.dV)
                
                # Gamma (Second Derivative F_vv)
                gamma = (F[i+1, j+1] - 2*F[i+1, j] + F[i+1, j-1]) / (self.dV**2)
                
                # Merton's Equation (1) discretized for backward time step:
                # New_Price = Old_Price + dt * (Change)
                # Note: "Old_Price" here is actually F[i+1] (future) because we move backwards
                
                theta_term = (0.5 * self.sigma**2 * V**2 * gamma) + \
                             ((self.r * V - d1_val) * delta) - \
                             (self.r * F[i+1, j]) + \
                             d2_val
                
                F[i, j] = F[i+1, j] + self.dt * theta_term

            # 2. Apply Boundary Conditions
            
            # Lower Boundary (V=0)
            if boundary_lower:
                F[i, 0] = boundary_lower(t_current)
            else:
                # Default: If V=0, Value usually 0 (unless D2 is positive)
                F[i, 0] = F[i+1, 0] * (1 - self.r * self.dt) # Simple discount
                
            # Upper Boundary (V -> Infinity)
            if boundary_upper:
                F[i, -1] = boundary_upper(t_current, self.V_max)
            else:
                # Linear extrapolation (standard for call options)
                F[i, -1] = 2*F[i, -2] - F[i, -3]

        return F

# --- Example Usage: Replicating Section 3 (M-M Theorem) ---

# Parameters
r = 0.05        # 5% Risk free rate
sigma = 0.2     # 20% Volatility
T = 1.0         # 1 Year maturity
V_max = 200     # Max firm value to model
FaceValue = 100 # Debt Face Value (B)

# Instantiate the Model
model = MertonPDEModel(r, sigma, T, V_max, N_v=100, N_t=2000)

# 1. Price the Equity (Call Option on Firm Value)
# Payoff at T: Max(0, V - B) [cite: 209]
equity_payoff = lambda V: max(0, V - FaceValue)
F_equity = model.solve(equity_payoff)

# 2. Price the Debt (Bond)
# Payoff at T: Min(V, B) [cite: 200]
debt_payoff = lambda V: min(V, FaceValue)
# Note: In section 3, Merton assumes D1 (coupon) = C. We assume Zero Coupon for simplicity here.
F_debt = model.solve(debt_payoff)

# 3. Validation: M-M Theorem
# V_L (Levered Firm) = Equity + Debt
# V_L should equal V (Unlevered Firm Value)
current_V_index = 50 # Let's look at V=100 (Index 50 because V_max=200, N_v=100)
V_actual = model.V_grid[current_V_index]

Equity_Value = F_equity[0, current_V_index]
Debt_Value = F_debt[0, current_V_index]
Firm_Value_Derived = Equity_Value + Debt_Value

print(f"--- Modigliani-Miller Theorem Check (Merton 1977) ---")
print(f"Underlying Firm Assets (V): {V_actual:.2f}")
print(f"Derived Equity Value (S):   {Equity_Value:.2f}")
print(f"Derived Debt Value (D):     {Debt_Value:.2f}")
print(f"Total Levered Value (S+D):  {Firm_Value_Derived:.2f}")
print(f"Difference (Error):         {abs(Firm_Value_Derived - V_actual):.5f}")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(model.V_grid, F_equity[0, :], label='Equity (Call Option)')
plt.plot(model.V_grid, F_debt[0, :], label='Risky Debt')
plt.plot(model.V_grid, F_equity[0, :] + F_debt[0, :], 'k--', label='Total Firm Value (Sum)')
plt.xlabel('Value of Firm Assets (V)')
plt.ylabel('Value of Security')
plt.title('Merton (1977): Pricing Equity and Debt as Contingent Claims')
plt.legend()
plt.grid(True)
plt.show()