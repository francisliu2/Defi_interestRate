#!/usr/bin/env python3
"""
Demonstration of adaptive Gil-Pelaez parameters for different h0 values.
"""

print("Adaptive Gil-Pelaez Parameters Demonstration")
print("=" * 60)
print()

print("🎯 MOTIVATION:")
print("The characteristic function φ(s) behaves very differently for different h₀:")
print()
print("Small h₀ (close to boundary):")
print("  • φ(s) decays rapidly → smaller s_max needed")
print("  • Sharp features → higher point density required")
print("  • Fast convergence")
print()
print("Large h₀ (far from boundary):")
print("  • φ(s) decays slowly → larger s_max needed")
print("  • Smoother behavior → fewer points acceptable")
print("  • Slower convergence, more oscillatory integrand")
print()

print("⚙️ ADAPTIVE STRATEGY IMPLEMENTED:")
print()
print("1. h₀-based scaling:")
print("   h₀ < 0.2:  s_max × 0.5,   n_points × 1.5   (close to boundary)")
print("   0.2 ≤ h₀ < 0.5:  s_max × 1.0,   n_points × 1.0   (moderate)")
print("   0.5 ≤ h₀ < 1.0:  s_max × 1.5,   n_points × 1.2   (far)")
print("   h₀ ≥ 1.0:  s_max × (2 + 0.5ln(h₀)), n_points × (1.3 + 0.2ln(h₀))")
print()

print("2. Time-based scaling:")
print("   time_factor = 1 + 0.3 × ln(max(T, 0.1))")
print("   Longer times need higher precision")
print()

print("3. Grid density adaptation:")
print("   Small h₀: 40% of points in [s_min, s_max/20] (dense near 0)")
print("   Large h₀: 30% of points in [s_min, s_max/10] (less dense near 0)")
print()

print("4. Convergence checking:")
print("   - Iterative refinement with parameter scaling")
print("   - Relative error monitoring")
print("   - Automatic parameter adjustment")
print()

print("📊 EXAMPLE PARAMETER SELECTION:")
print("Base parameters: s_max = 50, n_points = 1000, T = 0.5")
print()

# Simulate the adaptive parameter selection
def adaptive_params_demo(h0, T):
    """Simulate the adaptive parameter selection."""
    base_s_max = 50.0
    base_n_points = 1000
    
    # h0-based scaling
    if h0 < 0.2:
        s_max_factor = 0.5
        n_points_factor = 1.5
        regime = "close to boundary"
    elif h0 < 0.5:
        s_max_factor = 1.0
        n_points_factor = 1.0
        regime = "moderate distance"
    elif h0 < 1.0:
        s_max_factor = 1.5
        n_points_factor = 1.2
        regime = "far from boundary"
    else:
        import math
        s_max_factor = 2.0 + 0.5 * math.log(h0)
        n_points_factor = 1.3 + 0.2 * math.log(h0)
        regime = "very far"
    
    # Time scaling
    import math
    time_factor = 1.0 + 0.3 * math.log(max(T, 0.1))
    
    # Final parameters
    s_max = base_s_max * s_max_factor * time_factor
    n_points = int(base_n_points * n_points_factor * time_factor)
    
    # Bounds
    s_max = max(20.0, min(s_max, 500.0))
    n_points = max(500, min(n_points, 5000))
    
    return s_max, n_points, regime

test_cases = [
    (0.1, "very close to liquidation"),
    (0.3, "moderate safety margin"),
    (0.8, "comfortable distance"),
    (1.5, "far from boundary"),
    (3.0, "very safe position")
]

print("h₀\tRegime\t\t\ts_max\tn_points\tEfficiency")
print("-" * 70)

for h0, description in test_cases:
    s_max, n_points, regime = adaptive_params_demo(h0, 0.5)
    efficiency = 50000 / (s_max * n_points)  # Relative computational cost
    print(f"{h0:.1f}\t{description:20s}\t{s_max:6.1f}\t{n_points:4d}\t\t{efficiency:.3f}")

print()
print("✅ BENEFITS OF ADAPTIVE PARAMETERS:")
print("• Computational efficiency: Use minimal resources for each h₀")
print("• Numerical accuracy: Optimize grid for characteristic function behavior")
print("• Automatic convergence: Built-in refinement for difficult cases")
print("• Monotonicity preservation: Consistent precision across h₀ range")
print()

print("🔧 USAGE:")
print("# Default: uses adaptive parameters")
print("cdf = solver.gil_pelaez_cdf(T=0.5, h0=1.0, lambda_X0=2.0, lambda_Y0=2.0)")
print()
print("# With convergence checking:")
print("result = solver.gil_pelaez_cdf_with_convergence(T=0.5, h0=1.0, lambda_X0=2.0, lambda_Y0=2.0)")
print("print(f'CDF: {result[\"cdf\"]:.6f}, Converged: {result[\"converged\"]}')")
print()

print("✅ ADAPTIVE PARAMETERS IMPLEMENTED - Optimal efficiency and accuracy!")