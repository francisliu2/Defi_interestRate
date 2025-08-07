#!/usr/bin/env python3
"""
Debug script for MGF first hitting time solver
"""
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import warnings
from mgf_first_hitting_time import MGFFirstHittingTime, create_example_parameters

def test_mgf_debug():
    print("=== MGF DEBUG TEST ===")
    
    # Create simple parameters
    parameters = create_example_parameters()
    
    print("Parameters:")
    print(f"  sigma_h: {parameters.sigma_h}")
    print(f"  eta_X: {parameters.eta_X}, eta_Y: {parameters.eta_Y}")
    print(f"  delta_X: {parameters.delta_X}, delta_Y: {parameters.delta_Y}")
    print(f"  mu_X - mu_Y: {parameters.mu_X - parameters.mu_Y}")
    print()
    
    # Initialize solver
    try:
        mgf_solver = MGFFirstHittingTime(parameters)
        print("✓ MGF solver initialized successfully")
    except Exception as e:
        print(f"✗ MGF solver initialization failed: {e}")
        return
    
    # Test parameters
    h0 = 0.5
    lambda_X0 = parameters.lambda_X0
    lambda_Y0 = parameters.lambda_Y0
    
    print(f"Test conditions: h0={h0}, λ_X0={lambda_X0}, λ_Y0={lambda_Y0}")
    print()
    
    # Test MGF computation for small s values
    s_values = [0.01, 0.05, 0.1, 0.5, 1.0]
    
    for s in s_values:
        print(f"Testing s = {s}")
        
        try:
            # Test ODE system evaluation at initial conditions
            y0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            dydt = mgf_solver.ode_system_real(0.0, y0, s)
            print(f"  ODE evaluation at h=0: {dydt}")
            
            # Test ODE system evaluation at small h
            y_test = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
            dydt_test = mgf_solver.ode_system_real(0.01, y_test, s)
            print(f"  ODE evaluation at h=0.01: {dydt_test[:3]}...")
            
            if np.any(np.isnan(dydt_test)) or np.any(np.isinf(dydt_test)):
                print(f"  ✗ ODE system returns NaN/Inf")
                continue
            
            # Test MGF computation
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                mgf = mgf_solver.moment_generating_function(s, h0, lambda_X0, lambda_Y0)
                
                if w:
                    for warning in w:
                        print(f"  Warning: {warning.message}")
            
            if np.isnan(mgf):
                print(f"  ✗ M({s}) = NaN")
            elif np.isinf(mgf):
                print(f"  ✗ M({s}) = Inf")
            elif np.real(mgf) <= 0:
                print(f"  ✗ M({s}) = {mgf} (should be > 0)")
            else:
                print(f"  ✓ M({s}) = {mgf}")
            
        except Exception as e:
            print(f"  ✗ Exception: {e}")
        
        print()
    
    # Test validation
    print("Running validation...")
    validation = mgf_solver.validate_solution(0.1, h0, lambda_X0, lambda_Y0)
    print(f"Validation result: {validation}")

if __name__ == "__main__":
    test_mgf_debug()