import sys, os
sys.path.insert(0, os.path.abspath("src"))

import numpy as np
from params import JumpDiffusionParams
from stable_strat import create_stablecoin_short_strategy

X_params = JumpDiffusionParams(mu=0.0, sigma=0.001, lam=0.001, delta=0.0, eta=10.0)
setattr(X_params, "direction", -1)
Y_params = JumpDiffusionParams(mu=0.0, sigma=0.85, lam=0.12, delta=0.25, eta=4.0)
setattr(Y_params, "direction", -1)

strategy = create_stablecoin_short_strategy(
    W0=10000.0, days=30,
    r_X=0.04, r_Y=0.08, b_X=0.8,
    X_params=X_params, Y_params=Y_params,
    X0=1.0, Y0=1.0, fht_func=None,
)
print('Strategy created. T=', strategy.p.T)

# Evaluate psi_h over grid using health_process.psi_h
thetas = np.linspace(-2.0, 2.0, 401)
valid = []
exceptions = []
for th in thetas:
    try:
        _ = strategy.health_process.psi_h(th)
        valid.append(True)
    except Exception as e:
        valid.append(False)
        exceptions.append((th, type(e).__name__, str(e)))

valid_thetas = thetas[np.array(valid)]
print('Valid count:', valid_thetas.size, '/', thetas.size)
if valid_thetas.size > 0:
    print('Valid theta domain:', valid_thetas.min(), valid_thetas.max())
else:
    print('No valid thetas found — printing first 10 exceptions for diagnosis:')
    for entry in exceptions[:10]:
        print(entry)

from collections import Counter
exc_types = Counter([t for (_, t, _) in exceptions])
print('\nException types summary:')
for k,v in exc_types.items():
    print(f'  {k}: {v}')
