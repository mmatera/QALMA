#!/usr/bin/env python

import matplotlib as mpl

mpl.use("module://mpl_ascii")
import glob
import pickle
import sys

import numpy as np
from matplotlib import pyplot as plt

from qalma.evolution.simulation import Simulation
from qalma.operators.states import DensityOperatorMixin, GibbsDensityOperator

for idx, filename in enumerate([fn for pat in sys.argv[1:] for fn in glob.glob(pat)]):
    print(filename)
    is_hdf5 = True
    if filename[-4:] == ".pkl":
        is_hdf5 = False
        obs_key = 0
        with open(filename, "rb") as f:
            simulation = pickle.load(f)
    elif filename[-3:] == ".h5":
        simulation = Simulation.load_hdf5(filename, allow_pickle=True)
        # Files written before the format 1 store the keys as strings.
        obs_key = "0" if "0" in simulation.expect_ops else 0
    else:
        print("wrong file extension", filename)
        exit(-1)

    if not simulation.expect_ops:
        if simulation.states:
            system = simulation.states[0].system
            print("build expect from states")
            sz_total = sum(system.site_operator("Sz", s) for s in system.sites)

            def ensure_is_state(k):
                if not isinstance(k, DensityOperatorMixin):
                    return GibbsDensityOperator(k)
                return k

            simulation.expect_ops[obs_key] = [
                ensure_is_state(k).to_qutip_operator().expect(sz_total)
                for k in simulation.states
            ]
            if is_hdf5:
                print(">> storing expect ops")
                simulation.save_hdf5(filename)
            else:
                with open(filename, "wb") as f:
                    pickle.dump(simulation, f)
            print("data are now stored...")
        else:
            print(" does not contain information. Skip.")
            continue

    (plt.plot if idx == 0 else plt.scatter)(
        simulation.time_span[: len(simulation.expect_ops[obs_key])],
        [np.real(x) for x in simulation.expect_ops[obs_key]],
        label=filename[:20],
    )


plt.legend()
plt.savefig("sz_evol.svg")
plt.show()
