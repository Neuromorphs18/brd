# Live demo of a delay network on BrainDrop.
# See brd/dyanmics/delay_network.ipynb for details.
import nengo

# This forces the backend to be BrainDrop
# and use a large enough dt to keep up with IO.
# requires nengo-gui:add-dt-cmd-line-option
import inspect
settings = inspect.getargvalues(
    inspect.stack()[1][0]).locals['self'].settings
settings.backend = "nengo_brainstorm"
settings.dt = 0.05

import numpy as np
from nengolib import Lowpass
from nengolib.networks import readout
from nengolib.signal import Balanced
from nengolib.synapses import PadeDelay, pade_delay_error, ss2sim

theta = 0.1
order = 3
freq = 3
power = 1.5

print("PadeDelay(%s, %s) => %f%% error @ %sHz" % (
    theta, order, 100*abs(pade_delay_error(theta*freq, order=order)), freq))
pd = PadeDelay(theta=theta, order=order)

# Heuristic for normalizing state so that each dimension is ~[-1, +1]
rz = Balanced()(pd, radii=1./(np.arange(len(pd))+1))
sys = rz.realization

# Compute matrix to transform from state (x) -> sampled window (u)
t_samples = 10
C = np.asarray([readout(len(pd), r)
                for r in np.linspace(0, 1, t_samples)]).dot(rz.T)
assert C.shape == (t_samples, len(sys))

n_neurons = 128  # per dimension
tau = 0.018329807108324356  # guess from Terry's notebook
map_hw = ss2sim(sys, synapse=Lowpass(tau), dt=None)
assert np.allclose(map_hw.A, tau*sys.A + np.eye(len(sys)))
assert np.allclose(map_hw.B, tau*sys.B)

syn_probe = Lowpass(tau)
map_out = ss2sim(sys, synapse=syn_probe, dt=settings.dt)

with nengo.Network() as model:
    u = nengo.Node(output=0, label='u')

    # This is needed because a single node can't connect to multiple
    # different ensembles. We need a separate node for each ensemble.
    Bu = [nengo.Node(output=lambda _, u, b_i=map_hw.B[i].squeeze(): b_i*u,
                     size_in=1, label='Bu[%d]' % i)
          for i in range(len(sys))]
    
    X = []
    for i in range(len(sys)):
        X.append(nengo.Ensemble(
            n_neurons=n_neurons, dimensions=1, label='X[%d]' % i))

    w = nengo.Node(size_in=t_samples)
    for i in range(len(sys)):
        nengo.Connection(u, Bu[i], synapse=None)
        nengo.Connection(Bu[i], X[i], synapse=tau)
        for j in range(len(sys)):
            nengo.Connection(X[j], X[i], synapse=tau,
                             function=lambda x_j, a_ij=map_hw.A[i, j]: a_ij*x_j)
        
        nengo.Connection(X[i], w, transform=C.dot(map_out.A[:, i:i+1]),
                         synapse=syn_probe)
    nengo.Connection(u, w, transform=C.dot(map_out.B), synapse=syn_probe)
