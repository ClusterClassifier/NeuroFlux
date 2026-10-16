# Methods

## Environment and objective

For channel `c` in task `j`, occupancy is `x[c,t] ~ Bernoulli(p[c,j])`. One channel is assigned a relatively low occupancy probability in each task. Actions `0`–`7` attempt transmission; action `8` idles. Reward is `+1` for a clear-channel transmission, `−1` for a collision, and `−0.05` for idle. The seven illustrative regimes are trained in order T1–T7.

## Learning loop

The DQN has layers `80 → 256 ReLU → 256 ReLU → 9`, with Adam learning rate `0.001`, discount `0.97`, batch size `64`, and a target-network copy every `1,000` training steps. Its target and TD loss are

```text
y = r + 0.97(1 − done) max_a′ Q_target(s′,a′)
L_TD = mean[(y − Q_online(s,a))²]
```

An active buffer holds `10,000` transitions; episodic memory holds `2,000` high-absolute-TD-error selections. When available, a batch mixes `51` active and `13` episodic samples. EWC adds `(λ_t/2) Σ_k Σ_i F[k,i](θ_i − θ*[k,i])²`, where `λ_t = 4/[1 + 10 Var_500(|TD error|)]`. The local worker estimates diagonal importance from squared Q-value gradients.

A custom ADWIN-style monitor receives mean absolute batch TD error (`δ=0.002`). Snapshots are saved every `100` steps. An alarm may restore the latest snapshot and triggers the configured memory, importance, and reset response. This is target-network DQN and a local worker, not Double DQN or deployed mobile-edge offloading.
