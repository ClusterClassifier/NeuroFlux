# System Overview

**Neuroflux** is the paper name for the implemented Neurone V1 prototype. It studies continual learning for one secondary user selecting among eight synthetic radio channels or an idle action across seven sequential primary-user occupancy regimes (T1–T7).

The observation contains five recent SNR and occupancy values per channel (80 values). A feed-forward DQN produces nine action values. Training uses Boltzmann exploration; evaluation selects the highest-value action. The learner combines an active replay buffer, persistent episodic memory, and elastic weight consolidation (EWC). A custom ADWIN-style detector monitors batch temporal-difference (TD) error; an alarm can restore a recent policy snapshot, refresh memory and importance estimates, and reset monitoring and exploration state. Parameter importance is computed by a local asynchronous worker.

The environment draws channel occupancy independently from task-specific Bernoulli probabilities. The system does **not** implement typed traffic-drift classification, a recurrent policy, component-specific plasticity, realistic sensing errors, a remote edge server, or physical-layer throughput. Those belong to broader project proposals or other versions. See `methods.md` and `experiment_protocol.md` for the exact V1 study.
