"""Original AST-based generation of isolated ablation workers."""

import ast

# Applies the original overrides before the generated worker calls main().
def _v1_prepare_source(source_cells, cfg, variant, seed):
    modules = [ast.parse(source) for source in source_cells]
    main = next(node for tree in modules for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == 'main')

    class RunConfiguration(ast.NodeTransformer):
        def visit_Assign(self, node):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            replacements = {
                'EXPERIMENT_SEED': seed, 'TASK_STEPS': cfg['task_steps'],
                'BATCH_SIZE': cfg['batch_size'], 'GAMMA': cfg['gamma'],
                'TARGET_UPDATE': cfg['target_update_steps'],
                'ROLLBACK_INTERVAL': cfg['rollback_interval'],
            }
            for name, value in replacements.items():
                if name in names:
                    node.value = ast.Constant(value=value)
            if 'device' in names and cfg['device'] != 'auto':
                node.value = ast.parse('torch.device(' + repr(cfg['device']) + ')', mode='eval').body
            if 'complexity_df' in names:
                for child in ast.walk(node):
                    if isinstance(child, ast.Dict):
                        for i, key in enumerate(child.keys):
                            if isinstance(key, ast.Constant) and key.value == 'evaluation_episodes_final':
                                child.values[i] = ast.Constant(value=cfg['final_episodes'])
            return self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                if node.func.id in {'run_evaluation', 'evaluate_policy_on_env', 'evaluate_baseline'}:
                    for keyword in node.keywords:
                        if keyword.arg == 'num_episodes' and isinstance(keyword.value, ast.Constant):
                            old = keyword.value.value
                            key = {50: 'pre_task_episodes', 100: 'after_task_episodes',
                                   200: 'final_episodes'}.get(old)
                            if key:
                                keyword.value = ast.Constant(value=cfg[key])
            return self.generic_visit(node)

        def visit_If(self, node):
            if variant == 'no_episodic_replay' and ast.unparse(node.test) == 'len(episodic_memory) > num_episodic':
                node.test = ast.Constant(value=False)
            return self.generic_visit(node)

    RunConfiguration().visit(main)
    body = []
    for tree in modules:
        for node in tree.body:
            if isinstance(node, ast.If) and '__name__' in ast.unparse(node.test):
                continue
            body.append(node)
    overrides = ''
    if variant == 'fixed_ewc':
        overrides = "\n_OriginalEWC = EWC\nclass EWC(_OriginalEWC):\n    def get_adaptive_lambda(self):\n        return self.lambda_max\n"
    elif variant == 'no_ewc':
        overrides = "\n_OriginalEWC = EWC\nclass EWC(_OriginalEWC):\n    def compute_ewc_loss(self):\n        return torch.tensor(0.0, device=next(self.policy_net.parameters()).device)\n"
    elif variant == 'no_rollback':
        overrides = "\ndef load_rollback_checkpoint(policy_net, filepath):\n    return False\n"
    elif variant == 'uniform_replay':
        overrides = "\n_OriginalReplayBuffer = ReplayBuffer\nclass ReplayBuffer(_OriginalReplayBuffer):\n    def get_top_k_transitions(self, k, policy_net, target_net, gamma, device, batch_size=512):\n        return random.sample(list(self.buffer), min(k, len(self.buffer)))\n"
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    compatibility = "\n_v1_original_sample = random.sample\ndef _v1_sequence_sample(population, k, *args, **kwargs):\n    if isinstance(population, deque):\n        population = list(population)\n    return _v1_original_sample(population, k, *args, **kwargs)\nrandom.sample = _v1_sequence_sample\n"
    return ast.unparse(module) + '\n' + compatibility + overrides + '\nmain()\n'
