"""Original ablation settings; values match the final notebook cell."""

SEEDS = [42, 123, 2026]
CASES = ["adaptive_ewc", "rollback", "episodic_replay"]
OUTPUT_DIR = "neurone_v1_ablation_results"
TASK_STEPS = 10_000
BATCH_SIZE = 64
GAMMA = 0.97
TARGET_UPDATE_STEPS = 1_000
ROLLBACK_INTERVAL = 100
AFTER_TASK_EPISODES = 100
PRE_TASK_EPISODES = 50
FINAL_EPISODES = 200
DEVICE = "auto"
SKIP_COMPLETED = True
MAKE_PLOTS = True
CONTINUE_ON_ERROR = False
DRY_RUN = False

ABLATION_CONFIG = {
    "seeds": SEEDS, "cases": CASES, "output_dir": OUTPUT_DIR,
    "task_steps": TASK_STEPS, "batch_size": BATCH_SIZE, "gamma": GAMMA,
    "target_update_steps": TARGET_UPDATE_STEPS, "rollback_interval": ROLLBACK_INTERVAL,
    "after_task_episodes": AFTER_TASK_EPISODES, "pre_task_episodes": PRE_TASK_EPISODES,
    "final_episodes": FINAL_EPISODES, "device": DEVICE,
    "skip_completed": SKIP_COMPLETED, "make_plots": MAKE_PLOTS,
    "continue_on_error": CONTINUE_ON_ERROR, "dry_run": DRY_RUN,
}
