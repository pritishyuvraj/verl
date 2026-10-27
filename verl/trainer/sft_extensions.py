"""Optional lifecycle hooks for the native SFT trainer (no alternate fit loop).

Hooks execute on every rank. Collectives must therefore be entered by every
rank; rank-zero-only IO belongs inside the callback, after collective gathers.
Generation evaluation runs after the entire training process has exited. A phase
boundary does not change the optimizer's total training steps or LR schedule.
"""

from time import perf_counter

from hydra.utils import instantiate


def synchronized_time(device):
    """Read a wall clock only after pending accelerator work has completed."""
    device.synchronize()
    return perf_counter()


def build_callbacks(configs):
    return [instantiate(config) for config in configs]


def notify(callbacks, event, **kwargs):
    for callback in callbacks:
        handler = getattr(callback, event, None)
        if handler is not None:
            handler(**kwargs)


def phase_stop_step(resume_step, total_steps, stop_after_step=None):
    if type(total_steps) is not int or total_steps <= resume_step:
        raise ValueError("Training horizon must exceed the resumed step")
    stop = total_steps if stop_after_step is None else stop_after_step
    if type(stop) is not int or not resume_step < stop <= total_steps:
        raise ValueError("Phase boundary must be after resume and within the fixed training horizon")
    return stop


def validate_launch(config):
    validation = config.get("generation_validation")
    if validation is not None:
        # Workers receive a resolved native phase config from the controller;
        # direct invocation cannot release GPUs and perform the eval lifecycle.
        raise RuntimeError(
            "Use verl.trainer.generation_sft for the generation validation lifecycle; "
            "direct native training is blocked. Held-out panel/driver qualification are pending."
        )
