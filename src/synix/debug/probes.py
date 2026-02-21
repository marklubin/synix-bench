"""Probe dataclass and helpers for mid-run breakpoint questions.

Probes inject researcher questions into the agent's conversation at
specific trigger points (e.g., after a push/pop, heap event, or step N).
The response is logged without polluting the agent's task context.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Probe:
    """A breakpoint probe that fires at a specific trigger point.

    Attributes:
        trigger: When to fire -- "pop_frame", "push_frame", "heap_alloc",
                 "step=N" (one-shot), or "done" (after agent finishes).
        question: The researcher question to inject.
        fired: For one-shot probes (step=N), tracks whether already fired.
    """

    trigger: str
    question: str
    fired: bool = False


def parse_probe_spec(spec: str) -> Probe | None:
    """Parse a probe spec string like 'after:pop_frame:What is your plan?'.

    Returns a Probe or None if the format is invalid.
    """
    parts = spec.split(":", 2)
    if len(parts) == 3 and parts[0] == "after":
        return Probe(trigger=parts[1], question=parts[2])
    return None


def check_probes(
    probes: list[Probe],
    event: dict | None,
    heap_events: list[dict],
    step_num: int,
) -> list[Probe]:
    """Return probes whose triggers match the current state."""
    to_fire = []
    event_type = event.get("type") if event else None
    for probe in probes:
        t = probe.trigger
        if t == "pop_frame" and event_type == "pop":
            to_fire.append(probe)
        elif t == "push_frame" and event_type == "push":
            to_fire.append(probe)
        elif t == "heap_alloc" and any(he["type"] == "alloc" for he in heap_events):
            to_fire.append(probe)
        elif t.startswith("step="):
            try:
                target = int(t.split("=", 1)[1])
            except ValueError:
                continue
            if step_num == target and not probe.fired:
                probe.fired = True
                to_fire.append(probe)
    return to_fire
