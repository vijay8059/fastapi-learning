from collections.abc import Awaitable, Callable
from pathlib import Path

import yaml

from app.graph.state import AgentState

_CAPSULES_DIR = Path(__file__).parent.parent / "capsules"


def _load_capsule_content(path: Path) -> str | None:
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text())
    return data["content"].strip()


def build_capsule_composition_node() -> Callable[[AgentState], Awaitable[dict[str, str]]]:
    """Stage B — merge tiny prompt fragments ("capsules") into one system prompt.

    Mirrors assist-workflows' capsule_composition node: "core" always loads,
    plus one fragment selected by the grounding role. Real capsules are
    priority-merged YAML with triggers/owned_fields; ours is just two files
    concatenated, to show the shape of the idea without the full registry.
    """

    async def capsule_composition_node(state: AgentState) -> dict[str, str]:
        role = state["grounding"]["role"]

        fragments = [_load_capsule_content(_CAPSULES_DIR / "core.yaml")]
        fragments.append(_load_capsule_content(_CAPSULES_DIR / "role" / f"{role}.yaml"))

        composed = "\n\n".join(fragment for fragment in fragments if fragment)
        return {"system_prompt": composed}

    return capsule_composition_node
