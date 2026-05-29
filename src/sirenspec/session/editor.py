"""The ``/edit`` suggestive assistant engine for ``sirenspec launch``.

:class:`EditAssistant` turns a natural-language instruction into a *proposed* change to the
workflow YAML.  It never writes silently: it returns the full updated YAML plus a unified
diff for the user to accept, edit, or reject.  The assistant runs on the user's own provider
key — Anthropic first, then OpenAI — and an accepted change must pass ``validate`` before it
is written back to the workflow file.
"""

from __future__ import annotations

import difflib
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from sirenspec.core.models import Workflow
from sirenspec.exceptions import EditAssistantError, SirenSpecError
from sirenspec.providers.registry import resolve_provider
from sirenspec.yaml.parser import load_workflow

# Provider preference order — Anthropic first, then OpenAI — per the locked design decision.
DEFAULT_ANTHROPIC_MODEL = "anthropic:claude-haiku-4-5-20251001"
DEFAULT_OPENAI_MODEL = "openai:gpt-4o-mini"

EDIT_SYSTEM_PROMPT = (
    "You are a careful SirenSpec workflow editor. You are given the current workflow.yaml and "
    "an instruction. Apply the instruction and return the COMPLETE updated workflow YAML and "
    "nothing else — no prose, no explanation, no markdown fences. Preserve everything the "
    "instruction does not ask you to change."
)


@dataclass(frozen=True)
class ProposedChange:
    """A suggested edit awaiting the user's accept / edit / reject decision.

    :param instruction: The natural-language instruction that produced the change.
    :param new_yaml: The complete proposed workflow YAML.
    :param diff_lines: Unified-diff lines from the current YAML to the proposed YAML.
    """

    instruction: str
    new_yaml: str
    diff_lines: list[str]


def choose_model(env: Mapping[str, str] | None = None) -> str:
    """Pick the assistant model from available provider keys (Anthropic before OpenAI).

    :param env: Environment mapping to inspect; defaults to ``os.environ``.
    :raises EditAssistantError: If neither an Anthropic nor an OpenAI key is present.
    :returns: The model URI to use for the assistant.
    """
    env = os.environ if env is None else env
    if env.get("ANTHROPIC_API_KEY"):
        return DEFAULT_ANTHROPIC_MODEL
    if env.get("OPENAI_API_KEY"):
        return DEFAULT_OPENAI_MODEL
    raise EditAssistantError("No provider key found: set ANTHROPIC_API_KEY or OPENAI_API_KEY to use /edit.")


def strip_code_fences(text: str) -> str:
    """Strip a surrounding Markdown code fence from *text* if the model added one.

    :param text: Raw model output that may be wrapped in ``` fences.
    :returns: The fence-free YAML body.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    # Drop the opening fence (e.g. ``` or ```yaml) and a trailing fence if present.
    lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


class EditAssistant:
    """Suggestive workflow editor backed by the user's own LLM provider.

    :param model_uri: Explicit provider URI to use; when ``None`` the model is chosen from
        available environment keys (Anthropic before OpenAI).
    """

    def __init__(self, model_uri: str | None = None) -> None:
        self.model_uri = model_uri

    def resolve_model(self) -> str:
        """Resolve the model URI for this assistant.

        :returns: The explicit ``model_uri`` if set, otherwise the env-derived default.
        """
        return self.model_uri or choose_model()

    async def propose(self, current_yaml: str, instruction: str) -> ProposedChange:
        """Ask the assistant to apply *instruction* and return a proposed change.

        :param current_yaml: The current workflow YAML text.
        :param instruction: The user's natural-language edit instruction.
        :raises EditAssistantError: If no provider is available or the call fails.
        :returns: The :class:`ProposedChange` for the user to review.
        """
        try:
            provider = resolve_provider(self.resolve_model())
        except SirenSpecError as exc:
            raise EditAssistantError(f"Could not initialise the assistant provider: {exc}") from exc

        messages = [
            {"role": "system", "content": EDIT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Current workflow.yaml:\n{current_yaml}\n\nInstruction: {instruction}"},
        ]
        try:
            raw = await provider.complete(messages)
        except SirenSpecError as exc:
            raise EditAssistantError(f"Assistant request failed: {exc}") from exc

        new_yaml = strip_code_fences(raw)
        diff_lines = list(
            difflib.unified_diff(
                current_yaml.splitlines(),
                new_yaml.splitlines(),
                fromfile="workflow.yaml",
                tofile="proposed",
                lineterm="",
            )
        )
        return ProposedChange(instruction=instruction, new_yaml=new_yaml, diff_lines=diff_lines)

    def validate(self, new_yaml: str, workflow_path: Path) -> Workflow:
        """Validate proposed YAML by loading it through the normal workflow parser.

        The proposal is written to a temporary sibling of *workflow_path* (so relative
        sub-workflow refs and ``env_file`` paths resolve identically) and loaded with the
        same ``validate`` gate used everywhere else.  The temp file is always removed.

        :param new_yaml: The proposed workflow YAML to validate.
        :param workflow_path: The real workflow file path (used to site the temp file).
        :raises EditAssistantError: If the proposed YAML fails validation.
        :returns: The validated :class:`~sirenspec.core.models.Workflow`.
        """
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix=".sirenspec-edit-",
            dir=str(workflow_path.parent),
            delete=False,
            encoding="utf-8",
        )
        tmp_path = Path(tmp.name)
        try:
            tmp.write(new_yaml)
            tmp.close()
            return load_workflow(str(tmp_path))
        except SirenSpecError as exc:
            raise EditAssistantError(f"Proposed change failed validation: {exc}") from exc
        except (ValueError, FileNotFoundError) as exc:
            raise EditAssistantError(f"Proposed change failed validation: {exc}") from exc
        finally:
            tmp_path.unlink(missing_ok=True)
