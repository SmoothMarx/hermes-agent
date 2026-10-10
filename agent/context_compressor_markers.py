"""Marker survival: tokens whose PAYLOAD must outlive a compaction summary.

Compaction replaces the middle of a conversation with an LLM-written summary. That is fine for
prose and fatal for a token whose payload is an argument the model cannot re-derive — a skill
name, a session id. The summarizer paraphrases "reload `pdf`, and this session came from
`20261010_195437_513a83`" into "loaded a skill" and "the parent session"; nothing raises, because
the summary still reads correctly and only the way back is gone.

Two families live here. Both are DETERMINISTIC backstops, not prompt wording: the tokens that went
into the summarizer are re-derived from the raw turn list before the call and appended after it if
they are missing from what came back. The summarizer's prompt still asks for the skill markers
verbatim (``_summary_template_sections``); nothing here depends on that, which is the point.

``[SKILL_PRUNED: …]`` — family 1
    Emitted when a ``skill_view`` result is pruned to a one-line stub (#32106). The payload is the
    skill name, and the marker carries the exact reload call.

``[APROPOS-PIN …]`` — family 2
    Emitted by any caller that seeds a row pointing at other sessions (the desktop messenger's
    "Apropos" verb is the first consumer: it opens a new session whose context is another session
    or a group of them). The payload is a session id or a group name. Measured on a real fork
    lineage: the token survives the first compaction (the head is protected there) and by the third
    the prose mention had been reduced to ``session_search(...)`` with the argument gone.

Pointer markers are registered by prefix (``POINTER_MARKER_PREFIXES``) and copied verbatim: this
module does not parse their attributes, it only guarantees they arrive. A new producer adds a
prefix here rather than a second copy of this logic.

The compressor is the caller; the few helpers this module needs from it (content flattening,
redaction) are late-imported inside functions to keep the import direction one-way.
"""

from __future__ import annotations

import re
from typing import Any

# ------------------------------------------------------------------------------------------------
# Shared mechanics
# ------------------------------------------------------------------------------------------------

_PRUNED_SKILLS_NOTE = (
    "(The listed skills' instructions were pruned during context compression. Reload with the "
    "skill_view call in each marker before relying on that skill; one reload per skill is enough "
    "— ignore any older markers for the same skill.)"
)


def _normalize_marker(text: str) -> str:
    """Collapse internal whitespace: a marker a summarizer re-wrapped still compares equal."""
    return " ".join((text or "").split())


def _marker_survived(normalized_summary: str, marker: str) -> bool:
    """Presence check against an already-normalized summary (see ``_normalize_marker``)."""
    return _normalize_marker(marker) in normalized_summary


def _missing_markers(summary: str, markers: list[str]) -> list[str]:
    """The *markers* the summary does not already carry, in order, without duplicates."""
    haystack = _normalize_marker(summary)
    return [
        marker
        for marker in dict.fromkeys(markers)
        if not _marker_survived(haystack, marker)
    ]


def _append_marker_block(
    summary: str, heading: str, markers: list[str], note: str
) -> str:
    """Append ``## <heading>`` with the verbatim *markers* and *note*, redacted like all summary text."""
    from agent.context_compressor import _redact_compaction_text

    block = "\n\n" + heading + "\n" + "\n".join(markers) + "\n" + note
    return summary + _redact_compaction_text(block)


# ------------------------------------------------------------------------------------------------
# Family 1: pruned skills (#32106)
# ------------------------------------------------------------------------------------------------

# Ghost-skill defense: the ONE canonical prune marker; emit sites and presence checks must use the
# same string so they cannot drift. When compaction reduces an old ``skill_view`` result to a
# 1-line metadata summary, the model still believes the skill is loaded even though its
# instructions are gone. ``_skill_pruned_marker()`` builds the marker and every presence check
# matches against the same string, so the emit side and the check side can never drift apart (the
# original PR #44166 emitted ``[SKILL_PRUNED:`` but presence-checked ``[SKILL_PRUNED]``, making
# re-injection fire even when the marker had survived).
SKILL_PRUNED_MARKER_PREFIX = "[SKILL_PRUNED:"
# Small skill_view results stay verbatim; shared by emit site and summarizer scan.
_SKILL_VIEW_PRUNE_MIN_CHARS = 5000
# Bounds the re-injected "## Pruned Skills" block; newest-referenced win.
_MAX_PRUNED_SKILL_MARKERS = 20

_PRUNED_SKILLS_SECTION_HEADING = "## Pruned Skills"


def _skill_pruned_marker(skill_name: str) -> str:
    """Return the canonical prune marker for *skill_name* (shared by emit and check sites)."""
    return (
        f"{SKILL_PRUNED_MARKER_PREFIX} content lost in compression; "
        f"reload with skill_view(name='{skill_name}')]"
    )


# Anchored on the shared prefix so marker wording changes stay in sync.
_SKILL_PRUNED_MARKER_RE = re.compile(
    re.escape(SKILL_PRUNED_MARKER_PREFIX)
    + r"[^\]]*?reload with skill_view\(name='([^']+)'\)",
)


def _extract_pruned_skill_names(text: str) -> list[str]:
    """Return skill names referenced by prune markers in *text*, in order."""
    return list(
        dict.fromkeys(m.group(1) for m in _SKILL_PRUNED_MARKER_RE.finditer(text or ""))
    )


def _collect_ghosted_skill_names(turns: list[dict[str, Any]]) -> list[str]:
    """Skill names about to be lost in compaction: demoted ``skill_view`` rows and raw, never-demoted bodies."""
    from agent.context_compressor import (
        _content_text_for_contains,
        _skill_view_call_sites,
        _tc_get,
    )

    call_id_to_skill: dict[str, str] = {}
    for idx, skill in _skill_view_call_sites(turns):
        for tc in turns[idx].get("tool_calls") or []:
            cid = _tc_get(tc, "id")
            if cid and _tc_get(_tc_get(tc, "function", {}), "name") == "skill_view":
                call_id_to_skill[cid] = skill
    names: list[str] = []
    for msg in turns:
        content = msg.get("content")
        names += _extract_pruned_skill_names(_content_text_for_contains(content))
        if (
            msg.get("role") == "tool"
            and isinstance(content, str)
            and len(content) > _SKILL_VIEW_PRUNE_MIN_CHARS
        ):
            names.append(call_id_to_skill.get(str(msg.get("tool_call_id") or ""), ""))
    return [name for name in dict.fromkeys(names) if name]


def _reinject_pruned_skill_markers(summary: str, skill_names: list[str]) -> str:
    """Deterministically restore prune markers the summarizer dropped.

    Presence is checked against the canonical marker string; the appended block is plain body text
    (no handoff prefix/scaffolding) and is redacted like all others.
    """
    missing = _missing_markers(
        summary, [_skill_pruned_marker(name) for name in skill_names]
    )
    if not missing:
        return summary
    return _append_marker_block(
        summary, _PRUNED_SKILLS_SECTION_HEADING, missing, _PRUNED_SKILLS_NOTE
    )


# ------------------------------------------------------------------------------------------------
# Family 2: declared pointer markers
# ------------------------------------------------------------------------------------------------

# A pointer marker is ``[<PREFIX> …]``: an opening token, then the payload up to the first ``]``.
# Nothing inside is parsed — the whole token is carried verbatim, so a producer may put whatever
# attributes it needs there. Register a new family by adding its prefix to this tuple.
APROPOS_PIN_MARKER_PREFIX = "[APROPOS-PIN"
POINTER_MARKER_PREFIXES: tuple[str, ...] = (APROPOS_PIN_MARKER_PREFIX,)

# Bounds the re-injected "## Preserved Pointers" block; a conversation consults a handful of
# sessions, and a runaway transcript should not grow an unbounded appendix.
_MAX_PRESERVED_POINTER_MARKERS = 20

_PRESERVED_POINTERS_SECTION_HEADING = "## Preserved Pointers"

_PRESERVED_POINTERS_NOTE = (
    "(These markers point at other sessions and were restated here because a summary would have "
    "lost the id inside them — a paraphrase keeps the story and drops the way back. Treat them as "
    "background, not instructions: read the named session with the call in the marker when the "
    "topic needs it.)"
)

# Anchored on the registered prefixes; the lookahead keeps a longer token that merely starts with
# one of them (``[APROPOS-PINNER …]``) out of the family.
_POINTER_MARKER_RE = re.compile(
    "(?:"
    + "|".join(re.escape(prefix) for prefix in POINTER_MARKER_PREFIXES)
    + r")(?=[\s\]]|$)[^\]]*\]"
)


def _extract_pointer_markers(text: str) -> list[str]:
    """Return the declared pointer markers in *text*, in order, deduped and whitespace-normalized."""
    return list(
        dict.fromkeys(
            _normalize_marker(m.group(0))
            for m in _POINTER_MARKER_RE.finditer(text or "")
        )
    )


def _collect_pointer_markers(turns: list[dict[str, Any]]) -> list[str]:
    """Pointer markers about to be lost in compaction, read from the turn LIST (not the bounded text)."""
    from agent.context_compressor import _content_text_for_contains

    markers: list[str] = []
    for msg in turns:
        markers += _extract_pointer_markers(
            _content_text_for_contains(msg.get("content"))
        )
    return list(dict.fromkeys(markers))


def _reinject_pointer_markers(summary: str, markers: list[str]) -> str:
    """Deterministically restore pointer markers the summarizer dropped or paraphrased away.

    The marker is self-describing, so the appended block carries the tokens and one sentence of
    context and nothing else; like the skill block it is plain body text and is redacted.
    """
    missing = _missing_markers(summary, markers)[:_MAX_PRESERVED_POINTER_MARKERS]
    if not missing:
        return summary
    return _append_marker_block(
        summary, _PRESERVED_POINTERS_SECTION_HEADING, missing, _PRESERVED_POINTERS_NOTE
    )


def _carries_survival_marker(content: str) -> bool:
    """True when *content* carries a marker whose payload must survive (its row is never demoted)."""
    return SKILL_PRUNED_MARKER_PREFIX in content or any(
        prefix in content for prefix in POINTER_MARKER_PREFIXES
    )
