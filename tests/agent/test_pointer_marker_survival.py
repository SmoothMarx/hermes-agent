"""Declared pointer markers survive compaction.

A conversation can be told, in its opening row, which other sessions it continues: an "Apropos"
seed carries ``[APROPOS-PIN parent=<session id> profile=<profile>]`` (a group seed carries
``group="…"`` instead). Compaction summarises that row away, and the summariser paraphrases the
token into prose — "the parent session" — with the id gone, which breaks the chain back to the
earlier session silently: the summary still reads correctly.

Same defense as the ghost-skill marker (``test_ghost_skill_pruning``): the declared tokens are
extracted from the raw turn list BEFORE the summary call and re-injected after it whenever the
summarizer dropped them. Markers are registered by prefix, so the family is not tied to one plugin:
a producer that seeds context rows adds its prefix and nothing else.

The integration test is the invariant: a marker that no other channel can carry (a tool row) is
gone from a summarizer-mocked ``compress()`` unless re-injection puts it back.
"""

from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor
from agent.context_compressor_markers import (
    _MAX_PRESERVED_POINTER_MARKERS,
    _PRESERVED_POINTERS_SECTION_HEADING,
    _carries_survival_marker,
    _collect_pointer_markers,
    _extract_pointer_markers,
    _reinject_pointer_markers,
    _reinject_pruned_skill_markers,
    _skill_pruned_marker,
)

PIN = "[APROPOS-PIN parent=20261010_195437_513a83 profile=default]"
GROUP_PIN = '[APROPOS-PIN group="Novi Sad trip" profile=default members=8]'


def _make_compressor(**overrides):
    kwargs = dict(
        model="test/model",
        quiet_mode=True,
        protect_first_n=1,
        protect_last_n=2,
    )
    kwargs.update(overrides)
    with patch(
        "agent.context_compressor.get_model_context_length", return_value=100000
    ):
        return ContextCompressor(**kwargs)


class TestExtractPointerMarkers:
    def test_reads_single_and_group_pins_in_order(self):
        text = f"seed\n{PIN}\ntalk\n{GROUP_PIN}\n"
        assert _extract_pointer_markers(text) == [PIN, GROUP_PIN]

    def test_dedupes_repeated_pins(self):
        assert _extract_pointer_markers(f"{PIN}\n{PIN}") == [PIN]

    def test_copies_the_token_verbatim_without_parsing_it(self):
        """Whatever the producer put between the prefix and ``]`` is carried as-is."""
        odd = '[APROPOS-PIN a=1 b=two words c="quoted"]'
        assert _extract_pointer_markers(odd) == [odd]

    def test_ignores_a_longer_token_that_merely_starts_with_the_prefix(self):
        assert _extract_pointer_markers("[APROPOS-PINNER nope]") == []

    def test_bare_prefix_is_a_marker(self):
        assert _extract_pointer_markers("[APROPOS-PIN]") == ["[APROPOS-PIN]"]

    def test_empty_input_is_empty(self):
        assert _extract_pointer_markers("") == []
        assert _extract_pointer_markers(None) == []


class TestCollectPointerMarkers:
    def test_reads_the_turn_list_including_content_blocks(self):
        turns = [
            {"role": "user", "content": f"opening row\n{PIN}"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"noted\n{GROUP_PIN}"}],
            },
            {"role": "assistant", "content": "no marker here"},
        ]
        assert _collect_pointer_markers(turns) == [PIN, GROUP_PIN]

    def test_collects_nothing_from_a_transcript_without_pins(self):
        assert _collect_pointer_markers([{"role": "user", "content": "plain"}]) == []


class TestReinjectPointerMarkers:
    def test_restores_a_marker_the_summarizer_paraphrased_away(self):
        summary = "## Goal\nThe user continued earlier work and asked for a report."
        out = _reinject_pointer_markers(summary, [PIN])
        assert PIN in out
        assert _PRESERVED_POINTERS_SECTION_HEADING in out
        assert out.startswith(summary)

    def test_does_not_duplicate_a_marker_that_survived(self):
        out = _reinject_pointer_markers(f"body\n{PIN}", [PIN])
        assert out.count(PIN) == 1

    def test_reinjects_only_the_missing_markers(self):
        out = _reinject_pointer_markers(f"body\n{PIN}", [PIN, GROUP_PIN])
        assert out.count(PIN) == 1
        assert out.count(GROUP_PIN) == 1

    def test_a_rewrapped_marker_counts_as_survived(self):
        rewrapped = PIN.replace(" ", "\n  ")
        out = _reinject_pointer_markers(f"body\n{rewrapped}", [PIN])
        assert out == f"body\n{rewrapped}"

    def test_bounds_its_own_block(self):
        markers = [
            f"[APROPOS-PIN parent=sess-{i}]"
            for i in range(_MAX_PRESERVED_POINTER_MARKERS + 5)
        ]
        out = _reinject_pointer_markers("body", markers)
        assert out.count("[APROPOS-PIN parent=") == _MAX_PRESERVED_POINTER_MARKERS

    def test_appended_block_is_redacted(self, monkeypatch):
        import agent.redact as redact_mod

        monkeypatch.setattr(redact_mod, "_REDACT_ENABLED", False, raising=False)
        secret = "ghp_" + "a1B2" * 6
        out = _reinject_pointer_markers("body", [f"[APROPOS-PIN parent={secret}]"])
        assert secret not in out

    def test_families_inject_independently(self):
        """A summary that already carries a skill marker gains only the pointer block."""
        with_skill = _reinject_pruned_skill_markers("## Goal\nwork", ["pdf"])
        out = _reinject_pointer_markers(with_skill, [PIN])
        assert out.count(_skill_pruned_marker("pdf")) == 1
        assert out.count(_PRESERVED_POINTERS_SECTION_HEADING) == 1


class TestSurvivalMarkerGuard:
    def test_recognises_both_families(self):
        assert _carries_survival_marker(f"row {PIN}")
        assert _carries_survival_marker("row " + _skill_pruned_marker("pdf"))
        assert not _carries_survival_marker("plain tool output")


class TestPointerMarkerSurvivesRealCompress:
    """The invariant: the payload is in the summary of a real ``compress()``.

    A tool row is used deliberately — the verbatim-user-row channel cannot carry it, so only
    re-injection can put the marker back.
    """

    def _mock_response(self, text):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = text
        return response

    def _summary_text_of(self, result):
        for msg in result:
            if ContextCompressor.classify_summary_content(msg.get("content")):
                return msg["content"]
        raise AssertionError(f"no summary message found in {result!r}")

    def _transcript_with_pin_row(self):
        return [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "continue that other session for me"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_pin",
                        "type": "function",
                        "function": {
                            "name": "session_search",
                            "arguments": '{"session_id":"20261010_195437_513a83"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_pin", "content": f"1 result\n{PIN}"},
            {"role": "assistant", "content": "Continuing from there."},
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "more work " + "z" * 500},
            {"role": "user", "content": "latest ask"},
            {"role": "assistant", "content": "ack"},
        ]

    def test_paraphrased_pin_is_restored_into_the_summary(self):
        c = _make_compressor(protect_first_n=1, protect_last_n=2)
        # The summarizer keeps the story and drops the id — the measured failure.
        drop_response = self._mock_response(
            "## Goal\nContinue the earlier session and finish the report.\n\n"
            "## Completed Actions\n1. Read the earlier session [tool: session_search]"
        )
        with (
            patch.object(c, "_find_tail_cut_by_tokens", return_value=7),
            patch(
                "agent.context_compressor.call_llm", return_value=drop_response
            ) as mock_call,
        ):
            result = c.compress(self._transcript_with_pin_row(), force=True)
        assert mock_call.called
        summary_text = self._summary_text_of(result)
        assert PIN in summary_text
        # Stored iterative-update state carries it too, so the next round cannot lose it.
        assert PIN in c._previous_summary
