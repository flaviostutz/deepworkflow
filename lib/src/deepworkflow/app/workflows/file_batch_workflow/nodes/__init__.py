"""Shared utilities for workflow nodes."""

from __future__ import annotations

import json
import re

from deepworkflow.shared.types import EvaluateFeedback, JudgeFinding, JudgeLevel, JudgeVerdict

_WORD_LIMIT_TITLE = 10
_WORD_LIMIT_REASON = 30
_CHAR_LIMIT_DETAILS = 200
_CHAR_LIMIT_FIX = 100


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


def _truncate_chars(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def parse_evaluate_output(content: str) -> tuple[JudgeLevel, list[EvaluateFeedback]]:
    """Parse the evaluate agent's JSON output into verdict and feedbacks."""
    text = content.strip()

    # Extract JSON from fenced code block if present
    code_block_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return JudgeLevel.ERROR, [
            EvaluateFeedback(file="general", type=JudgeLevel.ERROR, description="Judge produced invalid JSON output.")
        ]

    verdict_str = data.get("evaluate_quality_verdict", "ERROR").upper()
    try:
        verdict = JudgeLevel[verdict_str]
    except KeyError:
        verdict = JudgeLevel.ERROR

    feedbacks = []
    for fb in data.get("evaluate_quality_feedbacks", []):
        fb_type_str = fb.get("type", "ERROR").upper()
        try:
            fb_type = JudgeLevel[fb_type_str]
        except KeyError:
            fb_type = JudgeLevel.ERROR
        feedbacks.append(
            EvaluateFeedback(
                file=fb.get("file", "unknown"),
                type=fb_type,
                description=fb.get("description", ""),
                proposal=fb.get("proposal", ""),
            )
        )

    return verdict, feedbacks


def parse_evaluate_verdict(content: str) -> JudgeVerdict:
    """Parse a judge node's JSON output into a JudgeVerdict (rule 13 schema).

    Expected JSON::

        {
            "verdict": "OK|INFO|WARNING|ERROR",
            "findings": [
                {
                    "level": "OK|INFO|WARNING|ERROR",
                    "title": "<10 words>",
                    "reason": "...",
                    "details": "...",
                    "fix": "...",
                }
            ],
        }

    On any parse failure returns a single ERROR finding so routing treats the
    output as needing attention rather than silently converging.
    """
    text = content.strip()

    code_block_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return JudgeVerdict(
            verdict=JudgeLevel.ERROR,
            findings=[JudgeFinding(level=JudgeLevel.ERROR, title="Judge produced invalid JSON output")],
        )

    verdict_str = data.get("verdict", "ERROR").upper()
    try:
        verdict = JudgeLevel[verdict_str]
    except KeyError:
        verdict = JudgeLevel.ERROR

    findings: list[JudgeFinding] = []
    for fb in data.get("findings", []):
        level_str = fb.get("level", "ERROR").upper()
        try:
            level = JudgeLevel[level_str]
        except KeyError:
            level = JudgeLevel.ERROR
        findings.append(
            JudgeFinding(
                level=level,
                title=fb.get("title", ""),
                reason=fb.get("reason", ""),
                details=fb.get("details", ""),
                fix=fb.get("fix", ""),
            )
        )

    if not findings:
        findings = [JudgeFinding(level=verdict, title="No findings provided")]

    return JudgeVerdict(verdict=verdict, findings=findings)
