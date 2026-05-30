"""Shared utilities for workflow nodes."""

from __future__ import annotations

import json
import re

from deepworkflow.shared.types import JudgeFeedback, JudgeVerdict


def parse_judge_output(content: str) -> tuple[JudgeVerdict, list[JudgeFeedback]]:
    """Parse the judge agent's JSON output into verdict and feedbacks."""
    text = content.strip()

    # Extract JSON from fenced code block if present
    code_block_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return JudgeVerdict.ERROR, [
            JudgeFeedback(file="general", type=JudgeVerdict.ERROR, description="Judge produced invalid JSON output.")
        ]

    verdict_str = data.get("judge_verdict", "ERROR").upper()
    try:
        verdict = JudgeVerdict[verdict_str]
    except KeyError:
        verdict = JudgeVerdict.ERROR

    feedbacks = []
    for fb in data.get("judge_feedbacks", []):
        fb_type_str = fb.get("type", "ERROR").upper()
        try:
            fb_type = JudgeVerdict[fb_type_str]
        except KeyError:
            fb_type = JudgeVerdict.ERROR
        feedbacks.append(
            JudgeFeedback(
                file=fb.get("file", "unknown"),
                type=fb_type,
                description=fb.get("description", ""),
                proposal=fb.get("proposal", ""),
            )
        )

    return verdict, feedbacks
