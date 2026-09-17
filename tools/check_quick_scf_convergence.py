#!/usr/bin/env python3
"""Reject QUICK label runs containing failed or missing SCF evaluations.

Some QUICK builds print normal termination and return success after exhausting
SCF cycles. A finite binary label is therefore insufficient evidence of SCF
convergence. Check the complete appended QUICK log before using any labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def check_log(path: Path, expected_frames: int) -> dict[str, int | str]:
    """Require exactly one successful SCF convergence record per input frame."""
    if expected_frames <= 0:
        raise ValueError("expected frame count must be positive")
    converged = 0
    with path.open('r', encoding='utf-8', errors='replace') as source:
        for number, line in enumerate(source, start=1):
            if re.search(r'NO CONVERGENCE|RAN OUT OF CYCLES|SCF\s+FAILED', line, re.IGNORECASE):
                raise ValueError(f"QUICK SCF failure at line {number}; reject all labels from this run")
            if 'REACH CONVERGENCE AFTER' in line:
                converged += 1
    if converged != expected_frames:
        raise ValueError(f"QUICK SCF count {converged} differs from expected {expected_frames}")
    return {'status': 'passed', 'converged_frames': converged}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--expected-frames', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(check_log(args.log, args.expected_frames)))
