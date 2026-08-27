#!/usr/bin/env python3
"""Budget ledger and ticket enforcement for nanoWM training runs.

Rules (see kickoff prompt §3):
  - Every training run needs a Budget-Ticket approved in advance.
  - Every run appends one line to budget/ledger.jsonl on completion.
  - This script refuses to authorize a ticket that would exceed remaining
    budget for its milestone, or remaining total budget.
  - Overrun on a milestone is only allowed to be covered by pulling M4's
    unused allocation, never by inflating total_hours.

Usage:
    python scripts/budget.py status
    python scripts/budget.py check-ticket --milestone M3_ablations_15M --hours 0.6
    python scripts/budget.py log --run-id r001 --milestone M3_ablations_15M \\
        --purpose "REPA ablation seed 0" --result "loss=1.23" \\
        --seed 0 --git-sha abc123 --start 2026-08-27T10:00:00 \\
        --end 2026-08-27T10:36:00 --gpu-seconds 4320
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = ROOT / "budget" / "ledger.jsonl"
ALLOCATION_PATH = ROOT / "budget" / "allocation.yaml"


def load_allocation():
    with open(ALLOCATION_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_ledger():
    if not LEDGER_PATH.exists() or LEDGER_PATH.stat().st_size == 0:
        return []
    entries = []
    with open(LEDGER_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def spent_hours_by_milestone(entries):
    spent = {}
    for e in entries:
        m = e["milestone"]
        spent[m] = spent.get(m, 0.0) + e["gpu_seconds"] / 3600.0
    return spent


def cmd_status(args):
    alloc = load_allocation()
    entries = load_ledger()
    spent = spent_hours_by_milestone(entries)
    total_budget = alloc["total_hours"]
    total_spent = sum(spent.values())

    print(f"nanoWM Budget Ledger — {LEDGER_PATH}")
    print(f"{'Milestone':<22} {'Budget(h)':>10} {'Spent(h)':>10} {'Remaining(h)':>13}")
    for name, m in alloc["milestones"].items():
        b = m["budget_hours"]
        s = spent.get(name, 0.0)
        print(f"{name:<22} {b:>10.2f} {s:>10.3f} {b - s:>13.3f}")
    print("-" * 60)
    print(f"{'TOTAL':<22} {total_budget:>10.2f} {total_spent:>10.3f} {total_budget - total_spent:>13.3f}")


def cmd_check_ticket(args):
    alloc = load_allocation()
    entries = load_ledger()
    spent = spent_hours_by_milestone(entries)
    total_spent = sum(spent.values())
    total_budget = alloc["total_hours"]

    if args.milestone not in alloc["milestones"]:
        print(f"REJECTED: unknown milestone '{args.milestone}'", file=sys.stderr)
        sys.exit(1)

    m_budget = alloc["milestones"][args.milestone]["budget_hours"]
    m_spent = spent.get(args.milestone, 0.0)
    m_remaining = m_budget - m_spent
    total_remaining = total_budget - total_spent

    if args.hours > total_remaining:
        print(
            f"REJECTED: ticket requests {args.hours:.3f}h but only "
            f"{total_remaining:.3f}h remain in total project budget.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.hours > m_remaining:
        print(
            f"REJECTED: ticket requests {args.hours:.3f}h but only "
            f"{m_remaining:.3f}h remain for milestone '{args.milestone}'. "
            f"Overrun must be requested explicitly by borrowing from M4, "
            f"not auto-approved.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"APPROVED: {args.hours:.3f}h for '{args.milestone}' "
        f"(milestone remaining after: {m_remaining - args.hours:.3f}h, "
        f"total remaining after: {total_remaining - args.hours:.3f}h)"
    )


def cmd_log(args):
    entry = {
        "run_id": args.run_id,
        "start": args.start,
        "end": args.end,
        "gpu_seconds": args.gpu_seconds,
        "milestone": args.milestone,
        "purpose": args.purpose,
        "result": args.result,
        "seed": args.seed,
        "git_sha": args.git_sha,
    }
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"Logged run '{args.run_id}' ({args.gpu_seconds / 3600.0:.3f} GPU-hours) to ledger.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Print remaining budget per milestone and total")

    p_check = sub.add_parser("check-ticket", help="Validate a proposed run ticket against remaining budget")
    p_check.add_argument("--milestone", required=True)
    p_check.add_argument("--hours", required=True, type=float, help="Planned wall-clock hours for this run")

    p_log = sub.add_parser("log", help="Append a completed run to the ledger")
    p_log.add_argument("--run-id", required=True)
    p_log.add_argument("--start", required=True)
    p_log.add_argument("--end", required=True)
    p_log.add_argument("--gpu-seconds", required=True, type=float)
    p_log.add_argument("--milestone", required=True)
    p_log.add_argument("--purpose", required=True)
    p_log.add_argument("--result", required=True)
    p_log.add_argument("--seed", required=True, type=int)
    p_log.add_argument("--git-sha", required=True)

    args = p.parse_args()
    {"status": cmd_status, "check-ticket": cmd_check_ticket, "log": cmd_log}[args.command](args)


if __name__ == "__main__":
    main()
