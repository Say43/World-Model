#!/usr/bin/env python3
"""Hard hardware gate: refuse to run on anything but the expected GPUs.

`machine_shape: "NvidiaTeslaT4"` in kernel-metadata.json *requests* 2xT4, but
nothing guarantees it -- a kernel can come up on a single P100 (the "Gpu"
shape), and Kaggle's allocation is not something the notebook controls. Every
number this project produces is hardware-specific: step time, MFU, VRAM
headroom, whether fp16 is worth using at all. A profiling run that silently
lands on a P100 does not just produce wrong numbers, it produces
*plausible-looking* wrong numbers and spends weekly quota to do it.

So this fails loudly before anything expensive starts.

Usage:
    python scripts/preflight_gpu.py                 # expect 2x T4
    python scripts/preflight_gpu.py --min-count 1   # allow a single GPU
    python scripts/preflight_gpu.py --allow-any     # report only, never fail
"""
import argparse
import sys

EXPECTED_CC = (7, 5)  # Turing sm75: no bf16, no FlashAttention-2


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-count", type=int, default=2, help="Minimum number of GPUs required (default: 2)")
    p.add_argument("--name-contains", default="T4", help="Substring every GPU name must contain (default: T4)")
    p.add_argument("--allow-any", action="store_true", help="Report hardware but never fail")
    args = p.parse_args(argv)

    try:
        import torch
    except ImportError:
        print("PREFLIGHT FAIL: torch is not importable", file=sys.stderr)
        return 1

    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("PREFLIGHT FAIL: no CUDA device visible", file=sys.stderr)
        return 0 if args.allow_any else 1

    count = torch.cuda.device_count()
    problems = []
    print(f"Visible GPUs: {count}")
    for i in range(count):
        name = torch.cuda.get_device_name(i)
        cc = torch.cuda.get_device_capability(i)
        total_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  [{i}] {name} | compute capability {cc[0]}.{cc[1]} | {total_gb:.1f} GB")
        if args.name_contains and args.name_contains.lower() not in name.lower():
            problems.append(f"GPU {i} is '{name}', expected a name containing '{args.name_contains}'")
        if cc != EXPECTED_CC:
            problems.append(
                f"GPU {i} has compute capability {cc[0]}.{cc[1]}, expected "
                f"{EXPECTED_CC[0]}.{EXPECTED_CC[1]} (sm75/Turing)"
            )

    if count < args.min_count:
        problems.append(f"found {count} GPU(s), need at least {args.min_count}")

    if problems:
        print("\nPREFLIGHT FAIL:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        if args.allow_any:
            print("(--allow-any set: continuing anyway)", file=sys.stderr)
            return 0
        print(
            "\nThis run was configured for 2x Tesla T4 (sm75). Numbers measured on\n"
            "different hardware are not comparable and must not enter the ledger.\n"
            "Check kernel-metadata.json has machine_shape: \"NvidiaTeslaT4\".",
            file=sys.stderr,
        )
        return 1

    print(f"\nPREFLIGHT OK: {count}x {args.name_contains}, sm{EXPECTED_CC[0]}{EXPECTED_CC[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
