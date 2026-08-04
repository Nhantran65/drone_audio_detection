#!/usr/bin/env python3
"""Apply a narrow compatibility fix for Chromium's classic scrollbar width."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


STYLE = """<style data-portable-overflow-compat>
.analytics-top-bar {
  left: 0 !important;
  right: 0 !important;
  width: 100% !important;
  margin-left: 0 !important;
  margin-right: 0 !important;
}
</style>
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path)
    args = parser.parse_args(argv)
    content = args.html.read_text(encoding="utf-8")
    if "data-portable-overflow-compat" not in content:
        if "</head>" not in content:
            raise ValueError(f"HTML has no closing head tag: {args.html}")
        content = content.replace("</head>", STYLE + "</head>", 1)
        args.html.write_text(content, encoding="utf-8")
    print(f"phase=portable_overflow_compat output={args.html}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
