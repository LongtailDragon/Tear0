from __future__ import annotations

import sys
import textwrap
from typing import TextIO


def format_ascii_box(title: str, body: str, *, width: int = 88) -> str:
    width = max(24, width)
    inner = width - 2
    border = "+" + "-" * inner + "+"
    lines = [border, f"| {title[: inner - 2]:<{inner - 2}} |", border]
    text = body.strip() or " "
    for raw_line in text.splitlines() or [""]:
        wrapped = textwrap.wrap(raw_line, width=inner - 2, replace_whitespace=False) or [""]
        for line in wrapped:
            lines.append(f"| {line:<{inner - 2}} |")
    lines.append(border)
    return "\n" + "\n".join(lines) + "\n"


def print_bullet(*values: object, sep: str = " ", file: TextIO | None = None) -> None:
    print("> " + sep.join(str(value) for value in values), file=file or sys.stdout)
