"""Fold freshly measured macros into metrics_results.tex.

Values are replaced in place so the file keeps its layout and its commented-out history; macros
that exist in the new set but not in the file are appended, and macros the file defines that we
did not remeasure are listed on stderr so nothing is silently left stale.
"""
import re
import sys
from pathlib import Path

PAPER = Path("/home/arnaud/Documents/ResearchNotes/Topics/KineticsObserver/Papers/"
             "IJRR/Third_submission/Paper/metrics_results.tex")
PATTERN = re.compile(r"^(\s*)\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}(.*)$")


def read_macros(paths):
    values = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            match = PATTERN.match(line)
            if match:
                values[match.group(2)] = match.group(3)
    return values


def main(sources):
    new = read_macros(sources)
    used, output = set(), []
    for line in PAPER.read_text().splitlines():
        match = PATTERN.match(line)
        if match and match.group(2) in new:
            name = match.group(2)
            used.add(name)
            output.append(f"{match.group(1)}\\newcommand{{\\{name}}}{{{new[name]}}}{match.group(4)}")
        else:
            output.append(line)
    missing = [name for name in new if name not in used]
    if missing:
        output.append("")
        output.append("% Measured on the retained configuration, not present in the previous file.")
        output.extend(f"\\newcommand{{\\{name}}}{{{new[name]}}}" for name in sorted(missing))
    PAPER.write_text("\n".join(output) + "\n")
    print(f"replaced {len(used)} macros, appended {len(missing)}")
    stale = sorted({match.group(2) for line in PAPER.read_text().splitlines()
                    if (match := PATTERN.match(line)) and match.group(2) not in new})
    print(f"left untouched ({len(stale)}):", file=sys.stderr)
    for name in stale:
        print(f"  {name}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
