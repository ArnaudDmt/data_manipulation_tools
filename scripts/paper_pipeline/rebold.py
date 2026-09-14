"""Move the \textbf markers of the result tables onto the estimator that actually wins.

The tables mark the best value of each metric in bold by hand, so the markers stop matching as
soon as the numbers change. Every mean cell of a comparison column is unwrapped, then the
minimum is wrapped again -- ties included, which is how the previous version marked them.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

PAPER = Path("/home/arnaud/Documents/ResearchNotes/Topics/KineticsObserver/Papers/"
             "IJRR/Third_submission/Paper")
MAIN = PAPER / "main.tex"
MACROS = PAPER / "metrics_results.tex"

# The comparison tables; the flexibility table compares tunings, not estimators, so it is left
# alone -- its rows are three settings of the same estimator and none of them is "best".
COMPARISONS = {
    "Multicontact": ["Kineticsobserver", "KoZpc", "Kowithoutwrenchsensors", "Hartley", "Tilt"],
    "Flatodometry": ["Kineticsobserver", "KoZpc", "Kowithoutwrenchsensors", "Hartley"],
    "Slippingodometry": ["Kineticsobserver", "KoZpc", "Kowithoutwrenchsensors", "Hartley", "Tilt"],
    "Longwalk": ["Kineticsobserver", "KoZpc", "Kowithoutwrenchsensors", "Hartley", "Tilt"],
}
COLUMNS = [("Relerror", name) for name in ("Transxy", "Transz", "Tilt", "Yaw")] + \
          [("Velerror", name) for name in ("EstimateXy", "EstimateZ")]

CALL = r"\getErrorResult{%s}{%s}{%s}{%s}{Meanabs}"


def values():
    pattern = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}")
    found = {}
    for line in MACROS.read_text().splitlines():
        if line.lstrip().startswith("%"):
            continue
        match = pattern.search(line)
        if match:
            found[match.group(1)] = float(match.group(2))
    return found


def main():
    numbers = values()
    text = MAIN.read_text()
    winners = defaultdict(set)
    for category, estimators in COMPARISONS.items():
        for kind, metric in COLUMNS:
            scores = {}
            for estimator in estimators:
                key = f"{category}{estimator}{kind}{metric}Meanabs"
                if key in numbers:
                    scores[estimator] = numbers[key]
            if not scores:
                continue
            best = min(scores.values())
            for estimator, score in scores.items():
                if score == best:
                    winners[(category, kind, metric)].add(estimator)

    # VALINOR's macros were not remeasured by the rebuild: they still carry August values. A cell
    # it "wins" would be a comparison between two different runs, so those are reported rather
    # than trusted silently.
    stale = {"Tilt"}
    suspect = sorted((f"{category} {kind}/{metric}", ", ".join(sorted(best)))
                     for (category, kind, metric), best in winners.items()
                     if best & stale)
    print(f"{'categorie':20s} {'colonne':22s} gagnant")
    for (category, kind, metric), best in sorted(winners.items()):
        print(f"  {category:18s} {kind + '/' + metric:22s} {', '.join(sorted(best))}")
    if suspect:
        print(f"\nATTENTION {len(suspect)} cellules gagnees par VALINOR, dont les macros datent "
              f"d'aout et n'ont pas ete recalculees:", file=sys.stderr)
        for cell, who in suspect:
            print(f"  {cell}", file=sys.stderr)

    changed = 0
    for (category, kind, metric), best in winners.items():
        for estimator in COMPARISONS[category]:
            call = CALL % (category, estimator, kind, metric)
            bold = "\\textbf{" + call + "}"
            if estimator in best:
                if bold not in text and call in text:
                    text = text.replace(call, bold)
                    changed += 1
            elif bold in text:
                text = text.replace(bold, call)
                changed += 1
    backup = MAIN.with_suffix(MAIN.suffix + ".before-rebold-" + __import__("time").strftime("%Y%m%d-%H%M"))
    backup.write_text(MAIN.read_text())
    MAIN.write_text(text)
    print(f"{changed} bold markers moved")


if __name__ == "__main__":
    main()
