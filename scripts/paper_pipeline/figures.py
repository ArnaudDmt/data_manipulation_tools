"""Regenerate every data figure, install it in the paper, and export PNG and PDF for all of them.

Each figure script writes into PAPER_FIG_OUT (see figure_harness); this collects what it produced,
copies it next to main.tex under the name the LaTeX expects, and fills the export folder with both
formats -- including the figures no script here produces, so the folder is always complete.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as m

HERE = Path(__file__).resolve().parent
OUT = m.WORK / "figures"
PYTHON = m.ROOT / "env/bin/python"
# The paper includes this one as a bitmap; everything else as vector.
AS_PNG = {"multicontact-odom-traj"}


def regenerate(selection):
    produced, failed = {}, []
    environment = dict(os.environ, PAPER_FIG_OUT=str(OUT))
    for name, (command, filename) in m.FIGURES.items():
        if selection not in ("all", name):
            continue
        script, *arguments = command.split()
        print(f"--- {name}  ({command})")
        # plotMultipleTrajs names its output after the project's group, not after the figure, so
        # the manifest entry is a pattern. Clear what matches first: a leftover from another
        # project would otherwise be collected as this one's.
        for stale in OUT.glob(filename):
            stale.unlink()
        result = subprocess.run([str(PYTHON), str(HERE / script), *arguments],
                                cwd=m.ROOT, env=environment,
                                capture_output=True, text=True)
        matches = sorted(OUT.glob(filename), key=lambda q: q.stat().st_mtime)
        source = matches[-1] if matches else OUT / filename
        if result.returncode != 0 or not matches:
            failed.append(name)
            print(f"    ECHEC (code {result.returncode})")
            print("    " + "\n    ".join(result.stderr.strip().splitlines()[-6:]))
            continue
        # Several scripts share a generic output name, so take it away immediately.
        target = OUT / f"{name}.pdf"
        source.replace(target)
        produced[name] = target
        print(f"    -> {target.name}")
    return produced, failed


def install(produced):
    for name, path in produced.items():
        if name in AS_PNG:
            # Keep both: the paper includes the bitmap, the export folder wants the vector too.
            png = m.PAPER / f"{name}.png"
            subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile",
                            str(path), str(png.with_suffix(""))], check=True)
        shutil.copy(path, m.PAPER / f"{name}.pdf")
        print(f"installe {name}")


def export():
    m.EXPORT.mkdir(parents=True, exist_ok=True)
    missing = []
    for name in list(m.FIGURES) + m.STATIC_FIGURES:
        pdf, png = m.PAPER / f"{name}.pdf", m.PAPER / f"{name}.png"
        if pdf.exists():
            shutil.copy(pdf, m.EXPORT / f"{name}.pdf")
            subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile",
                            str(pdf), str(m.EXPORT / name)], check=True)
        elif png.exists():
            shutil.copy(png, m.EXPORT / f"{name}.png")
            subprocess.run(["convert", str(png), str(m.EXPORT / f"{name}.pdf")], check=True)
        else:
            missing.append(name)
            continue
    both = sorted({p.stem for p in m.EXPORT.glob("*.pdf")}
                  & {p.stem for p in m.EXPORT.glob("*.png")})
    print(f"\n{len(both)} figures en PNG et PDF dans {m.EXPORT}")
    for name in missing:
        print(f"  ABSENTE du papier: {name}", file=sys.stderr)


if __name__ == "__main__":
    arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
    selection = arguments[0] if arguments else "all"
    OUT.mkdir(parents=True, exist_ok=True)
    produced, failed = ({}, []) if "--export-only" in sys.argv else regenerate(selection)
    if produced and "--no-install" not in sys.argv:
        install(produced)
    export()
    if failed:
        print(f"\n{len(failed)} figures en echec: {', '.join(failed)}", file=sys.stderr)
        raise SystemExit(1)
