"""One colour lookup for every paper figure.

generate_metrics_plots computes the palette once, keyed by observer abbreviation ("KO") with
components in 0-1, and hands it to each plotting script. These scripts were written against the
long names ("KineticsObserver") and 0-255 components, so passing the shared palette straight in
either raised KeyError or produced near-black curves. Resolving both conventions here lets every
figure share one palette instead of falling back to its own defaults.
"""

ABBREVIATIONS = {
    "KineticsObserver": "KO",
    "KO": "KineticsObserver",
    "KOWithoutWrenchSensors": "KO_WWS",
    "KO_WWS": "KOWithoutWrenchSensors",
    "Controller": "Control",
    "Control": "Controller",
}


def resolve(colors, name):
    """Colour of one estimator as a 0-255 (r, g, b) triple, under either naming convention."""
    for key in (name, ABBREVIATIONS.get(name)):
        if key is not None and key in colors:
            rgb = colors[key][:3]
            return tuple(round(c * 255) if max(rgb) <= 1.0 else c for c in rgb)
    raise KeyError(f"aucune couleur pour {name!r} parmi {sorted(colors)}")


def rgba(colors, name, alpha=1):
    r, g, b = resolve(colors, name)
    return f"rgba({r}, {g}, {b}, {alpha})"
