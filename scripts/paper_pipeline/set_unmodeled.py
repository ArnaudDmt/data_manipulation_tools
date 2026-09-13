"""Write one scalar into both disturbance-wrench process variances of the retained tuning.

Edited as text so the file keeps its comments and layout; the observer YAML is read by mc_rtc and
by the pipeline alike, and a round trip through a YAML dumper would lose both.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as m

value = sys.argv[1]
target = m.WORK / "configs/clean/MCKineticsObserver.yaml"
text = target.read_text()
for key in ("unmodeledForceProcessVariance", "unmodeledTorqueProcessVariance"):
    text, count = re.subn(rf"(?m)^(\s*{key}:\s*)\[[^\]]*\]", rf"\g<1>[{value}, {value}, {value}]", text)
    if count != 1:
        raise SystemExit(f"{key} trouve {count} fois dans {target}, attendu 1")
target.write_text(text)
print(f"{target.relative_to(m.ROOT)}: unmodeled wrench process = {value}")
