"""Concatenates the vendored/ported modules into ONE file for a notebook %%writefile cell (no package imports at run time)."""
import re, sys
parts = []
for f in ("ledger.py", "fate_arc.py", "select_pool.py", "entry.py"):
    s = open(f"fate_ssm/{f}").read()
    s = re.sub(r"^from \.ledger import Ledger\s*$", "", s, flags=re.M)
    s = re.sub(r"^from __future__ import annotations\s*$", "", s, flags=re.M)
    s = s.replace("from . import fate_arc", "")
    parts.append(f"\n# ===== {f} =====\n" + s)
hdr = "from __future__ import annotations\nimport itertools, json, math, sys, time\nfrom typing import Callable, Iterable, Optional\nimport numpy as np\n"
open("fate_ssm_runtime.py", "w").write(hdr + "".join(parts))
print("wrote fate_ssm_runtime.py", sum(len(p) for p in parts), "chars")
