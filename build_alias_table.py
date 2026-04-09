"""
PropEdge V16.0 — build_alias_table.py
──────────────────────────────────────────────────────────────────────────────
Run ONCE on your machine before generating season JSONs.

This script:
  1. Reads all player names from both props Excel files and both game log CSVs
  2. For every props player name, tries all resolution strategies against
     the game log canonical names (exact, suffix-strip, prefix, fuzzy)
  3. Prints every UNRESOLVED name (genuine mismatch needing a manual alias)
  4. Prints every RESOLVED name showing which strategy matched
  5. Outputs a ready-to-paste PLAYER_ALIASES dict for player_name_aliases.py

Usage:
    python3 build_alias_table.py

Output:
    - Console report of all matches/mismatches
    - alias_audit.txt — full report saved to disk
    - alias_table_generated.py — copy-paste alias dict for player_name_aliases.py
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import re, sys, unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from config import FILE_GL_2425, FILE_GL_2526, FILE_PROPS, FILE_SEASON_2425

# ── Normalisation helpers ─────────────────────────────────────────────────────
def norm(s: str) -> str:
    n = unicodedata.normalize("NFD", str(s))
    n = "".join(c for c in n if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]", "", n.lower()).strip()

def norm_strip(s: str) -> str:
    """norm() then strip Jr/Sr/II/III/IV suffix."""
    return re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv)$", "", norm(s), flags=re.IGNORECASE).strip()

def token_overlap(a: str, b: str) -> float:
    ta = set(norm_strip(a).split())
    tb = set(norm_strip(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


# ── Load game log canonical names ─────────────────────────────────────────────
def load_gl_players() -> tuple[set[str], set[str]]:
    gl25, gl26 = set(), set()
    if FILE_GL_2425.exists():
        df = pd.read_csv(FILE_GL_2425, low_memory=False)
        gl25 = set(df["PLAYER_NAME"].dropna().unique())
        print(f"  2024-25 game log: {len(gl25)} unique players")
    else:
        print(f"  ⚠ {FILE_GL_2425} not found")
    if FILE_GL_2526.exists():
        df = pd.read_csv(FILE_GL_2526, low_memory=False)
        gl26 = set(df["PLAYER_NAME"].dropna().unique())
        print(f"  2025-26 game log: {len(gl26)} unique players")
    else:
        print(f"  ⚠ {FILE_GL_2526} not found")
    return gl25, gl26


# ── Load props player names ───────────────────────────────────────────────────
def load_props_players() -> tuple[set[str], set[str]]:
    p2425, p2526 = set(), set()

    # 2024-25 props — separate file uploaded by Salman
    # Check common locations
    for candidate in [
        ROOT / "source-files" / "PropEdge_2024_25_Match_and_Player_Prop_lines.xlsx",
        Path.home() / "Downloads" / "PropEdge_2024_25_Match_and_Player_Prop_lines.xlsx",
        Path.home() / "Documents" / "PropEdge_2024_25_Match_and_Player_Prop_lines.xlsx",
    ]:
        if candidate.exists():
            try:
                xl = pd.read_excel(candidate, sheet_name="Player_Points_Props")
                p2425 = set(xl["Player"].dropna().unique())
                print(f"  2024-25 props: {len(p2425)} unique players (from {candidate.name})")
            except Exception as e:
                print(f"  ⚠ 2024-25 props read error: {e}")
            break
    else:
        print(f"  ⚠ 2024-25 props file not found — place it in source-files/")

    # 2025-26 props — main Excel file
    if FILE_PROPS.exists():
        try:
            xl = pd.read_excel(FILE_PROPS, sheet_name="Player_Points_Props")
            xl["Date"] = pd.to_datetime(xl["Date"])
            xl26 = xl[xl["Date"] >= pd.Timestamp("2025-10-01")]
            p2526 = set(xl26["Player"].dropna().unique())
            print(f"  2025-26 props: {len(p2526)} unique players")
        except Exception as e:
            print(f"  ⚠ 2025-26 props read error: {e}")
    else:
        print(f"  ⚠ {FILE_PROPS} not found")

    return p2425, p2526


# ── Resolution engine ─────────────────────────────────────────────────────────
def resolve(prop_name: str, gl_players: set[str]) -> tuple[str | None, str]:
    """
    Try to match prop_name to a canonical game log player name.
    Returns (canonical_name, strategy) or (None, 'UNRESOLVED').
    """
    nmap = {norm(p): p for p in gl_players}
    nmap_strip = {norm_strip(p): p for p in gl_players}

    key      = norm(prop_name)
    key_strip = norm_strip(prop_name)

    # 1. Exact normalised match
    if key in nmap:
        return nmap[key], "EXACT"

    # 2. Suffix-stripped exact
    if key_strip in nmap_strip:
        return nmap_strip[key_strip], "SUFFIX_STRIP"

    # 3. Prefix-8 match (original engine fallback)
    if len(key) >= 6:
        for k, v in nmap.items():
            if len(k) >= 6 and key[:8] == k[:8]:
                return v, "PREFIX8"

    # 4. Suffix-stripped prefix-8
    if len(key_strip) >= 6:
        for k, v in nmap_strip.items():
            if len(k) >= 6 and key_strip[:8] == k[:8]:
                return v, "STRIP+PREFIX8"

    # 5. Token overlap ≥ 80%
    best_score, best_match = 0.0, None
    for p in gl_players:
        score = token_overlap(prop_name, p)
        if score > best_score:
            best_score, best_match = score, p
    if best_score >= 0.80:
        return best_match, f"TOKEN_OVERLAP({best_score:.0%})"

    # 6. Token overlap ≥ 60% (flag as uncertain)
    if best_score >= 0.60:
        return best_match, f"UNCERTAIN({best_score:.0%})"

    return None, "UNRESOLVED"


# ── Main audit ────────────────────────────────────────────────────────────────
def audit_season(
    season: str,
    prop_players: set[str],
    gl_players: set[str],
    lines: list[str],
) -> dict[str, str]:
    """
    Audit one season. Returns dict of {norm(prop_name): canonical_gl_name}
    for all cases where the prop name differs from the canonical name.
    """
    if not prop_players:
        lines.append(f"\n  No props players for {season} — skipping.")
        return {}

    lines.append(f"\n{'='*70}")
    lines.append(f"  {season} — {len(prop_players)} prop players vs {len(gl_players)} game log players")
    lines.append(f"{'='*70}")

    unresolved = []
    needs_alias = {}  # prop_name → canonical_gl_name (only where they differ)
    uncertain = []
    strategy_counts: dict[str, int] = {}

    for prop_name in sorted(prop_players):
        canonical, strategy = resolve(prop_name, gl_players)
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        if strategy == "UNRESOLVED":
            unresolved.append(prop_name)
            lines.append(f"  ✗ UNRESOLVED  '{prop_name}'")

        elif strategy.startswith("UNCERTAIN"):
            uncertain.append((prop_name, canonical, strategy))
            lines.append(f"  ? UNCERTAIN   '{prop_name}' → '{canonical}' [{strategy}]")

        elif strategy == "EXACT":
            # Exact match — no alias needed
            pass

        else:
            # Resolved via a non-exact strategy — may need alias if names actually differ
            if norm(prop_name) != norm(canonical):
                needs_alias[prop_name] = canonical
                lines.append(f"  ~ ALIAS_NEED  '{prop_name}' → '{canonical}' [{strategy}]")
            # else: normalisation handles it, no explicit alias needed

    lines.append(f"\n  Strategy summary:")
    for s, c in sorted(strategy_counts.items()):
        lines.append(f"    {s:<25}: {c}")

    lines.append(f"\n  Unresolved ({len(unresolved)}):")
    for p in unresolved:
        lines.append(f"    '{p}'")

    if uncertain:
        lines.append(f"\n  Uncertain matches — verify manually ({len(uncertain)}):")
        for p, c, s in uncertain:
            lines.append(f"    '{p}' → '{c}' [{s}]")

    return needs_alias


def main() -> None:
    print("\n  PropEdge V16.0 — Name Alias Audit")
    print("  " + "─" * 60)

    print("\n[1/3] Loading game logs...")
    gl25, gl26 = load_gl_players()
    gl_all = gl25 | gl26  # combined for cross-season resolution

    print("\n[2/3] Loading props...")
    p2425, p2526 = load_props_players()

    print("\n[3/3] Auditing name matches...")
    lines: list[str] = []

    # Audit each season against its own game log, then against combined
    aliases_2425 = audit_season("2024-25", p2425, gl25 if gl25 else gl_all, lines)
    aliases_2526 = audit_season("2025-26", p2526, gl26 if gl26 else gl_all, lines)

    # Combined: any prop name that appears in either season but isn't resolved
    all_prop_players = p2425 | p2526
    aliases_combined = audit_season("COMBINED (all seasons)", all_prop_players, gl_all, lines)

    # ── Generate alias table ──────────────────────────────────────────────────
    all_aliases = {**aliases_2425, **aliases_2526, **aliases_combined}
    # Deduplicate: if same prop_name → same canonical from multiple seasons, keep one
    deduped: dict[str, str] = {}
    for prop, canonical in all_aliases.items():
        n = norm(prop)
        if n not in deduped or deduped[n] != canonical:
            deduped[n] = canonical

    alias_lines = [
        "",
        "# ── AUTO-GENERATED by build_alias_table.py ──────────────────────────────────",
        "# Copy this into player_name_aliases.py → PLAYER_ALIASES dict",
        "# Then add any UNRESOLVED names manually (check NBA.com for canonical spelling)",
        "PLAYER_ALIASES_GENERATED = {",
    ]

    # Always include the hardcoded known aliases first
    hardcoded = [
        ("carlton carrington",   "Bub Carrington",       "# nickname — unfixable by fuzzy"),
        ("herb jones",           "Herbert Jones",         "# shortened first name"),
        ("moe wagner",           "Moritz Wagner",         "# nickname"),
        ("ron holland",          "Ron Holland II",        "# missing suffix"),
        ("vincent williams jr",  "Vincent Williams Jr.",  "# missing trailing period"),
        ("nikola jokic",         "Nikola Jokić",          "# accent"),
        ("luka doncic",          "Luka Dončić",           "# accent"),
        ("kristaps porzingis",   "Kristaps Porziņģis",   "# accent"),
        ("bojan bogdanovic",     "Bojan Bogdanović",      "# accent"),
        ("dario saric",          "Dario Šarić",           "# accent"),
    ]

    alias_lines.append("    # ── Hardcoded (nickname/accent — fuzzy cannot catch these) ──")
    for k, v, comment in hardcoded:
        alias_lines.append(f'    "{k}":{" "*(28-len(k))}"{v}",{" "*(28-len(v))}{comment}')

    # Add auto-detected aliases (excluding those already hardcoded)
    hardcoded_keys = {k for k, _, _ in hardcoded}
    auto = {k: v for k, v in sorted(deduped.items()) if k not in hardcoded_keys}

    if auto:
        alias_lines.append("    # ── Auto-detected from audit ──────────────────────────────────")
        for k, v in auto.items():
            alias_lines.append(f'    "{k}":{" "*(28-len(k))}"{v}",')

    alias_lines.append("}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    report_path = ROOT / "alias_audit.txt"
    report_content = "\n".join(lines)
    report_path.write_text(report_content)
    print(f"\n  ✓ Full audit report → {report_path}")

    alias_path = ROOT / "alias_table_generated.py"
    alias_path.write_text("\n".join(alias_lines) + "\n")
    print(f"  ✓ Generated alias table → {alias_path}")

    # Print summary to console
    print(report_content)
    print("\n" + "\n".join(alias_lines))

    print(f"\n  ── NEXT STEPS ──────────────────────────────────────────────────────")
    print(f"  1. Review alias_audit.txt for UNRESOLVED and UNCERTAIN entries")
    print(f"  2. For each UNRESOLVED: find correct NBA.com name and add manually")
    print(f"  3. Copy PLAYER_ALIASES_GENERATED into player_name_aliases.py")
    print(f"  4. Run: python3 player_name_aliases.py  (self-test)")
    print(f"  5. Run: python3 run.py setup")


if __name__ == "__main__":
    main()
