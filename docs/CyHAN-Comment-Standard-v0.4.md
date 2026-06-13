# CyHAN - Comment & Docstring Standard (Python + C++)

*Draft v0.3*

This document codifies the comment, docstring, and section-divider conventions
for every CyHAN source file, organized by **file role** (launcher, orchestrator
entry, orchestration package, C++ engine, script/analysis tool, test). Each role
has a different audience and a different job, so each gets its own header
template and conventions; the shared building blocks (dividers, function
docstrings, inline comments, typography, log strings) are defined once in
Section 4 and referenced by every role. Applying it makes the comments read the
same way across all modules. Nothing here changes runtime behavior.

> **Changes from v0.2**
> 1. **Role-first structure.** v0.2 was organized by topic; v0.3 leads with a
>    per-role template (Section 3) because a launcher, an engine, and a test do
>    not document the same way. This is what standardizes comments across modules.
> 2. **Hyphen, not em dash.** The header separator and prose use a hyphen
>    (`-`), not an em dash. Em and en dashes are reserved for plot/figure
>    titles only (see 4.4). This reverses the v0.2 rule and matches the rest of
>    the project's typography policy.
> 3. **Launcher header template** (`WHAT <X> PRODUCES` / `METHOD` / `Run` /
>    `Outputs`) and the **USER OPTIONS** conventions, derived from the module
>    launchers (3.1).
> 4. A commented-out **option alternative** inside a USER OPTIONS block is
>    allowed as an operator toggle (3.1); commented-out **logic** is still banned.
> 5. Scientific notation (Greek letters, math operators) is allowed in
>    docstrings and comments (4.4).

---

## 0. Scope

- Applies to every `.py`, `.hpp`, `.h`, `.cpp`, `CMakeLists.txt`, and structured
  metadata file under `modules/` and `common/`.
- Does **not** apply to generated files (`.pyd`, `_build/`, build artifacts),
  third-party headers, or the `.venv` tree.
- When this standard and an existing file disagree, the **standard wins**,
  except for fixed external strings users may rely on (CLI `--help` text,
  on-disk output column names).

---

## 1. Universal rules (every file, every role)

1. **First content line is a one-line tag:** `<name_or_role> - <one-line purpose>.`
   The separator is a hyphen with a space on each side. End with a period.
2. **Author / POC line is mandatory,** immediately under the tag. Git blame is
   not a substitute for an explicit point of contact, and not every checkout has
   history. Use a real person, never `TBD` or a team alias as the sole contact.
3. **No em or en dashes in source text** (4.4). Use a hyphen, comma, or
   parentheses. Em/en dashes are allowed only inside plot/figure title strings.
4. **No revision-history blocks** at the top of files. Version control owns
   history; the Author / POC line is the only metadata banner in source.
5. **No commented-out logic** and **no `TODO` without an owner or tracker
   reference.** (A commented-out option *value* in a launcher USER OPTIONS block
   is a documented operator toggle, not logic; it is allowed; see 3.1.)
6. **No decorative ASCII art** beyond the three divider tiers in 4.1.

---

## 2. File roles at a glance

| Role | File pattern | Audience | Header template | Defining convention |
|------|--------------|----------|-----------------|---------------------|
| Launcher | `run_<name>.py` | Operator | 3.1 | USER OPTIONS block; operator voice |
| Orchestrator entry | `backend/python/main_<name>.py` | Developer | 3.2 | Public API; no user options |
| Orchestration package | `backend/python/<name>/*.py` | Developer | 3.3 | Library docstrings; dividers |
| C++ engine / bindings | `backend/engines/**` | Developer | 3.4 | Doxygen; engine contract |
| Script / analysis tool | `scripts/*.py`, `analysis/*.py` | Operator/dev | 3.5 | Standalone `Run` usage |
| Test | `tests/*.py`, `conftest.py` | Developer | 3.6 | What is validated |

The naming, layer placement, and role obligations of each file come from the
architecture standard (`CyHAN-Standard-v2.1.md`). This document covers only how
each is commented.

---

## 3. Per-role standards

### 3.1 Launcher (`run_<name>.py`)

The launcher is the operator's control surface. Its header is a self-contained
brief on **what the module produces and how to run it**; its body is one
**USER OPTIONS** block. Comments address the operator ("edit", "change ONLY this
string"), not a developer.

**Header template.**

```python
"""run_<name> - <ABBREV> launcher (CyHAN v2.1 5.3).

Author / POC : <Full Name>  <email>

User-facing entry for the <Full Name> (<ABBREV>) module. The operator edits the
USER OPTIONS block below and runs the script. No orchestration logic lives here;
the launcher hands the option block to ``main_<name>.run`` per 5.3.

================================================================================
WHAT <ABBREV> PRODUCES
================================================================================
<2 to 5 sentences in operator/domain terms: the deliverable and why it matters.>

================================================================================
METHOD & FORMULATION
================================================================================
<Numbered steps. Units in brackets, e.g. [events / year]. Define each symbol
once. Keep it faithful to the code, not aspirational.>

Run
---
  1. pip install -r requirements.txt
  2. Edit the USER OPTIONS block below, then:  python run_<name>.py
     ...or from the repository root:
         python modules/<name>/run_<name>.py

Outputs (data/outputs/):
  <relative/path>   - <what it contains>
"""
```

- The `WHAT ... PRODUCES` and `METHOD & FORMULATION` banners are **Tier-1
  dividers inside the docstring** (full 80-column equals run). Include both for
  any launcher whose method is non-trivial; a `Stages` or domain-specific banner
  (for example a kernel explanation) may be added.
- Keep `Run` and `Outputs` as underlined sub-sections (4.2 underline rule).

**USER OPTIONS block.** One contiguous block, opened and closed by a Tier-1
banner, grouped by Tier-3 dividers (4.1):

```python
# ===========================================================================
# USER OPTIONS  - edit anything in this block, then run the script
# ===========================================================================

# -- Stage selection --
# PRIMARY (default): ["pot"] POT only, on INPUT_CSV below.
# SECONDARY: add upstream stages in canonical order: "download","detrend","ntr".
#STAGES = ["download", "detrend", "ntr", "pot"]    # ready-to-use alternative
STAGES = ["detrend", "ntr", "pot"]

# -- Data semantics --
DT_HOURS       = 0.25          # time step (15 min)
DRY_VALUE      = -99999.0      # dry sentinel (water below ground)
AGGREGATE      = "mean"        # "mean" or "median" across a point's storms

# ===========================================================================
# END USER OPTIONS  - nothing below should need editing for routine use
# ===========================================================================
```

Per-option conventions:
- A **full-line comment** above a group or a non-obvious option states *why* it
  exists and what the realistic choices are.
- A **trailing comment** on the value states units, valid range, or effect (4.3).
  Two spaces before the `#`. Align values and comments within a group.
- **Commented-out alternatives are allowed and encouraged** when they document a
  common other setting the operator may swap in (the `#STAGES = ...` line above).
  This is an operator affordance, not dead code. It must be a value for an
  existing option, kept directly adjacent to the active line.
- Enumerate string/enum options inline: `# "mean" or "median"`.
- Everything below `END USER OPTIONS` (path anchoring, the C++ build helper, the
  `__main__` dispatch) is launcher plumbing; comment it as developer code (3.3),
  not operator prose.

### 3.2 Orchestrator entry (`backend/python/main_<name>.py`)

Non-user-facing realization of the orchestration role. The header states the
role, the entry point the launcher calls, and (when expanded) a short Public API.
No user options appear here.

**Thin entry** (workflow reduces to a validated dispatch):

```python
"""main_<name> - orchestrator entry (CyHAN v2.1 5.3).

Author / POC : <Full Name>  <email>

Receives the launcher's option dict, validates it into a <X>Config, and runs the
analysis. No user options live here.
"""
```

**Expanded entry** (multi-stage composition, batch, or dispatch): keep the same
header, then add a `Public API` section (4.2) listing the functions the launcher
and, when deployed, the shared API call, with one-line role notes:

```
Public API
----------
  launch_batch(root, ...)  -> int      parse CLI, resolve inputs, run per item
  run(config, ...)         -> result   single-item orchestration step
```

### 3.3 Orchestration package (`backend/python/<name>/*.py`)

Developer-facing library modules (config, io, orchestrator, writer, plots,
workflows, geo, postproc). Standard library header, NumPy-style function
docstrings, Tier-2/Tier-3 dividers for internal regions.

```python
"""<role> - <one-line purpose>.

Author / POC : <Full Name>  <email>

<1 to 4 sentences: what this module owns and what it does not. Note the role
boundary when relevant, e.g. "post-processing only; no engine calls".>
"""
```

- Public functions: summary line, then `Parameters` / `Returns` only when shapes
  or types are not obvious from the signature (4.2).
- Private helpers (`_name`): single-line docstring, no sections.
- Plotting modules keep their module-specific **semantic** color map local and
  import shared tokens, `style_ax`, and `save_figure` from `pystorm_common`.

### 3.4 C++ engine and bindings (`backend/engines/**`)

Engines are the numerical kernel: arrays in, arrays out, no I/O, no workflow.
Use Doxygen blocks so IDE and doxygen tooling can extract them.

**Engine header** (`.hpp` / `.cpp`), immediately after `#pragma once`:

```cpp
#pragma once
/**
 * @file        <file>.hpp
 * @brief       <one-line purpose>.
 *
 * @author      <Full Name>  <email>
 *
 * <Phase or algorithm notes, e.g. BUILD / SWAP.>
 *
 * Engine contract: arrays in, arrays out. All inputs/outputs use raw pointers or
 * std::vector; the pybind11 layer in bindings.cpp handles numpy <-> C++.
 * No config, no application I/O.
 */
```

**Bindings header** (`bindings.cpp`): state that it is the conduit and which
extension module it publishes:

```cpp
/**
 * @file        bindings.cpp
 * @brief       pybind11 layer exposing <engine>() to Python.
 *
 * @author      <Full Name>  <email>
 *
 * Wraps <ns>::<fn>() with numpy <-> C++ buffer conversions and publishes it as
 * the `_<tag>` extension module in the <name> package. Conduit only: it exposes
 * engine capability without adding orchestration (CyHAN 4.1).
 */
```

- Function docs: Doxygen `@param` / `@return`, shapes in parentheses `(n, p)`,
  row-major noted. Trailing `// [n]` shape comments on buffer declarations (4.3).
- `CMakeLists.txt` / `build.py`: a top comment naming the extension and its
  install destination is sufficient.

### 3.5 Script and analysis tool (`scripts/*.py`, `analysis/*.py`)

Standalone, runnable utilities (preprocessing, parameter sweeps, method
comparison, sensitivity, whitepaper figures). Launcher-like header minus the
USER OPTIONS block, with a `Run` section showing concrete invocations.

```python
"""<name> - <one-line purpose>.

Author / POC : <Full Name>  <email>

<What it does and what it writes. Note "Headless by design" if it only writes
files. State which module outputs or inputs it consumes.>

Run
---
    python scripts/<name>.py [--flag value]    # what the flag does
"""
```

`scripts/` and `analysis/` serve the same role under two names; comment them
identically.

### 3.6 Test (`tests/*.py`, `conftest.py`)

```python
"""test_<area> - smoke tests for the <ABBREV> module.

Author / POC : <Full Name>  <email>

Validates: (1) config construction; (2) <stage> on a synthetic record;
(3) <invariant>; (4) end-to-end <Orchestrator> run.
"""
```

- A one-line numbered "Validates:" list is the whole header; keep it current.
- `conftest.py`: a single-line module docstring stating what it wires (for
  example, putting `backend/python` and `common/python` on `sys.path`).
- Fixtures and helpers: one-line docstring; no sections.

---

## 4. Shared building blocks

### 4.1 Section dividers (three tiers, both languages)

**Tier 1** - top-level region. In launchers this is the `USER OPTIONS` block; in
docstrings it brackets the `WHAT PRODUCES` / `METHOD` banners; in long `.cpp`
files it marks a coarse phase. Full equals run to ~column 78:

```python
# ===========================================================================
# USER OPTIONS  - edit anything in this block, then run the script
# ===========================================================================
```

**Tier 2** - internal section heading (Data loading / Internal helpers / Public
API). Use sparingly, at most 3 to 5 per file:

```python
# ---------------------------------------------------------------------------
# Title-case heading
# ---------------------------------------------------------------------------
```

**Tier 3** - group label inside a config block, namespace, or function. Box-
drawing run padded to ~column 78:

```python
# -- PCA dry-node handling --
```
```cpp
// --- BUILD: maximin initialization ---
```

Rules: dividers have a blank line before and after; the C++ rule width matches
the Python equals width so visual weight is consistent across languages.

### 4.2 Function and method docstrings

**Python.** Every public function has a one-line summary ending in a period. Add
NumPy-style `Returns` (and `Parameters` only when shapes/types are non-obvious)
for public API. Section underlines match the heading length exactly (`Returns`
-> 7 dashes). Private helpers get a single line.

```python
def run_rtcs_selection(cfg):
    """RTCS selection (fixed k) - select a fixed-size Reduced TC Suite.

    Returns
    -------
    indices : ndarray [k_total]
    metrics : dict  (k, coverage, discrepancy, maximin)
    """
```

**C++.** Doxygen block above the declaration (header) or definition. Same fields
as Python via `@param` / `@return`.

```cpp
/**
 * Full PAM with FastPAM1 swap optimization.
 *
 * @param D       (n*n) row-major float64 distance matrix.
 * @param k       Number of medoids to select.
 * @param forced  Indices that must appear in the result (may be empty).
 * @return Sorted vector of k selected row indices.
 */
```

Shapes in brackets `[n_storms x p_params]` (Py) or parentheses `(n, p)` (C++);
units in parentheses `(units: m NAVD88)`. No `Examples` in function docstrings;
examples belong in the file header or README. Wrap at ~90 characters.

### 4.3 Inline comments

- **Full-line** comment for *why* the next block exists.
- **Trailing** comment for *what a value means* (units, shape, range). Two
  spaces before `#` (Python) or `//` (C++).

```python
"k_additional": 200,      # storms added on top of any forced/pre-selected
Y: np.ndarray             # float64  [n x m]  (cast up from float32)
```
```cpp
std::vector<double> d1(n), d2(n);   // distances to nearest / second medoid
```

Anti-patterns: restating the code (`i += 1  # increment i`); duplicating a type
already in the hint or C++ type; stale TODOs; commented-out logic.

### 4.4 Scientific notation and typography

- **Em/en dashes are not used in source text.** Use a hyphen, comma, or
  parentheses. The only place an em/en dash is allowed is inside a plot or
  figure title string (matplotlib `set_title`, axis labels), never in code,
  docstrings, comments, run scripts, or commits.
- **Scientific symbols are allowed and encouraged** where they aid the domain
  reader: Greek letters (mu, xi, sigma written as the glyphs), math operators
  (x for product, <=, >=, ->), and bullets in prose lists. Define each symbol
  once near first use.
- ASCII is the default for plain prose; reach for a symbol only when it is the
  natural scientific notation.

### 4.5 Workflow log strings

Not comments, but the same house style. Bracketed step index, four-space indent
for sub-lines; C++ engines match the convention when they print progress.

```python
print("\n[1] Loading data ...")
print(f"    Source : HDF5  ({h5})")
print(f"    X      : {X.shape}  (storms x parameters)")
```

Per-module prefixes (`[pot]`, `[ssh]`, `[tca]`) on operator-facing status lines
are fine; keep one prefix per module.

---

## 5. What we deliberately do NOT include

- No revision-history banners (version control owns history).
- No `TODO` / `// TODO` without an owner or tracker reference.
- No commented-out logic. (Commented-out option *values* in a launcher USER
  OPTIONS block are the one allowed exception; see 3.1.)
- No em/en dashes outside plot/figure titles.
- No decorative ASCII art beyond the three divider tiers.
- No `Examples` blocks inside function docstrings.

---

## 6. Quick reference

**By role.**

| Role | Header has | Special convention |
|------|-----------|--------------------|
| Launcher | tag, POC, WHAT PRODUCES, METHOD, Run, Outputs | USER OPTIONS block; operator voice |
| Orchestrator entry | tag, POC, (Public API if expanded) | no user options |
| Orchestration package | tag, POC, role boundary | NumPy docstrings; dividers |
| C++ engine | Doxygen @file/@brief/@author, engine contract | arrays in/out, no I/O |
| Bindings | Doxygen, "conduit", extension name | conversions only |
| Script / analysis | tag, POC, Run usage | headless note |
| Test | tag, POC, numbered Validates | keep list current |

**By element.**

| Element | Python | C++ |
|---------|--------|-----|
| File tag | `"""name - purpose."""` | `/** @file ... @brief ... */` |
| Author / POC | `Author / POC : Name <email>` | `@author Name <email>` |
| Tier 1 divider | `# ===...===` | `// ===...===` |
| Tier 2 divider | `# ---...---` | `// ---...---` |
| Tier 3 divider | `# -- label --` | `// --- label ---` |
| Public docstring | `"""Summary."""` + `Returns` | `/** Summary. */` + `@return` |
| Trailing comment | `value,   # meaning` | `value;   // meaning` |
| Separator | hyphen ` - ` | hyphen ` - ` |

---

## 7. Rollout

1. **Audit** every file under `modules/` and `common/` by role against Section 3;
   produce a divergence list (missing POC lines, em-dash headers, wrong divider
   widths, launcher headers missing `WHAT PRODUCES` / `METHOD`).
2. **Normalize headers to hyphens.** The launchers in POT, PST, and RTCS still
   use em-dash header separators (legacy v0.2 rule); convert them to hyphens.
3. **Normalize** divider widths, underline lengths, and Doxygen tag form in a
   single pass per module.
4. For any file missing an Author / POC line, **ask before assigning**; do not
   auto-populate from `git blame`.
5. Keep this document in `docs/` alongside `CyHAN-Standard-v2.1.md` as the
   long-lived reference.
