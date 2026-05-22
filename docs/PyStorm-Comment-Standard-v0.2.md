# PyStorm — Comment & Docstring Standard (Python + C++)

*Draft v0.2*

This document codifies the comment, docstring, and section-divider conventions
for every PyStorm source file — both Python (`.py`) and C++ (`.hpp` / `.cpp`).
It is a checklist for future edits and the reference applied during a
one-pass normalization of the existing scripts. Nothing in this document
changes runtime behavior.

> **Changes from v0.1:** (1) every file header now carries an explicit
> Author / Point-of-Contact field — we do not assume Git or GitHub history
> will always be available; (2) the standard covers C++ as well as Python,
> with one paired sub-section per topic.

---

## 0. Scope

- Applies to every `.py`, `.hpp`, `.h`, and `.cpp` file under `modules/`
  (launchers, engines, workflows, bindings, configs, tests).
- Does **not** apply to generated files (`.pyd`, build artifacts), third-party
  headers, or the `.venv` tree.
- When this standard and an existing file disagree, the **standard wins** —
  except for fixed external strings (e.g. CLI help users may rely on).

---

## 1. File header

Every source file opens with a header block whose first content line is a
one-sentence tag of the form `<module_or_role> — <one-line purpose>.`,
immediately followed by an **Author / POC** line. The author line is
mandatory: not every file lives in GitHub, and `git blame` is not a
substitute for an explicit point of contact.

### 1a. Python (`.py`)

Triple-quoted module docstring. Full template:

```python
"""<module_or_role> — <one-line purpose>.

Author / POC : <Full Name>  <email-or-handle>
Maintainers  : <Name 2>, <Name 3>          # optional, only when >1
Created      : YYYY-MM-DD                   # optional

<1-4 sentence summary: what this file owns, what it does not.>

<Optional named sections — include only the ones that apply.>

Usage
-----
    python scripts/<name>.py [--flags]

Public API
----------
  func_a(args)  ->  return_type
  func_b(args)  ->  return_type

Algorithm
---------
Step 1 — ...
Step 2 — ...

Input  : <path or producer>
Outputs: <path(s)>
"""
```

### 1b. C++ (`.hpp`, `.h`, `.cpp`)

Doxygen-style block comment, placed immediately after `#pragma once` (for
headers) or the include block (for `.cpp` files). Same fields as Python,
written with Doxygen tags so doxygen / IDE tooling can extract them.

```cpp
#pragma once
/**
 * @file        kmedoids_core.hpp
 * @brief       <one-line purpose>.
 *
 * @author      <Full Name>  <email-or-handle>
 * @maintainer  <Name 2>, <Name 3>            // optional, only when >1
 * @date        YYYY-MM-DD                    // optional
 *
 * <1-4 sentence summary.>
 *
 * Algorithm
 * ---------
 * Step 1 — ...
 * Step 2 — ...
 */
```

### Rules (both languages)

- **Author / POC line is mandatory.** Use a real person; do not write `TBD`
  or a team alias as the sole contact.
- Use an em-dash (`—`) between the module name and the one-line purpose,
  not a hyphen.
- Engine modules (pure numerical, no I/O) add the line
  `Engine contract: arrays in, arrays out. No config, no I/O.`
- Launcher scripts include `Usage`, `Input`, `Outputs`. Library modules
  include `Public API`; add `Algorithm` only when the math is non-obvious.
- Section underlines in Python docstrings must be exactly the length of
  the heading (`Usage` → 5 dashes).

---

## 2. Section dividers inside a file

Three tiers, used consistently across both languages.

### 2a. Python

**Tier 1** — top-level user-edit block (launchers only):

```python
# ===========================================================================
# USER OPTIONS  — edit anything in this block, then run the script
# ===========================================================================
...
# ===========================================================================
# END USER OPTIONS  — nothing below should need editing for routine use
# ===========================================================================
```

**Tier 2** — internal section heading:

```python
# ---------------------------------------------------------------------------
# Title Case heading, sentence-style
# ---------------------------------------------------------------------------
```

**Tier 3** — group label inside a dict / config literal:

```python
    # ── PCA ───────────────────────────────────────────────────────────────
```

### 2b. C++

Same three tiers, with C++ line-comment syntax. The slash run matches the
width of the Python equals-dash run so the visual weight is consistent
across languages.

**Tier 1** — major file region (rare; used in large `.cpp` files):

```cpp
// ==========================================================================
// PUBLIC ENTRY POINTS
// ==========================================================================
```

**Tier 2** — internal section heading:

```cpp
// --------------------------------------------------------------------------
// Title Case heading
// --------------------------------------------------------------------------
```

**Tier 3** — sub-group label inside a namespace or function:

```cpp
// ─── BUILD: maximin initialization ───────────────────────────────────────
```

### Rules (both languages)

- Tier 1 is reserved for genuinely top-level regions: `USER OPTIONS` in
  Python launchers, or coarse phase markers in long `.cpp` files.
- Tier 2 separates logical regions (Data loading / Internal helpers /
  Public API). Use sparingly — at most 3–5 per file.
- Tier 3 (box-drawing `──`) groups dict keys (Python) or related function
  blocks (C++); the trailing `─` run pads to ~column 78.
- Dividers always have a blank line before and after.

---

## 3. Function and method docstrings

### 3a. Python

Every public function has at minimum a one-line summary ending in a period.
NumPy-style `Returns` (and `Parameters` only when types or shapes are
non-obvious from the signature) are added for public API.

Minimal form (private helpers, simple public functions):

```python
def select_kmedoids(Z, k, seed, forced_indices=None):
    """Select k medoids from the rows of Z."""
```

Full form (workflow entry points, non-obvious returns):

```python
def run_rtcs_selection(cfg):
    """RTCS Selection (fixed k) — Select a fixed-size Reduced TC Suite.

    Returns
    -------
    indices : ndarray [k_total]
    metrics : dict  (k, coverage, discrepancy, maximin)
    """
```

### 3b. C++

Doxygen block comment immediately above the function declaration (in
headers) or above the definition (when only a `.cpp` exists). Use the same
field names as Python — Parameters, Returns — written as Doxygen `@param`
/ `@return` tags so IDEs can pick them up.

Minimal form:

```cpp
/** Select k medoids from rows of D using PAM. */
std::vector<int> pam(const double* D, int n, int k, uint64_t seed,
                    const std::vector<int>& forced);
```

Full form (public API, non-obvious arguments):

```cpp
/**
 * Full PAM with FastPAM1 swap optimization.
 *
 * @param D       (n*n) row-major float64 distance matrix.
 * @param n       Number of points.
 * @param k       Number of medoids to select.
 * @param seed    RNG seed for BUILD initialization.
 * @param forced  Indices that must appear in the result (may be empty).
 *
 * @return Sorted vector of k selected row indices.
 */
std::vector<int> pam(...);
```

### Rules (both languages)

- Private helpers (Python `_name` / C++ static or anonymous namespace):
  single-line docstring, no sections.
- Public functions: summary line, blank line, then optional Parameters /
  Returns.
- Array shapes go in brackets: `[n_storms x p_params]` (Py) or `(n, p)`
  (C++).
- Units go in parentheses: `(units: m NAVD88)`.
- No `Examples` section in function docstrings — examples belong in the
  file header or the README.
- Wrap docstrings / Doxygen blocks at ~90 characters.

---

## 4. Inline comments

- Use **full-line** comments for *why* the next block exists.
- Use **trailing** comments for *what a value means* (units, shape, valid
  range).
- Trailing comments have **two spaces** before the `#` (Python) or `//`
  (C++).

### 4a. Python

```python
"k_additional": 200,      # storms added on top of any forced/pre-selected

X:  np.ndarray            # float64  [n x p]
Y:  np.ndarray            # float64  [n x m]  (cast up from float32)
HC: Optional[np.ndarray]  # float64  [m x N_AER] or None

# BBOX_CONFIG = None   # ← uncomment to disable bbox filtering
```

### 4b. C++

```cpp
std::vector<int>    nearest(n), second_nearest(n);  // [n] per point
std::vector<double> d1(n), d2(n);                   // distances to 1st/2nd medoid

// j's nearest medoid is the one being removed -- use the second-nearest
// as a fallback unless the candidate itself is closer.
double new_cost = (d_j_cand < d2[j]) ? d_j_cand : d2[j];
```

### Anti-patterns

- Comments that restate the code (`i += 1  // increment i`).
- Stale TODOs without owner or issue reference.
- Commented-out code — delete it; version control remembers.
- Type information duplicated in comments when it is already in the
  Python type hint or C++ type.

---

## 5. Workflow log strings

Not comments, but consistent with the same house style. Workflow steps are
printed with a bracketed step index and a four-space indent for sub-lines.
C++ engines should match the convention when they emit progress to stdout
from inside long-running kernels.

```python
print("\n[1] Loading data ...")
print(f"    Source : HDF5  ({h5})")
print(f"    X      : {X.shape}  (storms x parameters)")
```

```cpp
std::cout << "\n[1] Loading data ...\n"
          << "    Source : HDF5  (" << h5 << ")\n";
```

---

## 6. What we deliberately do NOT include

- No revision-history blocks at the top of files — version control owns
  history. (The single Author / POC line in §1 is the **only** metadata
  banner that belongs in the source.)
- No `# TODO` / `// TODO` without an attached owner or issue reference;
  prefer the project tracker.
- No commented-out code left in source.
- No emoji or decorative ASCII art beyond the three divider tiers defined
  in §2.

---

## 7. Quick reference

| Element             | When to use                          | Python form              | C++ form                  |
|---------------------|--------------------------------------|--------------------------|---------------------------|
| File header         | Every source file                    | `"""name — purpose."""`  | `/** @file ... @brief */` |
| Author / POC line   | Mandatory in every file header       | `Author / POC : Name`    | `@author Name <email>`    |
| Tier 1 divider      | USER OPTIONS / coarse phase markers  | `# ====...====`          | `// ====...====`          |
| Tier 2 divider      | Internal regions of a file           | `# ----...----`          | `// ----...----`          |
| Tier 3 divider      | Sub-group label                      | `# ── label ──`          | `// ── label ──`          |
| Function docstring  | Every public function                | `"""Summary."""`         | `/** Summary. */`         |
| Returns section     | Public API, non-obvious returns      | `Returns\n-------`       | `@return ...`             |
| Trailing comment    | Units, role, valid range             | `value,      # meaning`  | `value;      // meaning`  |
| Workflow log        | Pipeline progress in workflows       | `"\n[N] Step ..."`       | `"\n[N] Step ..."`        |

---

## 8. Rollout

1. **Audit** every `.py` / `.hpp` / `.cpp` file under `modules/` against
   this standard; produce a divergence list (missing Author lines, wrong
   divider widths, etc.).
2. For each file missing an Author / POC line, **ask before assigning** —
   do not auto-populate from `git blame`, since that can be wrong or
   absent.
3. **Normalize** everything else in a single pass (underline lengths,
   Tier-3 box-char padding, Doxygen tag form).
4. Commit this document to `docs/` alongside the existing
   `CyHAN-Standard-v1.1.md` as the long-lived reference.
