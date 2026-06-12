# C++/Python Hybrid Architecture Network (CyHAN)
# Standard v2.1

---

## Document Control

**Title:** C++/Python Hybrid Architecture Network (CyHAN) Standard v2.1
**Status:** Minor Release (Compute-conditional engine; orchestrator-side batch marshaling)
**Scope:** Desktop and Cloud Scientific Computing Platforms
**Audience:** System architects, HPC developers, scientific software engineers, technical leadership

---

## 0. Conformance and Terminology

### 0.1 Requirement Keywords

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**,
**SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **MAY**, and **OPTIONAL** in this
document are to be interpreted as described in RFC 2119.

- **MUST** / **SHALL** / **REQUIRED**: an absolute requirement of the standard.
- **MUST NOT** / **SHALL NOT**: an absolute prohibition.
- **SHOULD** / **RECOMMENDED**: a strong default; deviation is permitted only
  with a deliberate, understood reason.
- **SHOULD NOT** / **NOT RECOMMENDED**: strong discouragement under the same caveat.
- **MAY** / **OPTIONAL**: genuinely discretionary; conformance does not depend
  on the choice either way.

### 0.2 Normative and Non-Normative Material

Sections 1 through 17 are **normative**: they define conformance obligations.

Appendix A (Migration) and any text explicitly marked *(non-normative)* or
introduced with *e.g.* are **illustrative only**. Tool names, library names, and
concrete commands that appear as examples carry no conformance force. A
conforming implementation MAY use entirely different tools provided the
normative role obligations are met.

### 0.3 Behavioral Compliance

Compliance with this standard is **behavioral**, not structural. Conformance is
determined by whether each architectural **role** is fulfilled with its defined
responsibilities and prohibitions, not by the presence or absence of any
particular file, directory, or named artifact.

The recommended folder structure (§16) and naming patterns (§3) exist to make
conformance easy to achieve and easy to read. They are not themselves the test.
Where this document mandates a specific artifact (notably the launcher and
orchestration files of §5), it does so because that artifact is the minimum
realization of a role, not because the filename is the obligation.

---

## 1. Purpose

The **C++/Python Hybrid Architecture Network (CyHAN)** (pronounced **“cyan”**)
establishes a unified computational architecture with a single canonical
execution path and reproducible execution semantics for high-performance
scientific and engineering platforms.

This standard defines:

- Layer boundaries
- Execution topology
- Module decomposition
- Root-level integration
- Responsibility isolation
- Deployment models
- Conformance requirements
- Architectural constraints

CyHAN is intended for long-lifecycle technical systems where correctness,
scalability, and maintainability are critical.

Version 2.0 retains the **module** as the unit of structural decomposition and
distribution, brings **root-level integration** (a shared API surface and shared
frontend adapters) into scope, and fixes the per-module entry contract: every
module ships a user-facing **launcher** and a non-user-facing **orchestrator**.
The language doctrine is unchanged and explicit: numerical computation is
**C++-first**, with Python employed where it is structurally beneficial:
orchestration, workflow control, validation, and AI/ML integration.

---

## 2. Problem Statement

Scientific and engineering platforms frequently diverge due to:

- Separate desktop and cloud implementations
- Duplicate orchestration logic
- Direct UI-to-engine coupling
- Inconsistent execution semantics
- Fragmented deployment models
- Monolithic capability repositories that resist independent distribution

Such divergence leads to:

- Non-reproducible results
- Increased maintenance burden
- Scaling limitations
- Hidden technical debt
- Inability to share or vendor individual capabilities

CyHAN eliminates divergence by enforcing:

- A single backend execution path
- Strict separation of UI, orchestration, and compute
- Clear authority boundaries between layers
- Self-contained module decomposition
- A defined root-level integration seam above the modules

CyHAN is compute-first, not UI-first.

---

## 3. Canonical Architecture

### 3.1 Execution Topology

```
Desktop Client (Qt C++)
or
Web Client (Browser)
│
▼
Python API
▼
Python Orchestration
▼
C++ Engines
```

This is the execution topology for a fully integrated CyHAN system. All compute
**MUST** traverse this stack. There **SHALL** be no alternate execution path in
compliant systems.

### 3.2 Architectural Ordering Rationale

The ordering reflects:

- Native-first engineering philosophy
- C++ as numerical authority
- Desktop as primary development environment
- Web as scalable distribution interface

Desktop and Web clients are peers. Neither owns computation.

### 3.3 Module Decomposition

CyHAN partitions capability into **modules**. Each module is a self-contained
vertical that realizes the lower portion of the execution topology, the C++
engine and its Python orchestration, for a single, well-defined capability.

Root-level integration (the shared API surface and shared frontend adapters that
sit above modules) is **in scope** in v2.0 and specified in §6. Modules remain
independently buildable and runnable without the root-level integration present.

A compliant module **SHALL**:

- Contain its own engine and orchestration role realization
- Be buildable, runnable, and distributable in isolation
- Not depend on the source trees of sibling modules

Module specification appears in §5; root-level integration in §6.

### 3.4 Relationship to Prior Versions

Version 1.0 established the canonical unified execution path. Version 1.1
introduced module decomposition and the C++-first language doctrine, and
explicitly deferred root-level integration (v1.1 §16.8), forbidding root-level
`backend/` and `frontend/` directories until a future amendment.

**Version 2.0 is that amendment.** It supersedes v1.1 §16.8: root-level
integration is now permitted and specified (§6, §16.9). It additionally fixes the
per-module entry contract (§5.3) and adopts the explicit RFC 2119 register and
behavioral-compliance statement of §0.

**Version 2.1** refines v2.0 without architectural change: it makes the C++
engine conditional on the presence of performance-critical computation (§5.1),
places dataset/input resolution and batch marshaling in the orchestrator (§5.3),
and clarifies the layout for engine-less modules and a shared `common/` library
(§16). See §13.

---

## 4. Layer Specifications

The CyHAN architecture defines three backend layers and two frontend layer
types. Each is an architectural **role** with defined responsibilities and
prohibitions. The artifacts that realize each role are specified per-layer and
in §5–§6.

---

### 4.1 C++ Engines

#### Role

C++ engines are the sole owners of performance-critical computation.

They **SHALL**:

- Implement numerical kernels
- Execute Monte Carlo simulation
- Perform mesh/grid operations
- Handle memory-intensive workloads
- Support controlled and reproducible execution

#### Constraints

C++ engines **SHALL NOT**:

- Expose HTTP endpoints
- Import UI frameworks
- Contain workflow orchestration logic
- Maintain frontend state
- Perform application-level file I/O on behalf of a workflow

#### Reproducibility

Where stochastic modeling is used:

- Random number generators **MUST** be seed-controlled
- Execution order **SHOULD** be stable
- Numerical behavior **MUST** be consistent across deployment modes

This requirement refers to execution reproducibility, not deterministic hazard
modeling philosophy.

#### Bindings

A module's engine is exposed to its orchestrator through a binding layer
(e.g. pybind11). A binding **SHALL** be a conduit, not an authority: it exposes
engine capability without leaking engine internals upward and without absorbing
orchestration responsibility. The choice of binding technology is non-normative.

---

### 4.2 Python Orchestration

#### Role

Python Orchestration is the architectural role responsible for workflow assembly
and execution control. Every module **SHALL** realize this role in Python, in a
**non-user-facing** orchestrator that the launcher invokes (see §5.3). The
orchestrator's internal form scales with module complexity.

**Artifact forms.**

- At minimum, a module **SHALL** provide a single importable orchestrator module
  (see §5.3). This is sufficient for a module whose workflow reduces to a direct
  engine invocation.
- When a module performs multi-step composition, input validation beyond direct
  type checks, AI/ML integration, ensemble coordination, or distributed worker
  management, the orchestrator **SHOULD** expand into a dedicated package while
  preserving the same import entry point.

In either form, the orchestration role's responsibilities and prohibitions are
invariant, and the role **SHALL NOT** be realized inside the launcher (§5.3).

#### Responsibilities

Python Orchestration **SHALL**:

- Validate domain-level inputs
- Compose engine calls
- Manage job lifecycle
- Coordinate distributed workers (where applicable)
- Perform lightweight post-processing
- Integrate AI/ML modules

#### Constraints

Python Orchestration **SHALL NOT**:

- Implement heavy numerical kernels
- Duplicate engine logic
- Contain UI rendering code
- Serve as the public API boundary

Python coordinates. It does not numerically dominate.

#### Language Doctrine

CyHAN is **C++-first**. Python is employed where it provides structural benefit
(orchestration, workflow control, validation, AI/ML integration), not as a
default. The Python footprint of a module **SHALL** be proportional to the
orchestration complexity it warrants. Python that is purely ceremonial and adds
no orchestration value is prohibited (§10).

---

### 4.3 Python API

#### Role

The Python API defines the authoritative system boundary for external clients.
It is the architectural seam between clients and module orchestration.

#### Scope in v2.0

The API surface **SHALL** be realized at the **root level**, above the module
layer, and **SHALL** dispatch into module orchestrators. It is specified in §6.

A web framework (e.g. FastAPI) is the conventional realization; the framework
choice is non-normative.

#### Responsibilities

The Python API **SHALL**:

- Expose stable, documented endpoints
- Validate input schemas
- Authenticate and authorize requests
- Dispatch to module orchestration
- Manage asynchronous execution

#### Authority

All external clients **MUST** communicate exclusively through this boundary. The
API **SHALL** import and call module orchestrators; it **SHALL NOT** reach past
an orchestrator to invoke an engine directly, and it **SHALL NOT** contain
numerical or workflow-composition logic of its own.

---

### 4.4 Qt Desktop Application

#### Role

The Qt Desktop Application is a native client adapter.

It **SHALL**:

- Own the GUI event loop
- Manage window lifecycle
- Render visualization
- Communicate with the Python API via HTTP or WebSocket

A module **MAY** ship its own self-contained Qt component under the module's
`frontend/desktop/` directory for module-scoped use. The root-level Qt shell that
aggregates module GUIs into a unified application is specified in §6.

#### Constraints

The Qt Desktop Application **SHALL NOT**:

- Directly invoke C++ engines in compliant deployments
- Contain orchestration logic
- Replicate backend computation

HTML inside Qt is optional and does not affect compliance.

---

### 4.5 Web Frontend

The Web Frontend **SHALL**:

- Render UI
- Submit jobs
- Monitor execution
- Visualize results

It **SHALL NOT**:

- Execute heavy computation
- Contain orchestration logic

A module **MAY** ship its own self-contained web component under the module's
`frontend/web/` directory. The root-level web frontend that aggregates module
UIs is specified in §6.

---

## 5. Modules

### 5.1 Definition

A **module** is a self-contained vertical that realizes a single, well-defined
CyHAN capability. A module **SHALL** contain:

- Its C++ engine (mandatory where the module performs performance-critical
  computation per §4.1; a module whose workload has no such computation conforms
  without one)
- Its orchestration role realization (mandatory; see §5.3)
- Its launcher (mandatory; see §5.3)

A module **MAY** additionally contain:

- An expanded orchestration package (in place of the minimum single file)
- Module-scoped frontend components (desktop, web, or both)
- Tests, reference data, and module documentation

### 5.2 Self-Containment

A module **SHALL**:

- Build independently of sibling modules
- Run independently of sibling modules
- Be vendorable as a standalone unit

A module **SHALL NOT**:

- Reach into the source tree of a sibling module
- Share build state with sibling modules

Modules **MAY** depend on root-level integration code (§6) and on a shared common
library where one is introduced; a module **SHALL** remain runnable in isolation
through its launcher even when root-level integration is absent.

### 5.3 Launcher and Orchestrator (Entry Contract)

Every module **SHALL** ship exactly two mandatory entry artifacts, with a strict
division of audience and responsibility.

**The launcher: `run_<name>.py`.**

The launcher is the **user-facing** control surface. Its intended reader and
editor is the scientist or operator running the module, not a developer.

The launcher **SHALL**:

- Reside at the module root: `modules/<name>/run_<name>.py`
- Present a clearly delimited user-options block (input paths, parameters,
  settings, and toggles) as the primary editable content
- Carry the module's header/description
- Invoke the orchestrator to execute the module

The launcher **SHALL NOT**:

- Contain orchestration logic (validation, engine-call composition, post-processing)
- Implement numerical computation
- Resolve datasets or inputs from registries, or iterate batches/ensembles over
  them: dataset/input resolution and batch marshaling are orchestration
  responsibilities and **SHALL** reside in the orchestrator, not the launcher.
  The launcher **MAY** hold the operator's declarative registries (the editable
  option data) but **SHALL NOT** perform the resolution over them.
- Be the place where the orchestration role is realized

The launcher is, in effect, a readable substitute for a long command-line
invocation: the operator edits values in one file rather than supplying every
path and option as a CLI flag.

**The orchestrator: `main_<name>.py`.**

The orchestrator is the **non-user-facing** realization of the Python
Orchestration role (§4.2). It is visible to developers but is not part of the
operator-facing surface.

The orchestrator **SHALL**:

- Reside under the module backend: `modules/<name>/backend/python/main_<name>.py`
- Be importable and expose the entry point the launcher calls
- Fulfill the orchestration responsibilities and prohibitions of §4.2
- Own dataset/input resolution and any batch or ensemble iteration over inputs,
  exposing a single entry point that the launcher calls with the operator's
  declarative option block
- Begin as a single file and expand into a `backend/python/<name>/` package as
  complexity warrants, preserving its import entry point

The launcher **SHALL** import and call the orchestrator. The orchestration role
**SHALL NOT** be inlined into the launcher even for trivial modules; the
separation of the user-facing launcher from the non-user-facing orchestrator is
invariant across module complexity.

> *(Non-normative.)* The names `run_` (launcher) and `main_` (orchestrator) share
> a deliberate pairing: `run_<name>.py` is what the operator runs; `main_<name>.py`
> is the substantive logic it drives.

### 5.4 Naming

The module identifier `<name>`:

- **SHALL** be `snake_case`
- **SHALL** identify the module end-to-end (directory name, launcher and
  orchestrator filenames, and any binding, plugin, or package artifacts derived
  from it)
- **SHOULD** describe the capability, not the implementation technology

This standard prescribes no project-wide prefix or branding. Implementations
**MAY** adopt a consistent `<prefix>` for derived artifact names (e.g. a Python
package prefix, binding module name, plugin filename) provided the convention is
applied consistently within the implementation. The naming-pattern table in §17
uses `<name>` and `<prefix>` as placeholders for this reason.

---

## 6. Root-Level Integration

Root-level integration is the shared layer that sits **above** the modules and
composes them into one system. It is in scope in v2.0.

### 6.1 Shared API Surface

The shared API **SHALL** live at `backend/api/python/`. It is **not** a module.

The shared API **SHALL**:

- Import per-module orchestrators (`modules/<name>/backend/python/...`)
- Dispatch external requests into the appropriate module orchestrator
- Fulfill the Python API role and constraints of §4.3

The shared API **SHALL NOT** invoke a module's engine directly, bypassing that
module's orchestrator.

### 6.2 Shared Frontend Adapters

Root-level frontend adapters compose module-scoped frontends into unified
clients.

- A shared desktop shell **MAY** live at `frontend/desktop/` and load each
  module's GUI component at runtime.
- A shared web frontend **MAY** live at `frontend/web/` and aggregate
  module UIs.

Root-level frontends **SHALL** communicate with the system exclusively through
the shared API (§6.1) and **SHALL NOT** contain orchestration logic or numerical
kernels.

### 6.3 Independence Guarantee

The presence of root-level integration **SHALL NOT** be required for a module to
build or run. Each module remains independently operable through its launcher
(§5.3). Root-level integration composes modules; it does not own them.

---

## 7. Execution Standard

### 7.1 Canonical Flow

```
Client (Request)
↓
API Layer
• Schema validation
• Authentication and authorization
• Request dispatch
↓
Module Orchestration
• Workflow construction
• Dependency resolution
• Engine coordination
↓
C++ Engine
• Reproducible numerical computation
↓
Module Orchestration
• Result aggregation
• Post-processing
• Metadata enrichment
↓
API Layer
• Serialization
• Response emission
↓
Client (Response)
```

All compute **SHALL** follow this sequence in an integrated deployment.

### 7.2 Direct Module Invocation

When a caller invokes a module directly through its launcher rather than through
the shared API, the flow reduces to:

```
Caller (CLI, script, or external Python code)
↓
Launcher (run_<name>.py)  : user options
↓
Module Orchestration (main_<name>.py)
↓
C++ Engine
↓
Module Orchestration
↓
Caller
```

The orchestration and engine roles **SHALL** be respected; only the API segment
is absent. This is a first-class deployment posture, not a degraded one.

---

## 8. Deployment Models

### 8.1 Desktop Mode

```
Desktop Client → API → Module Orchestration → C++ Engine
```

Changing deployment environment **SHALL** require only a configuration change.

### 8.2 Cloud Mode

```
Web Client → API → Module Orchestration → C++ Engine
```

Backend services may be containerized and horizontally scalable.

### 8.3 CLI / Launcher Mode

```
Command-Line Interface or Launcher → Module Orchestration → C++ Engine
```

CLI or launcher invocation **MAY** bypass the API role but **SHALL NOT** bypass
the orchestration role. The orchestrator (`main_<name>.py`) fulfills this role.

---

## 9. Execution Parity

CyHAN enforces **desktop–cloud execution parity**.

Given identical:

- Inputs
- Configuration
- Seeds
- Engine versions
- Module versions

Desktop and cloud deployments **SHALL** produce equivalent computational results.

---

## 10. Prohibited Architectural Anti-Patterns

The following practices violate this standard:

- Direct Desktop Client → C++ Engine invocation
- Direct Web Client → C++ Engine invocation
- Shared API reaching past an orchestrator to invoke an engine directly
- Parallel orchestration stacks for desktop and cloud deployments
- Numerical kernels embedded within UI layers
- Workflow orchestration logic embedded within C++ engine code
- Orchestration logic embedded within a module launcher (`run_<name>.py`)
- Multiple or forked execution paths that bypass orchestration
- Duplication of backend logic across clients
- Modules that reach into sibling module source trees
- Modules whose launcher or build references absolute system paths
- Use of Python where its presence is purely ceremonial and adds no
  orchestration value

---

## 11. Conformance Criteria

A system **SHALL** be considered CyHAN v2.1 compliant if:

- Heavy numerical computation resides in C++
- The orchestration role is realized in Python, in a non-user-facing
  orchestrator that the launcher invokes (§5.3)
- The launcher is a user-facing control surface containing options and a launch
  call, with no orchestration logic (§5.3)
- C++ engine code contains no HTTP, UI, or workflow orchestration logic
- Each module is self-contained per §5.2 and runnable in isolation via its launcher
- Each module ships both mandatory entry artifacts per §5.3
- The shared API, where deployed, is the sole client-facing boundary and
  dispatches only through orchestrators (§6.1)
- Desktop and Web frontends share backend semantics
- Execution parity is preserved per §9
- Reproducible execution is supported per §4.1

Compliance is **behavioral** (§0.3). The question is whether each role is
fulfilled with its responsibilities and prohibitions intact, not whether a
particular optional artifact happens to exist. The two mandatory entry artifacts
of §5.3 are required because they are the minimum realization of the launcher and
orchestrator roles, not as a structural checkbox independent of those roles.

---

## 12. Performance Doctrine

- Heavy compute **MUST** reside in C++
- Python **SHALL** coordinate, not dominate
- Python footprint per module **SHOULD** be proportional to orchestration complexity
- API overhead **SHOULD** be negligible relative to engine runtime
- Scaling **SHALL** occur at worker or orchestration level

---

## 13. Versioning

CyHAN follows semantic versioning:

- **MAJOR**: Architectural changes
- **MINOR**: Backward-compatible enhancements
- **PATCH**: Clarifications or corrections

Version 1.0 established the canonical unified execution path. Version 1.1
introduced module decomposition and the C++-first language doctrine. Version 2.0
is a **MAJOR** increment: it brings root-level integration into scope and lifts
the v1.1 §16.8 prohibition on root-level `backend/` and `frontend/` directories,
an architectural change, and additionally fixes the launcher/orchestrator
entry contract and adopts the explicit RFC 2119 register and
behavioral-compliance statement.

Version 2.1 is a **MINOR** increment. It reconciles §5.1 with the
behavioral-compliance doctrine (the C++ engine is mandatory only where the module
performs performance-critical computation; a module with no such computation
conforms without one), tightens §5.3 to place dataset/input resolution and batch
marshaling in the orchestrator rather than the launcher, and notes in §16 that
`backend/engines/cpp/` is omitted for pure-Python and optional-accelerator
modules and that a shared `common/` library (§5.2) MAY hold cross-module
presentation helpers. It is backward compatible: every v2.0-conformant module
remains conformant under v2.1.

---

## 14. Position Statement

CyHAN is:

- Compute-first
- C++-first in numerical authority
- Reproducible
- Multi-frontend
- Cloud-ready
- Desktop-native
- Modular
- Architecturally unified

CyHAN is not:

- UI-first
- Web-only
- Multi-path
- Monolithic C++
- Python-only
- Python-by-default

---

## 15. Terminology Rationale

The name **C++/Python Hybrid Architecture Network (CyHAN)** is deliberate. Each
term conveys a specific structural property of the system.

### 15.1 Why "Hybrid"

CyHAN combines multiple computational paradigms within a single coherent
execution model.

- **Language hybridization**: C++ provides high-performance numerically
  intensive computation; Python provides orchestration, workflow governance,
  integration, and API control. C++ is the default for computation; Python is
  employed where it provides structural benefit.
- **Deployment hybridization**: native desktop (Qt C++), cloud web (browser),
  and command-line / launcher execution, all sharing one backend execution path.
- **Execution hybridization**: compiled native engines for performance, dynamic
  orchestration for flexibility, and controlled stochastic modeling with
  reproducible execution semantics.

Hybrid does not imply fragmentation. It implies intentional integration of
complementary capabilities.

### 15.2 Why "Architecture"

CyHAN defines structural rules governing layer boundaries, authority separation,
execution flow, responsibility ownership, module decomposition, and compliance
constraints. It is not a framework or a library; it is a structural doctrine
governing system composition. The defining principle is the single canonical
execution path:

```
Client → Python API → Python Orchestration → C++ Engines
```

### 15.3 Why "Network"

CyHAN connects multiple independent components through well-defined boundaries
while preserving execution unity:

- **Multi-client network**: Qt desktop, web, CLI, and automation clients.
- **Service network**: API layer, orchestration layer, distributed compute
  workers, and engine modules.
- **Module network**: multiple self-contained modules, each independently
  buildable, runnable, and distributable, all conforming to the same doctrine.
- **Execution network**: requests traverse a defined sequence of connected layers.

"Network" reflects structured interconnection across languages, processes,
deployment environments, compute nodes, and modules.

### 15.4 Summary

- **Hybrid** describes intentional multi-paradigm integration.
- **Architecture** describes enforced structural doctrine.
- **Network** describes connected, layered, and modular computational composition.

---

## 16. Recommended Folder Structure

CyHAN v2.1 prescribes a **module-first** layout with a defined root-level
integration tier. This structure is recommended for all v2.1-compliant
implementations; conformance remains behavioral per §0.3.

### 16.1 Canonical Layout

```
project-root/
│
├── modules/
│   ├── <name>/
│   │   ├── run_<name>.py                  ← launcher (user-facing)
│   │   ├── README.md
│   │   ├── backend/
│   │   │   ├── engines/
│   │   │   │   └── cpp/                    ← C++ engine + bindings
│   │   │   └── python/
│   │   │       └── main_<name>.py          ← orchestrator (non-user-facing)
│   │   │                                     expands to backend/python/<name>/
│   │   ├── frontend/                       (optional)
│   │   │   ├── desktop/                    (optional, module-scoped Qt)
│   │   │   └── web/                        (optional, module-scoped web)
│   │   ├── tests/
│   │   ├── data/                          (optional; see §16.7)
│   │   │   ├── inputs/
│   │   │   │   ├── raw/                   ← unmodified source inputs
│   │   │   │   └── processed/             ← preprocessed inputs ready for the engine
│   │   │   └── outputs/                   ← engine / orchestrator results
│   │   └── docs/
│   └── <other modules>/
│
├── backend/
│   └── api/
│       └── python/                         ← shared API surface (§6.1)
│
├── frontend/
│   ├── desktop/                            ← shared desktop shell (§6.2, optional)
│   └── web/                                ← shared web frontend (§6.2, optional)
│
├── docs/
│   └── CyHAN-Standard-v2.1.md
│
└── tests/                                  (optional, cross-module)
```

> **Note (v2.1).** `backend/engines/cpp/` is present only for modules with
> performance-critical computation (§5.1); pure-Python and optional-accelerator
> modules omit it and still conform.

### 16.2 Layer Mapping

| Folder | Architectural Layer |
|---|---|
| `modules/<name>/run_<name>.py` | Launcher (user-facing entry, §5.3) |
| `modules/<name>/backend/engines/cpp/` | C++ Engine |
| `modules/<name>/backend/python/main_<name>.py` | Python Orchestration (orchestrator, §5.3) |
| `modules/<name>/backend/python/<name>/` | Python Orchestration (expanded package form) |
| `modules/<name>/frontend/desktop/` | Qt Desktop component (module-scoped) |
| `modules/<name>/frontend/web/` | Web component (module-scoped) |
| `backend/api/python/` | Python API (shared, §6.1) |
| `frontend/desktop/` | Qt Desktop shell (shared, §6.2) |
| `frontend/web/` | Web Frontend (shared, §6.2) |

The filesystem **SHALL** reflect the architectural separation of concerns.

### 16.3 Module Authority

The `modules/` directory is canonical. Each module **SHALL** contain its own
authoritative execution path, enforce Orchestration → Engine flow internally,
remain self-contained per §5.2, and contain no business logic outside its own
boundaries. Modules **SHALL NOT** depend on sibling module source trees.

### 16.4 Engine Isolation

The C++ engine layer within a module **SHALL** be isolated under
`modules/<name>/backend/engines/cpp/`, avoid dependencies on frontend
directories, avoid HTTP or UI imports, and remain portable and independently
testable. Bindings (if used) **SHALL NOT** collapse layer boundaries.

### 16.5 Launcher and Orchestrator Placement

Per §5.3:

- The launcher **SHALL** sit at the module root as `run_<name>.py`, the
  user-facing surface, maximally discoverable.
- The orchestrator **SHALL** sit under `backend/python/` as `main_<name>.py`,
  non-user-facing, importable by the launcher and (when deployed) by the shared
  API.

The module-root placement of the launcher is a deliberate divergence from v1.1
§16.1 (which located it under `scripts/`), chosen so the operator's single
editable file is the first thing visible in the module folder.

### 16.6 Frontend Adapters

Module-scoped frontend directories are adapters to the module's orchestration
role. They MAY include UI rendering, visualization, and local configuration.
They SHALL NOT include orchestration logic, numerical kernels, or alternate
execution paths. Module-scoped frontends are independent of the shared
root-level frontends (§6.2).

### 16.7 Module Data Directories

A module that reads or writes file-based data **SHOULD** use the following
convention for its `data/` directory:

```
modules/<name>/
└── data/
    ├── inputs/
    │   ├── raw/                            ← unmodified source inputs
    │   └── processed/                      ← preprocessed inputs ready for the engine
    └── outputs/                            ← engine and orchestrator results
```

These directories are the **default targets** that the launcher's user-options
block points at when no explicit path is supplied. The operator MAY override any
default by editing the corresponding path in `run_<name>.py`.

The three subdirectory names (`inputs/raw/`, `inputs/processed/`, `outputs/`)
are the standard convention when used. A module **MAY** use only the
subdirectories that apply to its workflow: a module whose engine consumes
source data directly need not create `inputs/processed/`; a module that
produces no file outputs need not create `outputs/`. The convention prescribes
**names**, not workflow shape.

Module `data/` directories **SHOULD** be used for reference fixtures, small
validation cases, and default operator targets. They are not the only allowed
location for working data; an operator MAY redirect any launcher path option to
storage outside the repository.

### 16.8 Structural Doctrine

The filesystem encodes architectural doctrine: modules are self-contained;
engines are isolated; the launcher is user-facing and the orchestrator is not;
orchestration is realized at appropriate scale; the shared API and frontends are
the integration tier. The layout reinforces execution discipline, supports
independent distribution of capabilities, and prevents architectural drift.

### 16.9 Root-Level Integration (Supersedes v1.1 §16.8)

v1.1 §16.8 reserved but forbade root-level `backend/` and `frontend/`
directories. **v2.0 lifts that prohibition.** The root-level integration tier is
now permitted and specified:

```
project-root/
├── backend/
│   └── api/                                ← shared API surface (§6.1)
├── frontend/
│   ├── desktop/                            ← shared desktop shell (§6.2)
│   └── web/                                ← shared web frontend (§6.2)
```

Root-level integration composes modules but **SHALL NOT** be required for any
module to build or run (§6.3).

### 16.10 Optional Extensions

Implementations MAY include additional directories such as:

```
├── common/                                ← shared library for cross-module helpers (§5.2)
├── scripts/
├── infrastructure/
├── docker/
├── ci/
├── config/
```

Such additions **SHALL NOT** violate the canonical execution hierarchy or the
module self-containment requirements of §5.2.

A shared `common/` library (CyHAN §5.2) MAY hold cross-module **presentation and
pure-utility** helpers (e.g. a plotting palette, a shared figure-style or
figure-save helper). It **SHALL NOT** hold module domain logic, numerical
kernels, or orchestration. A module that depends on `common/` remains
independently runnable through its launcher (§6.3); `common/` is an
integration-tier dependency, not a sibling-module source dependency, so it does
not breach the §5.2 prohibition on reaching into a sibling module's tree.

---

## 17. Naming-Pattern Reference

The patterns below are normative as **patterns** (the `<name>`/`<prefix>`
relationships and the `snake_case` rule of §5.4). The concrete artifact names and
extensions shown are illustrative *(non-normative)*: platform suffixes,
framework names, and the `<prefix>` value are implementation choices.

| Artifact | Pattern | Illustrative example *(non-normative)* |
|---|---|---|
| Module directory | `modules/<name>/` | `modules/wave1d/` |
| Launcher | `modules/<name>/run_<name>.py` | `run_wave1d.py` |
| Orchestrator | `modules/<name>/backend/python/main_<name>.py` | `main_wave1d.py` |
| Orchestrator (expanded) | `modules/<name>/backend/python/<name>/` | `.../wave1d/` |
| Python package (if prefixed) | `<prefix>_<name>` | `<prefix>_wave1d` |
| Engine binding module | `_<short>.<py>-<plat>.<ext>` | `_cw1d.cpython-312-...` |
| Qt plugin | `<prefix>_<name>_gui.<ext>` | `<prefix>_wave1d_gui.so` |
| Standalone app | `<prefix>_<name>_app` | `<prefix>_wave1d_app` |
| CMake feature flag | `BUILD_<NAME>` (upper) | `BUILD_WAVE1D` |
| C++ classes | `PascalCase`, short module tag prefix | `Wave1DSolver` |
| Python modules | `snake_case` | `orchestrator.py` |

The short C++ tag may abbreviate `<name>` where the full name would be unwieldy;
stay consistent within a module.

---

## Appendix A: Migration to the v2.0 Layout *(Non-Normative)*

This appendix records a reproducible procedure for migrating a monolithic or
v1-style engine tree (`engines/<name>/`) to the v2.0 module layout
(`modules/<name>/`). It is **non-normative**: nothing here defines conformance,
and the commands shown are illustrative. The procedure is deliberately mechanical.

### A.1 Trigger Conditions (do not start before)

- All in-flight numerical/physics changes for the engine have landed.
- The validation suite passes at its current best baseline.
- The working tree is clean.
- No background scans, probes, or long-running jobs touch the engine.

Mixing substantive changes into a migration muddles attribution if a regression
appears. Keep migration changes surgical and mechanical.

### A.2 File-Move Ledger (template)

Preserve history with a history-aware move (e.g. `git mv`).

| From | To |
|---|---|
| `engines/<name>/cpp/*.{cpp,hpp}` (non-GUI) | `modules/<name>/backend/engines/cpp/` |
| `engines/<name>/cpp/CMakeLists.txt` | `modules/<name>/backend/engines/cpp/CMakeLists.txt` |
| `engines/<name>/cpp/<name>_bindings.cpp` | `modules/<name>/backend/engines/cpp/` |
| `engines/<name>/cpp/gui/*` | `modules/<name>/frontend/desktop/` |
| `engines/<name>/src/<pkg>/*.py` | `modules/<name>/backend/python/` |
| `engines/<name>/run.py` | `modules/<name>/run_<name>.py` |
| orchestration logic (extracted from old run/driver) | `modules/<name>/backend/python/main_<name>.py` |
| `tests/engines/<name>/*` | `modules/<name>/tests/` |
| `engines/<name>/build/` | delete (build artifact) |

Note the v2.0-specific step: any orchestration logic that previously lived in a
run/driver script **MUST** be extracted into `main_<name>.py`; the new
`run_<name>.py` retains only the user-options block and the launch call (§5.3).

### A.3 Build Configuration Changes

When the engine's build file moves under `backend/engines/cpp/`, update install
destinations to be relative to the module root so a freshly cloned tree builds in
place without install-prefix gymnastics. *(Illustrative, e.g. CMake:)*

| Old destination | New destination |
|---|---|
| `.../../src/<pkg>` | `.../../python/` (orchestrator location) |
| GUI plugin path `cpp/plugins/` | `.../../frontend/desktop/plugins/` |
| App-bundle relative paths | walk up to the module root, not the old engine root |

For platform app bundles, verify the runtime path walk reaches the module root
on a fresh build before declaring the step done.

### A.4 Import Strategy

Prefer keeping the public import name stable. Point the package discovery
configuration at `modules/<name>/backend/python/` so external callers do not
change. Avoid layout-leaking import forms such as
`from <name>.backend.python import ...`.

### A.5 Launcher Rewrite

Anchor paths on the module root, not absolute system paths. The launcher resolves
the module root from its own location and derives all paths from there:

```python
from pathlib import Path
ROOT = Path(__file__).resolve().parent      # run_<name>.py sits at module root
# derived: ROOT / "backend" / "python", ROOT / "data", etc.
```

The launcher imports and calls the orchestrator:

```python
from backend.python.main_<name> import run   # entry point
```

### A.6 Validation Scripts

Move validation/probe scripts into `modules/<name>/research/` (or `tests/`).
Replace any hardcoded absolute `sys.path` insertion with module-anchored imports,
or rely on an editable install of the module.

### A.7 Documentation Updates

Update any memory/notes/feedback files that reference old `engines/<name>/...`
paths or specific line numbers. Refresh the module README with new build and run
commands.

### A.8 Smoke Test (run after each commit group)

```bash
# 1. C++ build           (illustrative)
# 2. Imports resolve:     python -c "import backend.python.main_<name>"
# 3. Headless run:        python modules/<name>/run_<name>.py --no-gui
# 4. GUI launches:        python modules/<name>/run_<name>.py
# 5. Validation regression: expect the same baseline error number as before
```

If step 5 drifts, the migration broke something. Bisect against the
pre-migration tag; do not "fix" physics during a migration; revert and reland
the substantive change separately.

### A.9 Commit Strategy and Rollback

Single PR with reviewable commits inside it (scaffolding → engine move →
orchestrator extraction → frontend move → build-path updates → launcher rewrite →
validation/tests → docs). Each commit should leave the tree buildable where
feasible. Because everything is one PR, a single revert to the pre-migration tag
restores the prior state.

### A.10 Out of Scope for a Migration PR

- Engine renames.
- Numerical-scheme or boundary-condition changes.
- Deletion of dead orphan stubs (do that in a preceding cleanup commit).
- Cross-module refactors (e.g. lifting shared code into `common/`).

Keep migration PRs mechanical and reproducible.

---
