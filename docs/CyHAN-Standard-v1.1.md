# C++/Python Hybrid Architecture Network (CyHAN)
# Standard v1.1

---

## Document Control

**Title:** C++/Python Hybrid Architecture Network (CyHAN) Standard v1.1  
**Status:** Revised Release  
**Scope:** Desktop and Cloud Scientific Computing Platforms  
**Audience:** System architects, HPC developers, scientific software engineers, technical leadership  

---

## 1. Purpose

The **C++/Python Hybrid Architecture Network (CyHAN)** (pronounced **“cyan”**) establishes a unified computational architecture with a single canonical execution path and reproducible execution semantics for high-performance scientific and engineering platforms.

This standard defines:

- Layer boundaries  
- Execution topology  
- Module decomposition  
- Responsibility isolation  
- Deployment models  
- Compliance requirements  
- Architectural constraints  

CyHAN is intended for long-lifecycle technical systems where correctness, scalability, and maintainability are critical.

Version 1.1 introduces the **module** as the unit of structural decomposition and distribution, and clarifies the language doctrine: numerical computation is C++-first, with Python employed where it is structurally beneficial (orchestration, workflow control, AI/ML integration). The launcher script is a required artifact of every module.

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

CyHAN is compute-first, not UI-first.

---

## 3. Canonical Architecture

### 3.1 Execution Topology

```
Desktop Client (Qt C++)
or
Web Client (React / Browser)
│
▼
Python API
▼
Python Orchestration
▼
C++ Engines
```

This is the **target execution topology** for a fully integrated CyHAN system. All compute **MUST** traverse this stack when integration is realized.

There SHALL be no alternate execution path in compliant systems.

---

### 3.2 Architectural Ordering Rationale

The ordering reflects:

- Native-first engineering philosophy  
- C++ as numerical authority  
- Desktop as primary development environment  
- Web as scalable distribution interface  

Desktop and Web clients are peers. Neither owns computation.

---

### 3.3 Module Decomposition

CyHAN v1.1 partitions capability into **modules**. Each module is a self-contained vertical that realizes the lower portion of the execution topology — the C++ engine and its Python orchestration — for a single, well-defined capability.

Root-level integration (the shared API surface and shared frontend adapters that sit above modules) is anticipated future work and is OUT OF SCOPE for v1.1. In pre-integration deployments, callers invoke module orchestration directly through the module launcher or programmatic Python imports.

A compliant module SHALL:

- Contain its own engine and orchestration layer  
- Be buildable, runnable, and distributable in isolation  
- Not depend on the source trees of sibling modules  

Module specification appears in §5.

---

## 4. Layer Specifications

The CyHAN architecture defines three backend layers and two frontend layer types. Each is an architectural **role** with defined responsibilities and prohibitions. The artifacts that realize each role are specified per-layer below.

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

#### Reproducibility

Where stochastic modeling is used:

- Random number generators **MUST** be seed-controlled  
- Execution order **SHOULD** be stable  
- Numerical behavior **MUST** be consistent across deployment modes  

This requirement refers to execution reproducibility, not deterministic hazard modeling philosophy.

---

### 4.2 Python Orchestration

#### Role

Python Orchestration is the architectural role responsible for workflow assembly and execution control. Every module SHALL realize this role in Python; the artifact form scales with module complexity.

**Artifact forms.**

- When a module performs workflow assembly, multi-step composition, input validation beyond direct type checks, AI/ML integration, ensemble coordination, or distributed worker management, orchestration **SHALL** be implemented as a dedicated Python package within the module.  
- When a module's workflow reduces to a direct engine invocation, the module launcher script (see §5.3) **SHALL** itself fulfill the orchestration role. A dedicated package is OPTIONAL.

In either form, the orchestration role's responsibilities and prohibitions are invariant.

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

CyHAN v1.1 is **C++-first**. Python is employed where it provides structural benefit — orchestration, workflow control, validation, AI/ML integration — not as a default. The Python footprint of a module SHALL be proportional to the orchestration complexity it warrants.

---

### 4.3 Python API

#### Role

The Python API defines the authoritative system boundary for external clients. It is the architectural seam between clients and module orchestration.

#### Scope in v1.1

In a fully integrated CyHAN system, the API surface **SHALL** be realized at the **root level**, above the module layer, and **SHALL** dispatch into module orchestrators.

The root-level API is anticipated future work and is OUT OF SCOPE for v1.1. In pre-integration deployments, the API role MAY be deferred; callers invoke module orchestration directly via the module launcher script or programmatic Python imports.

#### Responsibilities (when realized)

The Python API **SHALL**:

- Expose stable, documented endpoints  
- Validate input schemas  
- Authenticate and authorize requests  
- Dispatch to module orchestration  
- Manage asynchronous execution  

#### Authority

When realized, all external clients **MUST** communicate exclusively through this boundary.

---

### 4.4 Qt Desktop Application

#### Role

The Qt Desktop Application is a native client adapter.

It **SHALL**:

- Own the GUI event loop  
- Manage window lifecycle  
- Render visualization  
- Communicate with the Python API via HTTP or WebSocket (when API is realized)  

In v1.1, modules MAY ship their own self-contained Qt component under the module's `frontend/desktop/` directory for module-scoped use. A root-level Qt shell that aggregates module GUIs into a unified application is anticipated future work and is OUT OF SCOPE.

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

In v1.1, modules MAY ship their own self-contained web component under the module's `frontend/web/` directory. A root-level web frontend that aggregates module UIs is anticipated future work and is OUT OF SCOPE.

---

## 5. Modules

### 5.1 Definition

A **module** is a self-contained vertical that realizes a single, well-defined CyHAN capability. A module **SHALL** contain:

- Its C++ engine (mandatory)  
- Its orchestration role realization, in the form prescribed by §4.2 (mandatory)  
- Its launcher script (mandatory; see §5.3)  

A module **MAY** additionally contain:

- A dedicated Python orchestration package  
- Module-scoped frontend components (desktop, web, or both)  
- Tests, reference data, and module documentation  

### 5.2 Self-Containment

A module **SHALL**:

- Build independently of sibling modules  
- Run independently of sibling modules  
- Be vendorable as a standalone unit  

A module **SHALL NOT**:

- Reach into the source tree of a sibling module  
- Depend on root-level integration code (such code does not exist in v1.1)  
- Share build state with sibling modules  

Modules MAY depend on a shared common library when one is introduced; such a library is anticipated future work and is OUT OF SCOPE for v1.1.

### 5.3 Launcher Script

Every module **SHALL** ship a launcher script named `run_<name>.py`, where `<name>` is the module identifier. The launcher:

- **SHALL** be invocable from the module root  
- **SHALL** anchor its paths relative to the module root, not to absolute system paths  
- **SHALL** serve as the orchestration entry point when no dedicated orchestrator package is present (per §4.2)  
- **MAY** delegate to a dedicated orchestrator package when one is present  

### 5.4 Naming

The module identifier `<name>`:

- **SHALL** be `snake_case`  
- **SHALL** identify the module end-to-end (directory name, Python package name, launcher script name, any binding or plugin artifacts)  
- **SHOULD** describe the capability, not the implementation technology  

This standard does not prescribe a project-wide prefix or naming style beyond `snake_case` for module identifiers. Implementations MAY adopt their own conventions for derived artifact names (Python package prefixes, binding module names, plugin filenames) provided those conventions are consistent within the implementation.

---

## 6. Execution Standard

### 6.1 Canonical Flow (Target)

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

This is the canonical flow for a fully integrated CyHAN system. All compute **SHALL** follow this sequence.

### 6.2 Pre-Integration Flow (v1.1)

In v1.1 deployments where root-level API integration has not yet been realized, the flow reduces to:

```
Caller (CLI, script, or external Python code)

↓

Module Orchestration (launcher or package, per §4.2)

↓

C++ Engine

↓

Module Orchestration

↓

Caller
```

The orchestration and engine roles **SHALL** be respected; only the API segment is deferred.

---

## 7. Deployment Models

### 7.1 Desktop Mode

```
Desktop Client → API → Module Orchestration → C++ Engine
```

Changing deployment environment **SHALL** require only a configuration change.

### 7.2 Cloud Mode

```
Web Client → API → Module Orchestration → C++ Engine
```

Backend services may be containerized and horizontally scalable.

### 7.3 CLI Mode

```
Command-Line Interface → Module Orchestration → C++ Engine
```

CLI invocation **MAY** bypass the API role but **SHALL NOT** bypass the orchestration role. In modules without a dedicated orchestrator package, the launcher script fulfills this role.

### 7.4 v1.1 Reality

Desktop, Cloud, and CLI deployment modes require root-level API integration to be fully realized. In v1.1, CLI mode against individual modules is the operative deployment posture; full Desktop and Cloud modes are anticipated future work.

---

## 8. Execution Parity

CyHAN enforces **desktop–cloud execution parity**.

Given identical:

- Inputs  
- Configuration  
- Seeds  
- Engine versions  
- Module versions  

Desktop and cloud deployments **SHALL** produce equivalent computational results.

---

## 9. Structural Requirements

Minimum compliant structure:

```
project-root/
modules/
<name>/
backend/
engines/
cpp/
scripts/
run_<name>.py
```

Modules are canonical units of capability.  
Each module is self-contained per §5.

The shared API surface and root-level frontend adapters are anticipated future work and are OUT OF SCOPE for v1.1.

The full recommended layout appears in §16.

---

## 10. Prohibited Architectural Anti-Patterns

The following practices violate CyHAN Standard v1.1:

- Direct Desktop Client → C++ Engine invocation  
- Direct Web Client → C++ Engine invocation  
- Parallel orchestration stacks for desktop and cloud deployments  
- Numerical kernels embedded within UI layers  
- Workflow orchestration logic embedded within C++ engine code  
- Multiple or forked execution paths that bypass orchestration  
- Duplication of backend logic across clients  
- Modules that reach into sibling module source trees  
- Modules whose launcher or build references absolute system paths  
- Use of Python where its presence is purely ceremonial and adds no orchestration value  

---

## 11. Compliance Criteria

A system SHALL be considered CyHAN v1.1 compliant if:

- Heavy numerical computation resides in C++  
- Workflow orchestration is implemented in Python, in the artifact form prescribed by §4.2  
- C++ engine code contains no HTTP, UI, or workflow orchestration logic  
- Each module is self-contained per §5.2  
- Each module ships a launcher script per §5.3  
- When realized, the API surface is the sole client-facing boundary  
- Desktop and Web (when realized) share backend semantics  
- Execution parity is preserved per §8  
- Reproducible execution is supported per §4.1  

Compliance is behavioral. The presence or absence of an orchestrator package is not itself a compliance signal; the question is whether orchestration responsibilities are fulfilled in Python at the appropriate scale.

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

- **MAJOR** — Architectural changes  
- **MINOR** — Backward-compatible enhancements  
- **PATCH** — Clarifications or corrections  

Version 1.0 established the canonical unified execution path.  
Version 1.1 introduces module decomposition as the structural unit and clarifies the C++-first language doctrine.

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

The name **C++/Python Hybrid Architecture Network (CyHAN)** is deliberate.  
Each term conveys a specific structural property of the system.

---

### 15.1 Why "Hybrid"

CyHAN is *hybrid* because it intentionally combines multiple computational paradigms within a single coherent execution model.

The hybridization occurs along three dimensions:

#### Language Hybridization
- **C++** provides high-performance, memory-efficient, numerically intensive computation.
- **Python** provides orchestration, workflow governance, integration, and (when realized) API control.

Each language operates within clearly defined boundaries. Neither replaces the other. C++ is the default for computation; Python is employed where it provides structural benefit.

#### Deployment Hybridization
- Native desktop execution (Qt C++)
- Cloud-based web deployment (React / Browser)
- Command-line execution

All share the same backend execution path.

#### Execution Hybridization
- Compiled native engines for performance
- Dynamic orchestration for flexibility
- Controlled stochastic modeling with reproducible execution semantics

Hybrid does not imply fragmentation.  
It implies intentional integration of complementary capabilities.

---

### 15.2 Why "Architecture"

CyHAN is an *architecture* because it defines structural rules governing:

- Layer boundaries
- Authority separation
- Execution flow
- Responsibility ownership
- Module decomposition
- Compliance constraints

An architecture specifies:

- What each layer may do
- What each layer must not do
- How components interact
- Where authority resides

CyHAN is not a framework or a library.  
It is a structural doctrine governing system composition.

The defining architectural principle is the **single canonical execution path**:

```
Client → Python API → Python Orchestration → C++ Engines
```

All compliant systems must respect this structure.

---

### 15.3 Why "Network"

CyHAN is a *network* because it connects multiple independent components through well-defined boundaries while preserving execution unity.

The network dimension includes:

#### Multi-Client Network
- Qt Desktop client
- Web client
- CLI client
- Automation clients

#### Service Network
- API layer
- Orchestration layer
- Distributed compute workers
- Engine modules

#### Module Network
- Multiple self-contained modules
- Each independently buildable, runnable, and distributable
- All conforming to the same architectural doctrine

#### Execution Network
Requests traverse a defined sequence of connected layers, forming a structured computational network.

"Network" does not imply internet transport alone.  
It reflects structured interconnection across:

- Languages
- Processes
- Deployment environments
- Compute nodes
- Modules

CyHAN systems may operate locally, on-premise, or in distributed cloud environments while maintaining architectural integrity.

---

### 15.4 Summary

- **Hybrid** describes intentional multi-paradigm integration.
- **Architecture** describes enforced structural doctrine.
- **Network** describes connected, layered, and modular computational composition.

Together, the name reflects a unified, execution-consistent system design rather than a collection of tools.

---

## 16. Recommended Folder Structure

CyHAN v1.1 prescribes a **module-first** folder layout. Each module is self-contained and reflects the architectural layers internally.

This structure is recommended for all CyHAN v1.1-compliant implementations.

---

### 16.1 Canonical Layout

```
project-root/
│
├── modules/
│   ├── <name>/
│   │   ├── README.md
│   │   ├── backend/
│   │   │   ├── engines/
│   │   │   │   └── cpp/
│   │   │   └── python/                 (optional, per §4.2)
│   │   ├── frontend/                   (optional)
│   │   │   ├── desktop/                (optional)
│   │   │   └── web/                    (optional)
│   │   ├── scripts/
│   │   │   └── run_<name>.py
│   │   ├── tests/
│   │   ├── data/
│   │   └── docs/
│   └── <other modules>/
│
├── docs/
│   └── CyHAN-Standard-v1.1.md
│
└── tests/                              (optional, cross-module)
```

---

### 16.2 Layer Mapping

Within each module, directories mirror the architectural layers:

| Folder | Architectural Layer |
|---------|--------------------|
| `modules/<name>/backend/engines/cpp/` | C++ Engine |
| `modules/<name>/backend/python/` | Python Orchestration (package form, optional) |
| `modules/<name>/scripts/run_<name>.py` | Python Orchestration (launcher form) and module entry point |
| `modules/<name>/frontend/desktop/` | Qt Desktop component (module-scoped, optional) |
| `modules/<name>/frontend/web/` | Web component (module-scoped, optional) |

The filesystem **SHALL** reflect the architectural separation of concerns within each module.

---

### 16.3 Module Authority

The `modules/` directory is canonical.

Each module **SHALL**:

- Contain its own authoritative execution path  
- Enforce Orchestration → Engine flow internally  
- Remain self-contained per §5.2  
- Contain no business logic outside its own boundaries  

Modules **SHALL NOT** depend on sibling module source trees.

---

### 16.4 Engine Isolation

The C++ engine layer within a module **SHALL**:

- Be isolated under `modules/<name>/backend/engines/cpp/`  
- Avoid dependencies on frontend directories  
- Avoid HTTP or UI imports  
- Remain portable and independently testable  

Bindings (if used) **SHALL NOT** collapse layer boundaries. A binding is a conduit, not an authority.

---

### 16.5 Orchestration Realization

The orchestration role within a module is realized in one of two forms (per §4.2):

- **Package form:** `modules/<name>/backend/python/` containing a Python package that imports the engine bindings and exposes orchestration entry points.  
- **Launcher form:** `modules/<name>/scripts/run_<name>.py` directly orchestrating an engine invocation, with no dedicated package present.

The choice is determined by module complexity, not by stylistic preference.

---

### 16.6 Frontend Adapters

Module-scoped frontend directories are adapters to the module's orchestration role.

They MAY include:

- UI rendering code  
- Visualization utilities  
- Local configuration  

They SHALL NOT include:

- Orchestration logic  
- Numerical kernels  
- Alternate execution paths  

Module-scoped frontends are independent of the (anticipated future) root-level frontend integration.

---

### 16.7 Structural Doctrine

The filesystem is not arbitrary.

It encodes architectural doctrine:

- Modules are self-contained.  
- Engines are isolated.  
- Orchestration is realized at appropriate scale.  
- Backend is authoritative.  
- Frontends are clients.  

The module-first layout reinforces execution discipline, supports independent distribution of capabilities, and prevents architectural drift.

---

### 16.8 Future Work

Root-level integration is anticipated as a future amendment. Implementations SHOULD reserve, but SHALL NOT prematurely create, the following root-level structure:

```
project-root/
├── backend/
│   └── api/                            (future: shared API surface)
├── frontend/
│   ├── desktop/                        (future: integration desktop shell)
│   └── web/                            (future: integration web frontend)
```

Until that amendment, root-level `backend/` and `frontend/` directories SHALL NOT exist. Integration is module-scoped.

---

### 16.9 Optional Extensions

Implementations MAY include additional directories such as:

```
├── scripts/
├── infrastructure/
├── docker/
├── ci/
├── config/
```

Such additions SHALL NOT violate the canonical execution hierarchy or the module self-containment requirements of §5.2.

---
