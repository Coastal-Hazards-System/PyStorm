# PyStorm

### *A Modular Python Framework for Probabilistic Modeling of Coastal Storm Hazards*

---

## 🌊 Overview

**PyStorm** is an open-source, modular framework for probabilistic modeling and analysis, designed for engineers, scientists, and researchers working on coastal hazard quantification, stochastic storm simulation, life cycle analysis, and extreme value analysis. It is being devloped a Python-based platform supporting both graphical and script-drive workflows for flexible and computationally efficient execution. Future versions of PyStorm will be modular, extensible, and feature a graphical user interface (GUI) for advanced coastal hazard analysis. 

---
## *(Draft)* Canonical Layout
```
pystorm/
│
├── backend/
│   ├── api/                         # Python API layer — authoritative system boundary
│   │   ├── routes/                  # Endpoint definitions
│   │   ├── schemas/                 # Input/output validation (pydantic, etc.)
│   │   ├── auth/                    # Authentication and authorization
│   │   └── middleware/              # Request lifecycle hooks
│   │
│   ├── orch/                        # Python Orchestration — workflow assembly
│   │   ├── workflows/               # Named workflow definitions (e.g., surge_run.py)
│   │   ├── jobs/                    # Job lifecycle management
│   │   ├── dispatch/                # Engine call coordination
│   │   └── postproc/                # Lightweight post-processing, metadata enrichment
│   │
│   └── engines/                     # Python Compute Engines — numerical authority
│       ├── surge/                   # Storm surge simulation engine
│       ├── metamodel/               # Metamodel inference and prediction
│       ├── hazard/                  # Hazard curve construction, AEF computation
│       ├── sampling/                # Experimental design, k-medoids, LHS
│       └── weights/                 # DSW / JPM weighting schemes
│
├── frontend/
│   ├── desktop/                     # Qt desktop client (if/when added)
│   └── web/                         # React web client (if/when added)
│
├── cli/                             # CLI entry points (bypasses API, not orchestration)
│   └── run_pipeline.py
│
├── config/                          # Environment and run configuration
│   ├── defaults.yaml
│   └── schema.yaml
│
├── tests/
│   ├── engines/                     # Unit tests isolated to compute logic
│   ├── orch/                        # Integration tests for workflow assembly
│   └── api/                         # API contract tests
│
├── docs/
│   ├── CyHAN-Standard-v1.0.md
│   └── architecture.md
│
├── scripts/                         # Dev/ops utilities, not in execution path
│
└── pyproject.toml
```
