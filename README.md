# preAD_DPM — Benchmarking Parametric Disease Progression Models for Early Detection of Cognitive Decline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

This repository contains the code used in the paper:

> **Benchmarking parametric models of disease progression for early detection of cognitive decline**  
> Platero C., Bengoa J.  
> *Computer Methods and Programs in Biomedicine*, 2025.  
> DOI: [10.1016/j.cmpb.2025.109162](https://doi.org/10.1016/j.cmpb.2025.109162)  
> Open Access (UPM): [https://oa.upm.es/92018/](https://oa.upm.es/92018/)

---

## Overview

Parametric Disease Progression Models (DPMs) are data-driven tools that characterise the long-term temporal dynamics of biomarkers associated with neurodegenerative disease. This repository implements and benchmarks three established parametric DPM frameworks applied to the early detection of cognitive decline and Alzheimer's disease (AD):

| Folder | Framework | Language | Description |
|--------|-----------|----------|-------------|
| `GRACE/` | GRACE | R | Parametric DPM based on nonlinear mixed-effects modelling with generalized logistic functions |
| `Leaspy/` | Leaspy | Python | Spatiotemporal DPM using Riemannian geometry and variational inference |
| `RPDPM/` | RPDPM | MATLAB | Robust Parametric DPM using M-estimation and modified Stannard functions |

The three models are compared in their capacity to temporally order biomarker progression, align individual disease trajectories, and predict clinical status — evaluated on data from the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu/).

---

## Repository Structure

```
preAD_DPM/
├── GRACE/          # R scripts for the GRACE DPM
├── Leaspy/         # Python scripts for the Leaspy DPM
├── RPDPM/          # MATLAB scripts for the RPDPM
└── README.md
```

---

## Requirements

### GRACE (R)
- R ≥ 4.0
- Packages: `nlme`, `ggplot2`, `dplyr` (see `GRACE/requirements.R` if present)

### Leaspy (Python)
- Python ≥ 3.9
- [Leaspy](https://github.com/aramis-lab/leaspy) ≥ 1.3
- Install dependencies:
  ```bash
  pip install leaspy
  ```

### RPDPM (MATLAB)
- MATLAB ≥ R2020a
- Statistics and Machine Learning Toolbox
- Optimization Toolbox

---

## Data

The experiments in this paper use data from the **Alzheimer's Disease Neuroimaging Initiative (ADNI)**. Access to ADNI data requires registration at [http://adni.loni.usc.edu/](http://adni.loni.usc.edu/). Data cannot be redistributed as part of this repository.

Biomarkers used include volumetric MRI, PET (amyloid/FDG), CSF (Aβ42, tau, p-tau), and cognitive assessments (MMSE, ADAS-Cog, CDR-SB).

---

## Usage

### GRACE (R)

```r
# Navigate to the GRACE folder and run the main script
setwd("GRACE/")
source("main_GRACE.R")
```

### Leaspy (Python)

```python
# Navigate to the Leaspy folder
cd Leaspy/
python main_leaspy.py
```

### RPDPM (MATLAB)

```matlab
% Open MATLAB, navigate to the RPDPM folder, and run:
cd RPDPM
run_RPDPM
```

> Refer to the scripts within each subfolder for parameter configuration and dataset paths.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{platero2025benchmarking,
  title   = {Benchmarking parametric models of disease progression for early detection of cognitive decline},
  author  = {Platero, Carlos and Bengoa, J.},
  journal = {Computer Methods and Programs in Biomedicine},
  year    = {2025},
  doi     = {10.1016/j.cmpb.2025.109162},
  url     = {https://doi.org/10.1016/j.cmpb.2025.109162}
}
```

---

## Related Repositories

Other DPM-related tools from the same research group:

- [RPDPM_MCItoDementia](https://github.com/cplatero/RPDPM_MCItoDementia) — DPM from MCI to Dementia using RPDPM
- [twogrsurvana](https://www.nitrc.org/projects/twogrsurvana/) — Longitudinal survival analysis and two-group comparison for predicting MCI-to-AD progression
- [predict_mci2ad](https://www.nitrc.org/projects/predict_mci2ad/) — Predicting Alzheimer's conversion in MCI patients

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

Data collection and sharing for this project was funded by the Alzheimer's Disease Neuroimaging Initiative (ADNI) (National Institutes of Health Grant U01 AG024904) and DOD ADNI (Department of Defense award number W81XWH-12-2-0012). ADNI is funded by the National Institute on Aging, the National Institute of Biomedical Imaging and Bioengineering, and through generous contributions from the following: AbbVie, Alzheimer's Association; Alzheimer's Drug Discovery Foundation; Araclon Biotech; BioClinica, Inc.; Biogen; Bristol-Myers Squibb Company; CereSpir, Inc.; Cogstate; Eisai Inc.; Elan Pharmaceuticals, Inc.; Eli Lilly and Company; EuroImmun; F. Hoffmann-La Roche Ltd and its affiliated company Genentech, Inc.; Fujirebio; GE Healthcare; IXICO Ltd.; Janssen Alzheimer Immunotherapy Research & Development, LLC.; Johnson & Johnson Pharmaceutical Research & Development LLC.; Lumosity; Lundbeck; Merck & Co., Inc.; Meso Scale Diagnostics, LLC.; NeuroRx Research; Neurotrack Technologies; Novartis Pharmaceuticals Corporation; Pfizer Inc.; Piramal Imaging; Servier; Takeda Pharmaceutical Company; and Transition Therapeutics. The Canadian Institutes of Health Research is providing funds to support ADNI clinical sites in Canada.

---

## Contact

Carlos Platero — Universidad Politécnica de Madrid  
[https://github.com/cplatero](https://github.com/cplatero)
