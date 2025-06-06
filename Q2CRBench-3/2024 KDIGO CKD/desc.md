# 2021 ACR RA
## Introduction
The 2024 KDIGO CKD dataset is derived from the [2024 KDIGO guideline for chronic kidney disease](https://www.kidney-international.org/article/S0085-2538(23)00766-4/fulltext). It includes 9 clinical questions along with corresponding development documents organized for analysis and benchmarking.

## Dataset Structure
### Overview
```bash
2024_KDIGO_CKD/
│  desc.md                        # Description of this dataset
│  Included_Studies_with_PMID.json # Final studies included in the guideline with PMIDs
│  PICO_Information.json          # Clinical questions with their PICO components
├─ Evidence_Profiles/
│   ├─ metadata                   # Raw evidence profile data
│   ├─ outcomeinfo                # Outcome-level assessment results
│   └─ paperinfo                  # Study characteristics and metadata
│  Sreened_Records/               # Records searched with expert screening label
│  Search_Strategies/             # Search strategies used in PubMed MEDLINE
└─ Supplementary_Materials/       # supplementary files
```

### Description

Each clinical question is assigned a `PICO_IDX`, which typically consists of one or two elements. The first element represents a broad research question, while the second element (if present) indicates a more specific sub-question. The relationship between files can be traced through these index values.

If certain downstream files (e.g., evidence profiles) are missing for a given question, this reflects that the question did not progress to subsequent stages of guideline development due to limited or insufficient evidence.

The `Search_Strategies` folder includes the exact queries used to perform literature searches in PubMed MEDLINE. 

The `Included_Studies_with_PMID.json` file lists all studies that were included in the final guideline. PMID identifiers are provided to facilitate retrieval. The `Date` column in this file reflects a search window one year earlier in our experiment than the actual search date in the original guideline, because Ovid MEDLINE does not support exact DD/MM search filtering. This file was used as the label set for the literature search and study selection experiments in Quicker.

In the `Evidence_Profiles` directory:

* The metadata folder contains the raw evidence profiles in the most original form possible.

* The outcomeinfo and paperinfo folders divide these profiles into outcome-level results and study-level metadata, respectively, to facilitate targeted usage.

Finally, the `Supplementary_Materials` folder contains `expert-curated study screening results` which serve as ground truth for Quicker's evaluation.


