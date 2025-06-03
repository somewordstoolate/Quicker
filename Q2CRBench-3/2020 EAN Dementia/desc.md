# 2020 EAN Dementia
## Introduction
The 2020 EAN Dementia dataset is derived from the [2020 European Academy of Neurology(EAN) guideline for dementia](https://onlinelibrary.wiley.com/doi/10.1111/ene.14412). It includes 9 clinical questions along with corresponding development documents organized for analysis and benchmarking.

## Dataset Structure
### Overview
```bash
2020_EAN_Dementia/
│  desc.md                        # Description of this dataset
│  Included_Studies_with_PMID.json # Final studies included in the guideline with PMIDs
│  PICO_Information.json          # Clinical questions with their PICO components
├─ Evidence_Profiles/
│   ├─ metadata                   # Raw evidence profile data
│   ├─ outcomeinfo                # Outcome-level assessment results
│   └─ paperinfo                  # Study characteristics and metadata
├─ Search_Strategies/             # Search strategies used in Ovid MEDLINE
└─ Supplementary_Materials/       # supplementary files
```

### Description

Each clinical question is assigned a `PICO_IDX`, which typically consists of one or two elements. The first element represents a broad research question, while the second element (if present) indicates a more specific sub-question. The relationship between files can be traced through these index values.

If certain downstream files (e.g., evidence profiles) are missing for a given question, this reflects that the question did not progress to subsequent stages of guideline development due to limited or insufficient evidence.

The `Search_Strategies` folder includes the exact queries used to perform literature searches in Ovid MEDLINE. 

The `Included_Studies_with_PMID.json` file lists all studies that were included in the final guideline. PMID identifiers are provided to facilitate retrieval. The `Date` column in this file reflects a search window one year earlier in our experiment than the actual search date in the original guideline, because Ovid MEDLINE does not support exact DD/MM search filtering. This file was used as the label set for the literature search and study selection experiments in Quicker.

In the `Evidence_Profiles` directory:

* The metadata folder contains the raw evidence profiles in the most original form possible.

* The outcomeinfo and paperinfo folders divide these profiles into outcome-level results and study-level metadata, respectively, to facilitate targeted usage.

Finally, the `Supplementary_Materials` folder contains `expert-curated study screening results` and `screened numerical data`, which serve as ground truth for Quicker's evaluation.

