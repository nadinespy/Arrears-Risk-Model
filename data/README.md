# Data

The pipeline expects three input files in this directory. **None are committed to the repo** (the directory is gitignored except for this file), so fetch them yourself before running training or the notebooks.

## Expected layout

```
data/
├── README.md
├── household_data.xlsx
├── File_1_IoD2025_Index_of_Multiple_Deprivation.xlsx
└── lewisham_lsoa_to_ward_best-fit_lookup.geojson
```

## Files

### `household_data.xlsx`

17,831 Lewisham households across 21 variables (demographics, income, housing, benefits, arrears).

- **Source:** [`lb-lewisham/data-scientist-interview-task`](https://github.com/lb-lewisham/data-scientist-interview-task) on GitHub (file `data/household_data.xlsx`).
- **Licence:** **none specified** in the source repository (no `LICENSE` file, no in-file metadata). Treat as published for reference/educational use only — do not redistribute.
- **Population caveat:** the dataset covers ~14% of Lewisham's ~130,000 dwellings (Census 2021). Composition (92.5% not in work, 61% on Universal Credit, variables drawn from the benefits system) indicates this is a subset of households known to the council's benefits/housing system, not the borough population as a whole. Any pattern found describes this population, not Lewisham in general. See `notebooks/01_eda_and_imd.ipynb` for the inference behind this.

### `File_1_IoD2025_Index_of_Multiple_Deprivation.xlsx`

IMD25 ranks and deciles for all 33,755 Lower-layer Super Output Areas (LSOAs) in England.

- **Source:** Ministry of Housing, Communities and Local Government, *English Indices of Deprivation 2025*, available on [gov.uk](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025) (search "English Indices of Deprivation 2025" if the URL has changed).
- **Licence:** Open Government Licence v3.0.
- **Sheet used:** `IMD25` (the `Notes` sheet in this copy is empty).
- **Join key:** `lsoa21cd` — joining to `household_data.xlsx` filters to Lewisham's 175 LSOAs and attaches `IMD Rank` (1 = most deprived in England) and `IMD Decile` (1 = most deprived 10% of English LSOAs).

### `lewisham_lsoa_to_ward_best-fit_lookup.geojson`

Polygon boundaries for Lewisham's 175 LSOAs with a ward best-fit lookup (`wd25cd`, `wd25nm`).

- **Source:** distributed in [`lb-lewisham/data-scientist-interview-task`](https://github.com/lb-lewisham/data-scientist-interview-task). Upstream is the [ONS Open Geography Portal](https://geoportal.statistics.gov.uk/) (LSOA 2021 boundaries; ward best-fit lookup).
- **Licence:** Open Government Licence v3.0; contains OS data © Crown copyright and database right.
- **Used for:** choropleth maps in `notebooks/01_eda_and_imd.ipynb`.

## On reproducibility

Because `household_data.xlsx` has no licence, the analysis cannot be reproduced from this repository alone — you need to fetch the file from the source repo first. Tests in `tests/` rely on synthetic fixtures rather than the real data, so unit tests and CI run without it.
