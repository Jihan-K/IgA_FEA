# IgA_FEA: Pyrimidine Metabolic Flux Estimation in IgA Nephropathy

IgA_FEA estimates sample-wise metabolic flux through the **pyrimidine metabolism pathway** from RNA-seq gene expression of IgA nephropathy (IgAN) and control samples. It is adapted from [scFEA](https://github.com/changwn/scFEA) (single-cell Flux Estimation Analysis).

## Background: scFEA

Metabolomics cannot easily be measured at fine resolution, but enzyme-coding gene expression can. scFEA ([Alghamdi et al., *Genome Research* 2021](https://genome.cshlp.org/content/31/10/1867)) infers the fluxome from transcriptomic data by combining:

- **A metabolic network** described as a stoichiometry matrix (metabolites × reactions/modules) and a mapping from reactions to enzyme genes.
- **A flux balance model.** At steady state, each intermediate metabolite should be produced and consumed at the same rate, so flux estimates that leave metabolites imbalanced are penalized.
- **A graph neural network solver.** Each reaction gets its own small multi-layer neural network that maps the expression of its enzyme genes to a reaction rate. This captures the non-linear dependency between gene expression and flux. All networks are trained jointly under the flux balance constraints, without any measured flux as a label.

The original scFEA works on single-cell data with a human/mouse metabolic map organized into 168 modules. The predicted fluxome can be used for downstream analyses such as comparing metabolic states between cell groups, finding metabolic imbalance (metabolite stress), and assessing how much each enzyme gene affects flux.

## What this project changes

| | scFEA (upstream) | IgA_FEA (this repo) |
|---|---|---|
| Input | Single-cell RNA-seq (cells as columns) | Bulk RNA-seq samples: 3 control and 3 IgAN mouse kidneys |
| Network unit | Metabolic **modules** (`module_gene_*.csv`) | Individual **reactions**, from a gene–reaction rule (GRR) file |
| Network scope | Whole metabolic map (168 modules) | Pyrimidine metabolism (43 reactions, 66 metabolites) |
| Gene logic | Genes listed per module | GRR with OR (isozymes, one gene per line) and AND (subunits, `Gene1 and Gene2`) |
| Reaction direction | Non-negative flux for all modules | Per-reaction direction (`rxnDir`: forward, reversible, reverse) |
| Expression weighting | None | Tissue-specificity up-weighting of top-expressed genes per sample |
| Metabolite names | Separate `cName` file | Taken from the stoichiometry matrix row names |

## Model

For each sample, the network outputs a flux value for each of the 43 reactions. The stoichiometry matrix then turns those fluxes into a net balance for each metabolite. Training minimizes four loss terms:

| Loss | What it penalizes | Weight (default) |
|---|---|---|
| `loss1` balance | Squared net production/consumption of intermediate metabolites (those in more than one reaction) | `LAMB_BA = 1` |
| `loss2` negative | Flux whose sign contradicts the reaction's direction in `rxnDir`. Reversible reactions are not penalized. | `LAMB_NG = 1` |
| `loss3` cellVar | Mismatch between a sample's total flux and its overall expression level | `LAMB_CELL = 0` (disabled) |
| `loss4` reactionVar | Weak correlation between a sample's reaction fluxes and the summed expression of each reaction's genes (`1 − abs(pearson r)`) | `LAMB_RXN = 1` |

Each reaction network is `Linear(n_genes → 8) → Tanhshrink → Linear(8 → 1) → Tanhshrink`. It is trained with Adam (learning rate 0.008), using all samples as one batch.

## Repository structure

```
IgA_FEA/
├── src/
│   ├── IgA_FEA.py      # Main script: load, preprocess, train, save results
│   ├── ClassFlux.py    # FLUX network (one sub-network per reaction)
│   ├── DatasetFlux.py  # PyTorch Dataset wrapper
│   └── util.py         # Differentiable Pearson correlation for torch tensors
├── data/
│   ├── pyrimidine_grr_path.csv  # Gene–reaction rules
│   └── pyrimidine_matrix.csv    # Stoichiometry matrix with reaction directions
├── input/
│   └── IgA_all.csv     # Gene expression count matrix
└── output/             # Results
```

## Input files

### Gene expression: `input/IgA_all.csv`

This is a count matrix of 22,564 mouse genes (rows) × 6 samples (columns). The first column holds gene symbols. Raw or normalized counts both work. If the maximum value is above 50, the script applies `log2(x + 1)`.

| Group | Samples |
|---|---|
| Control | `Control1`, `Control2`, `Control3` |
| IgAN | `IgA1`, `IgA2`, `IgA3` |

### Gene–reaction rules: `data/pyrimidine_grr_path.csv`

This file has no header. Each line is `pathway,reaction,gene`:

```
pyrimidine,R_URIt,Slc29a2
pyrimidine,R_URIt,Slc29a1
pyrimidine,R_3AIB_Dtm,
```

- If a reaction has several lines, each line is an alternative enzyme (OR).
- `GeneA and GeneB` in one line means those subunits are all required (AND). They are replaced by their mean expression.
- An empty gene field means the reaction has no associated gene. Its expression input is zero (10 of the 43 reactions).
- Genes listed in the GRR but missing from the expression data are added with zero expression, and their names are printed.

### Stoichiometry matrix: `data/pyrimidine_matrix.csv`

```
,R_3AIB_Dtm,R_r0921,R_URIt,...
rxnDir,1,1,1,...
M_2dr1p_c,0,0,0,...
```

- **Header:** reaction IDs (43 reactions).
- **Row `rxnDir`:** reaction direction. `1` means forward only, `0` reversible, `-1` reverse only. Currently all reactions are `1`.
- **Remaining rows:** stoichiometric coefficients of each metabolite (66 metabolites). Negative means consumed and positive means produced.

Reaction and metabolite IDs follow Recon3D-style naming (`R_*`, `M_*_c` / `_e` / `_l` / `_m` for cytosol, extracellular, lysosome and mitochondria).

Every reaction in the matrix must have a rule in the GRR file, or the script stops with an error. Reactions that appear only in the GRR are skipped with a warning.

## Installation

You need Python 3 and the following packages:

```bash
pip install torch numpy pandas matplotlib tqdm
# only needed for --sc_imputation True
pip install magic-impute
```

Tested with torch 1.12.1, pandas 1.5.3 and numpy 1.23.3. A CUDA GPU is used automatically if available. The default dataset runs quickly on a CPU.

## Usage

Run the script from the `IgA_FEA/` folder, because `data`, `input` and `output` are resolved relative to the current directory:

```bash
cd IgA_FEA
python src/IgA_FEA.py
```

This trains for 100 epochs using `input/IgA_all.csv`, `data/pyrimidine_grr_path.csv` and `data/pyrimidine_matrix.csv`.

To see all options:

```bash
python src/IgA_FEA.py --help
```

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `data` | Folder with the GRR and stoichiometry files |
| `--input_dir` | `input` | Folder with the expression file |
| `--res_dir` | `output` | Output folder (created if missing) |
| `--test_file` | `IgA_all.csv` | Expression matrix (genes × samples) |
| `--grr_path` | `pyrimidine_grr_path.csv` | Gene–reaction rule file |
| `--stoichiometry_matrix` | `pyrimidine_matrix.csv` | Stoichiometry matrix file |
| `--train_epoch` | `100` | Number of training epochs |
| `--sc_imputation` | `False` | Apply MAGIC imputation (intended for sparse single-cell data, not needed for bulk) |
| `--output_flux_file` | `NULL` | Custom path for the flux output |
| `--output_balance_file` | `NULL` | Custom path for the balance output |

A plain file name is looked up inside the corresponding folder. A value containing `/` is used as a path as-is.

Example:

```bash
python src/IgA_FEA.py --train_epoch 200 --res_dir output/epoch200
```

## Preprocessing steps

1. **Tissue-specificity weighting.** Within each sample, genes ranked in the top 10% by expression are multiplied by 4 and genes in the top 10–20% are multiplied by 2 (`tissueSpecific`).
2. **Log transform.** If the maximum value is above 50, `log2(x + 1)` is applied.
3. **Gene matching.** Only genes in the GRR are kept. Missing GRR genes are added as zeros.
4. **Reaction inputs.** For each reaction, all genes outside its rule are masked to zero, and AND rules are applied.
5. **Noise.** Small uniform noise `U(0, 0.1)` is added to the expression inputs.

## Output files

Each run writes four files to `--res_dir`, tagged with the run time (`YYYYMMDD-HHMMSS`):

| File | Content |
|---|---|
| `flux_<time>.csv` | Predicted flux, samples × 43 reactions |
| `balance_<time>.csv` | Net balance (metabolite stress), samples × 66 metabolites |
| `lossValue_<time>.txt` | Per-epoch values of `loss1`–`loss4` and the total loss |
| `loss_<time>.png` | Convergence curves of each loss term |

**Reading the results:**

- **Flux:** a positive value means the reaction runs in its written direction. The values are relative, so compare them between samples or groups (for example IgAN vs control) rather than reading them as absolute rates.
- **Balance:** a value away from zero means the model predicts that metabolite accumulates (positive) or is depleted (negative) in that sample.
- **Convergence:** check `loss_<time>.png` before using the results. The total loss and the balance loss should decrease and level off. If they have not, increase `--train_epoch`.

`output/flux.csv` and `output/balance.csv` are results from an earlier run.

## Reproducibility and settings

`SEED = 62424` fixes both the NumPy random state (input noise) and the PyTorch random state (network initialization). The learning rate, loss weights, tissue-specificity weights and seed are constants at the top of [src/IgA_FEA.py](src/IgA_FEA.py).

## Citation

This code is based on scFEA. If you use it, please cite:

> N. Alghamdi, W. Chang, P. Dang, X. Lu, C. Wan, et al., C. Zhang. A graph neural network model to estimate cell-wise metabolic flux using single-cell RNA-seq data. *Genome Research* 31, 1867–1884 (2021). https://doi.org/10.1101/gr.271205.120

Original code: https://github.com/changwn/scFEA
