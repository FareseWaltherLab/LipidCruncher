# 🧬 LipidCruncher

**From complex lipidomics data to biological insights—no bioinformatics expertise required.**

Built by [The Farese & Walther Lab](https://www.mskcc.org/research/ski/labs/farese-walther) at Memorial Sloan Kettering Cancer Center.

[![Live App](https://img.shields.io/badge/🚀_Try_It-lipidcruncher.org-blue)](https://lipidcruncher.org)
[![Paper](https://img.shields.io/badge/📄_Paper-bioRxiv-green)](https://www.biorxiv.org/content/10.1101/2025.04.28.650893v2.article-metrics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ Why LipidCruncher?

| | |
|---|---|
| 📂 **4 Data Formats** | LipidSearch, MS-DIAL, Metabolomics Workbench, Generic CSV |
| 🔬 **QC + Normalization** | Integrated quality control with flexible normalization options |
| 📊 **Lipid-Specific Viz** | Saturation profiles, pathway maps, fatty acid composition heatmaps, and more |
| 📈 **High-Quality Outputs** | Interactive plots with SVG export and PDF reports |

---

## 🚀 How It Works

### Step 1: Standardize, Filter & Normalize
Import your data, define experimental conditions, and apply automatic standardization, filtering, and normalization (internal standards, protein concentration, or both). Internal standards consistency plots verify sample prep quality.

### Step 2: Quality Check & Analysis
Validate your data with box plots, BQC coefficient-of-variation analysis, correlation heatmaps, and PCA—then explore results with bar/pie charts, volcano plots, saturation profiles, metabolic pathway maps, clustered heatmaps, and fatty acid composition analysis. All on a single integrated page.

---

## 🧪 Try It Out

**No dataset?** Click "Load Sample Data" in the app to instantly try LipidCruncher with built-in test datasets for all supported formats (LipidSearch, MS-DIAL, Generic). No download required.

Sample datasets are also available in the [sample_datasets](https://github.com/FareseWaltherLab/LipidCruncher/tree/main/sample_datasets) folder.

---

## 💻 Local Installation
```bash
# Clone and setup
git clone https://github.com/FareseWaltherLab/LipidCruncher.git
cd LipidCruncher
python -m venv lipidcruncher_env
source lipidcruncher_env/bin/activate  # Windows: lipidcruncher_env\Scripts\activate
pip install -r requirements.txt

# Run
streamlit run src/main_app.py
```

**System requirement:** Install [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) for PDF export.
- Ubuntu/Debian: `sudo apt-get install poppler-utils`
- macOS: `brew install poppler`

---

## 🏗️ Architecture

LipidCruncher uses a layered architecture with strict separation of concerns:

```
UI Layer          →  Streamlit components (src/app/ui/)
Adapter Layer     →  Session state & caching (src/app/adapters/)
Workflow Layer    →  Pipeline orchestration (src/app/workflows/)
Services Layer    →  Pure business logic (src/app/services/)
Models Layer      →  Pydantic data models (src/app/models/)
```

All business logic (services and workflows) is **pure Python with no Streamlit dependency**, making it fully testable in isolation. The adapter layer is the sole bridge between the UI framework and business logic.

## 📁 Project Structure

```
src/
├── main_app.py                  # Entry point
└── app/
    ├── constants.py             # Shared constants and thresholds
    ├── models/                  # Immutable Pydantic models
    ├── services/                # Stateless business logic
    │   ├── data_cleaning/       #   Format-specific data cleaning
    │   ├── plotting/            #   13 visualization services
    │   ├── normalization.py     #   Internal standard / protein normalization
    │   ├── quality_check.py     #   Box plots, BQC, correlation, PCA
    │   └── statistical_testing.py  # Parametric & non-parametric tests
    ├── workflows/               # Multi-step pipeline orchestration
    ├── adapters/                # Streamlit session state & caching
    └── ui/                      # Streamlit UI components
        ├── sidebar/             #   Input controls
        ├── main_content/        #   Processing, QC, and analysis views
        └── content/             #   Static documentation text
tests/
├── unit/                        # 31 service & model tests
├── integration/                 # 3 end-to-end pipeline tests
└── ui/                          # 5 UI component tests
```

For detailed developer documentation, see `CODEBASE.md`.

---

## 🧪 Testing

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# With coverage
pytest tests/ --cov=src/app --cov-report=term-missing
```

---

## 📦 Dependencies

**Core:** streamlit, pandas, numpy, scipy, scikit-learn, statsmodels  
**Visualization:** plotly, matplotlib, seaborn, bokeh, kaleido  
**Documents:** pillow, reportlab, svglib, pdf2image, openpyxl

See `requirements.txt` for versions.

---

## 📧 Support

Questions, bugs, or feature requests? **abdih@mskcc.org**

---

## 📄 Citation

If you use LipidCruncher in your research, please cite our [paper](https://www.biorxiv.org/content/10.1101/2025.04.28.650893v2.article-metrics).

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.