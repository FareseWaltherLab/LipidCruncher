# 🧬 LipidCruncher

**From raw lipidomics data to biological insights—no bioinformatics expertise required.**

Built by [The Farese & Walther Lab](https://www.mskcc.org/research/ski/labs/farese-walther) at Memorial Sloan Kettering Cancer Center.

[![Live App](https://img.shields.io/badge/🚀_Try_It-lipidcruncher.org-blue)](https://lipidcruncher.org)
[![Paper](https://img.shields.io/badge/📄_Paper-bioRxiv-green)](https://www.biorxiv.org/content/10.1101/2025.04.28.650893v1)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ Why LipidCruncher?

| | |
|---|---|
| 📂 **4 Data Formats** | LipidSearch, MS-DIAL, Metabolomics Workbench, Generic CSV |
| 🔬 **QC + Normalization** | Integrated quality control with flexible normalization options |
| 📊 **Lipid-Specific Viz** | Saturation profiles, pathway maps, FACH analysis, and more |
| 📈 **Publication-Ready** | Interactive plots with SVG export and PDF reports |

---

## 🚀 How It Works

### Module 1: Standardize & Normalize
Import your data, define experimental conditions, and apply automatic standardization, filtering, and normalization (internal standards, protein concentration, or both). Internal standards consistency plots verify sample prep quality.

### Module 2: Quality Check
Validate your data before analysis. Box plots confirm normalization success, CoV analysis assesses BQC precision, and correlation heatmaps + PCA detect outliers.

### Module 3: Visualize & Analyze
Explore your lipidome with bar/pie charts, volcano plots, saturation profiles, metabolic pathway maps, clustered heatmaps, and fatty acid composition analysis—all interactive.

---

## 🧪 Try It Out

**No dataset?** Use our [sample data](https://github.com/FareseWaltherLab/LipidCruncher/tree/main/sample_datasets) from the ADGAT-DKO case study:
- **WT** (Wild Type): samples s1–s4  
- **ADGAT-DKO** (Double Knockout): samples s5–s8  
- **BQC** (Batch Quality Control): samples s9–s12

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

## 📦 Dependencies

**Core:** streamlit, pandas, numpy, scipy, scikit-learn, statsmodels  
**Visualization:** plotly, matplotlib, seaborn, bokeh, kaleido  
**Documents:** pillow, reportlab, svglib, pdf2image, openpyxl

See `requirements.txt` for versions.

---

## 📧 Support

Questions, bugs, or feature requests? **lipidcruncher@gmail.com**

---

## 📄 Citation

If you use LipidCruncher in your research, please cite our [paper](https://www.biorxiv.org/content/10.1101/2025.04.28.650893v1).

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.