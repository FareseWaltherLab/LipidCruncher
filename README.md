# LipidCruncher
An open-source, web-based platform for processing, visualizing, and analyzing lipidomic data developed by the Farese and Walther Lab at Memorial Sloan Kettering Cancer Center.

## About
LipidCruncher is a comprehensive tool designed to streamline the analysis of mass spectrometry-based lipidomics data. It addresses traditional challenges like manual spreadsheet handling and insufficient quality assessment by providing researchers with an intuitive interface for data processing, quality control, and visualization, making complex lipidomic analysis more accessible and efficient.

## Access
Access the current version online: [https://lipidcruncher.org](https://lipidcruncher.org)

## Key Features
LipidCruncher organizes the lipidomics analysis pipeline into three integrated modules:

### Module 1: Data Input, Standardization, Filtering, and Normalization
* **Versatile Data Input**: Import datasets from LipidSearch, Metabolomics Workbench, or generic CSV formats
* **Data Standardization**: Automatically align column naming for consistency across datasets
* **Filtering**: Clean data by removing empty rows, duplicates, and replacing null values with zeros
* **Normalization**: Choose from four options (none, internal standard-based, protein-based, or combined)

### Module 2: Quality Check and Anomaly Detection
* **Box Plots**: Visualize concentration distributions to confirm uniformity among replicates
* **Zero-Value Analysis**: Identify samples with excessive missing data
* **CoV Analysis**: Assess measurement precision using batch quality control (BQC) samples
* **Correlation Analysis**: Identify outliers through pairwise correlations between replicates
* **PCA**: Visualize sample clustering and flag potential outliers

### Module 3: Data Visualization, Interpretation, and Analysis
* **Bar and Pie Charts**: Display lipid class concentrations and proportional distributions
* **Metabolic Network Visualization**: Map lipid class changes in a metabolic context
* **Saturation Profiles**: Analyze fatty acid saturation levels (SFA, MUFA, PUFA)
* **Volcano Plots**: Highlight significant lipid changes with customizable thresholds
* **Heatmaps**: Provide a high-resolution view of lipidomic alterations with interactive features

## Dependencies
LipidCruncher requires Python 3.8+ and the following packages:

### Core Dependencies
* streamlit==1.22.0
* pandas==1.5.3
* numpy==1.24.3
* matplotlib==3.7.1
* plotly==5.18.0
* scikit-learn==1.3.0
* scipy==1.13.1
* statsmodels==0.14.0

### Visualization Dependencies
* seaborn==0.12.2
* bokeh==2.4.3
* kaleido==0.2.1

### Document Processing Dependencies
* pillow==9.5.0
* reportlab==3.6.12
* svglib==1.5.1
* pdf2image==1.17.0

### Additional Dependencies
* openpyxl==3.1.2
* selenium==4.10.0

### System Dependencies
For PDF processing functionality, you'll also need to install Poppler:
- **Ubuntu/Debian**: `sudo apt-get install poppler-utils`
- **macOS**: `brew install poppler`
- **Windows**: Download and install from [poppler releases](https://github.com/oschwartz10612/poppler-windows/releases)

## Installation and Local Deployment
### Prerequisites
* Python 3.8 or higher
* Git

### Clone the Repository
```bash
git clone https://github.com/FareseWaltherLab/LipidCruncher.git
cd LipidCruncher
```

### Set Up Environment and Install Dependencies
```bash
# Create and activate virtual environment
python -m venv lipidcruncher_env
source lipidcruncher_env/bin/activate  # On Windows: lipidcruncher_env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Run Locally
```bash
streamlit run main_app.py
```

## Status
LipidCruncher is actively maintained and updated. The web application is fully functional and available at [https://lipidcruncher.org](https://lipidcruncher.org). For additional methodological details and example applications, please refer to our [paper](https://www.biorxiv.org/content/10.1101/2025.04.28.650893v1).

## Test Dataset
Want to explore LipidCruncher but don't have your own dataset? Try our sample datasets available on the lab's [GitHub](https://github.com/FareseWaltherLab/LipidCruncher/tree/main/sample_datasets). These datasets are from the case study analyzed in our [paper](https://www.biorxiv.org/content/10.1101/2025.04.28.650893v1) describing LipidCruncher's features. The LipidSearch dataset is the original dataset containing raw, unprocessed values. Additionally, we provide the same data in Generic and Metabolomics Workbench formats, which include pre-normalized values for your convenience.

The case study experiment includes three conditions with four replicates each:
* **WT** (Wild Type): samples s1-s4
* **ADGAT-DKO** (Double Knockout): samples s5-s8
* **BQC** (Batch Quality Control): samples s9-s12

## Support
For bug reports, feature requests, or questions:
- Email: lipidcruncher@gmail.com

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.