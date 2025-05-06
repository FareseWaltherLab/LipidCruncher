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
LipidCruncher is built with:
* Python 3.8+
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Plotly
* Scikit-learn
* PDF2Image
* ReportLab

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

## Support
For bug reports, feature requests, or questions:
- Email: abdih@mskcc.org

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Learn More
To explore LipidCruncher and see how we used it for a case study, read our [paper](https://www.biorxiv.org/content/10.1101/2025.04.28.650893v1) published on bioRxiv.

## Citation
If you use LipidCruncher in your research, please cite:
```
Hamza, A., et al. (2025). LipidCruncher: An open-source web platform for processing, visualizing, and analyzing lipidomic data. bioRxiv. https://www.biorxiv.org/content/10.1101/2025.04.28.650893v1
```

## Status
LipidCruncher is actively maintained and updated. The web application is fully functional and available at [https://lipidcruncher.org](https://lipidcruncher.org). For additional methodological details and example applications, please refer to our [paper](https://www.biorxiv.org/content/10.1101/2025.04.28.650893v1).