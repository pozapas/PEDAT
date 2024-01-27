# PEDAT - Pedestrian Volume Data Visualization Dashboard

## Overview
PEDAT is a Streamlit-based web application designed to visualize pedestrian volume data in Utah. It offers an interactive and user-friendly interface to analyze and understand pedestrian traffic patterns effectively.

## Features
- **Interactive Data Visualization:** Utilizes libraries like Folium, Plotly, and Kepler.gl for dynamic and engaging data presentation.
- **Comprehensive Data Analysis:** Integrates with Pandas, and *Google Cloud BigQuery* for efficient data handling, manipulation, and analysis.
- **Customizable Views:** Offers various visualization options to cater to different analysis needs.
- **Customizable Report:** Ability to generate detailed reports for selected traffic signals and specific parameters, aiding in targeted analysis and decision-making.
  
## Installation

### Requirements
- Python 3.6 or higher
- Required Python libraries:
  - Streamlit >= 1.13.0
  - Folium >= 0.13, <0.15
  - Pandas, Plotly, Pydeck, DateTime, Matplotlib, NumPy
  - Google Cloud BigQuery, Google Auth
  - Other dependencies as listed in `requirements.txt`

### Steps
1. Clone the repository:
```bash
git clone https://github.com/pozapas/PEDAT.git
```
```bash
cd PEDAT
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
5. Run the Streamlit app:
```bash
streamlit run dash_beta.py
```
## Manuals and Guides
In the `manual` folder of the repository, you will find detailed guides to help you get started and make the most out of PEDAT:
- **PEDAT User Guide:** A comprehensive guide to using the PEDAT app, detailing all features and functionalities.
- **How to Dockerize PEDAT App:** Instructions on how to dockerize the PEDAT application for easy deployment and scalability.
- **How to Install PEDAT App from Docker Hub:** Step-by-step guide for installing the PEDAT app using the Docker image from Docker Hub.
  
## Docker Support
PEDAT is also available as a Docker image. To use it:
1. Pull the Docker image from Docker Hub:
```bash
docker pull pozapas/pedat
```
3. Run the Docker container:
```bash
docker run -p 8501:8501 pozapas/pedat
```

## Contributing
Contributions to PEDAT are welcome. Please read our guidelines and submit your pull requests or issues through GitHub.

## License
[MIT License](https://opensource.org/licenses/MIT)
