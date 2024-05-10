# Traffic Flow Analysis using Deep Learning and Visualization

This repository contains the code and documentation for our project titled "Uncovering Temporal and Spatial Relationships Among Objects in Traffic Flow Street Scenes: A Deep Learning and Visualization Approach". Our project utilizes state-of-the-art machine learning models and data visualization techniques to analyze urban traffic flow dynamics from both temporal and spatial perspectives.

## Contributors

- Jiacheng Shen - [shen.patrick.jiacheng@nyu.edu](mailto:shen.patrick.jiacheng@nyu.edu)
- Liyuan Geng - [lg3490@nyu.edu](mailto:lg3490@nyu.edu)

## Project Overview

The rapid evolution of urban transportation demands innovative approaches to traffic management and planning. This project leverages advanced deep learning techniques and the d3.js library to visualize traffic flow interactions, providing insights that help in urban planning and management.

### Key Features

- **Deep Learning Model**: Using LangSAM for object detection and tracking within complex urban street scenes.
- **Data Visualization**: Employing the d3.js library to represent traffic flow interactions dynamically and interactively.
- **Object Tracking**: Implementation of a simple object tracking algorithm to enrich our analysis with temporal data insights.


## Visualization Demos

Interactive visualizations are available through ObservableHQ:
- Temporal Data Visualization: [View Notebook](https://observablehq.com/d/25dde18c8a056bd5)
- Spatial Data Visualization: [View Notebook](https://observablehq.com/d/d2f630741298b9c3)

## Setup and Installation

Instructions on how to set up the project for development and testing purposes.

```bash
git clone https://github.com/lygeng0427/trafficFlowVis.git
cd trafficFlowVis
# install dependencies
pip install -r requirements.txt
```

## Repository Structure

This repository contains several scripts, configuration files, and other elements crucial for the project's functionality. Below is a detailed explanation of the repository's structure:

### Scripts

- `main.py`: The primary script that executes the main functionalities of the project.
- `convert_csv_file.py`: A utility script used to convert data formats, specifically handling CSV file transformations. Recent updates have made GPU count functions operational.
- `read_spatial.py`: Script to read and process spatial data inputs.
- `utils.py`: Contains utility functions that support various operations throughout the project.

### Configuration Files

- `bbox_spatial.yaml`: Configuration settings for bounding box calculations related to spatial data.
- `main.yaml`: Main configuration file that contains essential settings for running the `main.py` script.

### Requirements

- `requirements.txt`: Lists all Python dependencies required to run the scripts in this repository. Ensure you install these dependencies to avoid runtime issues.