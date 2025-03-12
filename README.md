# Statistics & modelling
![GitHub last commit](https://img.shields.io/github/last-commit/allenvox/statistics)<br>

Welcome to the `statistics` repository! This repository contains a collection of programming assignments and projects related to statistical modeling, data analysis, and simulation techniques, completed as part of my coursework. The projects span various topics in statistics and probability, implemented using Python, AnyLogic, and other tools. This README provides an overview of the contents, setup instructions, and usage guidelines.

## Overview

This repository is organized into folders, each representing a specific assignment or topic. The projects range from simple statistical simulations to complex models involving machine learning and system dynamics. The work reflects both theoretical understanding and practical implementation skills developed over multiple semesters.

## Repository Structure

The repository contains the following folders:

- **`anylogic-seir`**
  - **Description**: Contains an implementation of the SEIR (Susceptible-Exposed-Infected-Recovered) epidemic model using AnyLogic. The model simulates the spread of infectious diseases with an advanced version including age groups and quarantine effects.
  - **Tools**: AnyLogic.
  - **Key Features**: Multi-agent simulation, dynamic parameters (e.g., quarantine factor), visualization of infection dynamics.

- **`autocorrelation`**
  - **Description**: Explores the autocorrelation of time series data, including calculations and visualizations to analyze the dependency of values over lags.
  - **Tools**: Python, NumPy, Matplotlib.
  - **Key Features**: Autocorrelation function implementation, graphical representation.

- **`critical-path`**
  - **Description**: Implements algorithms to find the critical path in project management or network graphs, useful for optimizing task scheduling.
  - **Tools**: Python.
  - **Key Features**: Critical path method (CPM), network analysis.

- **`distribution-densities`**
  - **Description**: Investigates probability density functions and their visualizations for various statistical distributions (e.g., normal, exponential).
  - **Tools**: Python, Matplotlib, SciPy.
  - **Key Features**: Density plots, parameter variation analysis.

- **`inclusion-matrices`**
  - **Description**: Focuses on the creation and analysis of inclusion matrices, often used in set theory or data structuring.
  - **Tools**: Python, NumPy.
  - **Key Features**: Matrix operations, set representation.

- **`k-means`**
  - **Description**: Implements the K-means clustering algorithm for unsupervised machine learning, including data clustering and visualization.
  - **Tools**: Python, NumPy, Matplotlib, Scikit-learn.
  - **Key Features**: Cluster assignment, centroid computation, visualization of clusters.

- **`markov-chains`**
  - **Description**: Contains a simulation of randomized Markov chains with two 10×10 two-stochastic matrices. The project includes state transition generation, value normalization, frequency analysis, and visualization of behavior and autocorrelation.
  - **Tools**: Python, NumPy, Matplotlib.
  - **Key Features**: Two-stochastic matrices (inertial and uniform), exponential distribution, autocorrelation plots.

- **`predictions`**
  - **Description**: Focuses on predictive modeling, likely involving regression or time series forecasting techniques.
  - **Tools**: Python, Scikit-learn, Matplotlib.
  - **Key Features**: Prediction algorithms, model evaluation metrics.

- **`random-network-graphs`**
  - **Description**: Explores the generation and analysis of random network graphs, including properties like connectivity and degree distribution.
  - **Tools**: Python, NetworkX, Matplotlib.
  - **Key Features**: Graph generation, visualization, statistical properties.

## Setup Instructions

To run the projects in this repository, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/statistics.git
   cd statistics
   ```

2. **Install Dependencies:**
   - Ensure you have Python 3.x installed.
   - Install required libraries using pip:
     ```bash
     pip install numpy matplotlib scipy scikit-learn networkx
     ```
   - For AnyLogic project (`anylogic-seir`), install AnyLogic software (version 8 or higher recommended).

3. **Run the Code:**
   - Navigate to the desired folder (e.g., `cd markov-chains`).
   - Execute the main script (e.g., `python main.py`) if available, or open AnyLogic models directly.

## Usage

- Each folder contains its own scripts or models with specific instructions (if any) in the form of comments or additional README files.
- For Python-based projects, modify parameters (e.g., matrix sizes, step counts) in the code as needed.
- For AnyLogic project, open the `.alp` files in AnyLogic and run the simulations with the provided settings.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
