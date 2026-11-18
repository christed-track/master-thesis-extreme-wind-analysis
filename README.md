# Master Thesis – Extreme Wind Analysis

This repository contains the Python code used for the climate-data-based scenario analysis in my Master's thesis at BOKU University.

The analysis investigates changes in 50-year return levels (RL50) of tropical-cyclone-related wind speeds at selected offshore wind farm sites in Taiwan and Guangdong, China.

## Analysis workflow

The Python script performs the following steps:

- calculation of distances between tropical cyclone track points and offshore wind farm sites using the Haversine formula
- selection of tropical cyclone track points within a 75 km radius around each site
- extraction of storm-specific maximum wind speeds
- calculation of annual maxima
- fitting of a Generalized Extreme Value (GEV) distribution
- estimation of 50-year return levels (RL50)
- comparison of the historical period (1950–2014) with the future period (2015–2050)
- calculation and visualization of ΔRL50

## Data

The analysis is based on TRACK-processed data from the HadGEM3-GC31-HH climate model.

The underlying NetCDF data files are not included in this repository.

The script expects the following files in a local `data` folder:

- historical TRACK dataset for 1950–2014
- future TRACK dataset for 2015–2050

## Software requirements

The script uses the following Python libraries:

- NumPy
- xarray
- matplotlib
- SciPy

## Author

Dennis Christen  
Master's thesis, BOKU University  
2026
