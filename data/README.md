# Data Directory

The dataset required for the ACECQA education and care services analysis is not bundled with the repository because of its large size.

1. Download the **Education-services-with-station-access_loc.csv** file from your course site.
2. Save or unzip the CSV into this `data/` directory so that the relative path becomes `data/Education-services-with-station-access_loc.csv`.
3. The analysis scripts will automatically detect the file in this location.

If you use a different filename, update the `DATA_FILENAME` constant inside `analysis/acecqa_analysis.py` accordingly.
