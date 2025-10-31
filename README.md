# ACECQA Education and Care Services Analysis Toolkit

This repository contains helper code for the ACECQA exploratory data analysis assignment.
It automates the creation of visualisations and summary talking points that can be used in
slides and presentation scripts.

## Getting started

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the `Education-services-with-station-access_loc.csv` dataset from the course
   website and place it in the [`data/`](data/) directory. See [`data/README.md`](data/README.md)
   for the exact path expectations.
4. Generate all charts and summaries:
   ```bash
   python -m analysis.acecqa_analysis
   ```

## Outputs

Running the script produces the following artefacts inside the [`outputs/`](outputs/) folder:

- `quality_by_state.png` – stacked bar chart of quality ratings by state/territory.
- `quality_by_service_type.png` – top service categories ranked by quality mix.
- `rating_vs_train_distance.png` / `rating_vs_bus_distance.png` – regression plots testing
  the relationship between ratings and proximity to public transport.
- `capacity_vs_transport.png` – scatterplot comparing service capacity to transport access.
- `spatial_distribution.png` – choropleth-style point map of services across Australia.
- `operating_hours_by_rating.png` – optional box plot if operating-hour data is available.

A textual summary is printed to the console highlighting notable statistics that can be used
as key messages in the presentation.
