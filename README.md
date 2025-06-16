# SPECK: Spectrometric Toolkit

SPECK is an open‑source collection of utilities and datasets for the visualisation and analysis of
optical spectra.  It targets educational settings where simple yet powerful tools are needed to
handle Horiba `.LAB` files or standard CSV data.

The repository contains both **Python** applications and a **JavaScript** web viewer.  Example
spectra from discharge lamps, lasers and fluorescence samples are included.

## Repository Layout

```
labparser.py          # CLI converter: .lab → CSV (+ PDF preview)
spectra_lab_viewer.py # PyQt5 GUI to inspect and fit spectra
index.html            # Browser‑based viewer powered by Chart.js
compare.html          # Web tool for multi‑spectrum comparison
gallery.html          # HTML gallery of CCD screenshots
about.html            # Project information and credits
csv_out/              # Sample spectra in CSV format with a PDF summary
speckdata/            # Original .lab files
images/               # Logos and UI graphics
```

### Python tools

- **spectra_lab_viewer.py** – A full desktop application (PyQt5 + Matplotlib) to load one or
  multiple spectra, overlay them, adjust axes scales (linear, log, √) and fit peaks within a
  selected ROI.  Spectra can be exported as plots or simulated CCD images.
- **labparser.py** – Batch converter for Horiba `.LAB` files.  It generates CSV files and collects
  all spectra in a single PDF document.  Titles are read from `list.json` if present next to the
  input files.

Both scripts only depend on widely available packages (`PyQt5`, `matplotlib`, `numpy`, `scipy`,
`pillow`).  They can be compiled with Python ≥ 3.9.

### Web interface

`index.html` and `compare.html` implement a lightweight viewer entirely in JavaScript.  They read the
CSV files under `csv_out/` (or directly `.lab` files via a built‑in parser) and render interactive
plots using Chart.js.  A simulated CCD bar displays the colour distribution along the wavelength
axis.  The web tools run fully offline and are suitable for classroom use on any modern browser.

The `about.html` page summarises the project goals and lists third‑party dependencies—all released
under the MIT licence.  Logos and screenshots are located in `images/`.

### Data

Example spectra reside in `speckdata/` as original `.lab` files accompanied by `list.json`
containing human‑readable titles.  The `csv_out/` directory holds the converted CSV tables and a
precompiled PDF (`spettri.pdf`).  The accompanying `titoli.json` maps file names to titles and is
used by the HTML viewer.

## Installation

Install the required Python packages with:

```bash
pip install pyqt5 matplotlib numpy scipy pillow
```

No additional setup is required for the web viewer—simply open `index.html` in a browser.

## Usage

Run the desktop GUI:

```bash
python spectra_lab_viewer.py
```

Convert `.lab` files to CSV and PDF:

```bash
python labparser.py /path/to/lab_folder -o csv_out
```

## License

All original code in this repository is released under the MIT License.  See `about.html` for
credits of the JavaScript libraries used in the web interface.
