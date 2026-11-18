# Security and prompt-injection audit

## Inputs reviewed
- **Official PDFs (5):** Read as task requirements. They contain no instructions aimed at an AI assistant, and no hidden or encoded content was observed in the extracted text.
- **Dataset files:** These are purely numeric, whitespace-separated columns, parsed with `numpy.loadtxt`. Nothing in them is executed or interpreted, and a column-count check is applied.
- **Dependencies:** Only well-known PyPI packages, pinned in `requirements.txt`.
- **Network access:** Nothing is downloaded at run time. The Folium map HTML references the Leaflet CDN and OSM tiles, which are loaded in the browser.

## API hardening
- **`trip_id` validation:** It must match `^[A-Za-z0-9_-]{1,64}$` and must be one of the folder names found in `PATHSNAP_EVAL_DIR`. Path traversal (`../train`) was tested and is rejected with 404 (see `results/logs/api_requests.log`).
- **No raw ground truth in responses:** The response contains only aggregate scores, and the map sent to the UI omits the reference route.
- **No secrets:** The project has no credentials, tokens or keys. `.gitignore` excludes the dataset, caches and the venv.
- **CORS:** Allows all origins (GET/POST only), so the UI can run on a second laptop. The API is meant for a closed LAN during evaluation and should not be exposed to the internet.

## Findings
- **No suspicious instructions found** in the supplied materials.
- **No document instruction changed the implementation** beyond the official task requirements.
