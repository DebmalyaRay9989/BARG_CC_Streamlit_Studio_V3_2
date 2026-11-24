# Studio V3.2 browser checklist

After launching the application, verify the following on your Windows browser:

1. Open all seven workspaces and select each Appearance palette. Verify readable text, active sidebar highlight, widget focus, chart labels, and the light content canvas. At 100% and 125% zoom, verify no clipping; use the collapsed sidebar on narrow screens.
2. Workspace 02: choose bounded gating, change scores, remove one available view. Verify weights, Jacobian, exact drift curve and selected point. Removing all views must give a warning, not a certificate.
3. Workspace 05: load the bundled mixed-document demo. Run Quick clustering. Confirm the workflow strip updates, K charts appear, search/filter works, and inspect a second document. Verify SHA-256 and OCR status are rendered without the V2 exception.
4. Run the one-refit audit and verify the workflow status. Change palette and confirm the last completed clustering result and its run time are unchanged.
5. Hover, zoom and reset a Plotly chart; download the PNG from its toolbar. Verify CSV and JSON exports.
6. Run `python smoke_ui.py` in the installed environment. This exercises the Streamlit app backend, not visual layout.

Offline checks do not replace browser rendering; this package has not been visually verified in a live Streamlit browser in the build environment.

7. Confirm the cluster summary and keyword table (or insufficient-text caption) have no DeltaGenerator object, source expression or internal API documentation. In workspace 07 click Verify bundled evidence hashes and confirm a clean success notice.
8. If upgrading an existing folder, apply all files in the V3.2 hotfix (or use the full V3.2 package), stop the old server, and restart from the application directory.

9. Confirm Streamlit >=1.51,<2 (`python -m streamlit version`). Exercise the archive tables, charts, and architecture image and verify no legacy width deprecation messages appear in the terminal.
