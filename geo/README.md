# geo/

- `juridiccion.geojson` – the 8 district sub-areas ("jurisdicciones") of San Juan de Lurigancho (committed).
- `camaras_municipales.json` – **not committed.** Real municipal camera positions. Generate it with
  `python -m app.camaras_remote` (see the main README). Without it the API serves no cameras and the
  "verify camera" hint falls back to "verify the zone with a patrol".
- `camaras_vecinales.json` – **not committed.** Neighbourhood (private) cameras, position/brand/mode only.

Camera locations are sensitive infrastructure information, which is why they are git-ignored.
