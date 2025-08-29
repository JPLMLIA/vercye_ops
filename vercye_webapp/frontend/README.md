
# Vercye Dashboard — React + TypeScript (Vite)

This ports the original static dashboards into a modular React app while **preserving the exact styles** and encapsulating API calls.

## Quick start
```bash
npm i
npm run dev
```

The app expects a backend exposing the same endpoints used in the original pages:

- `GET /api/studies` → string[]
- `POST /api/studies` body: `{ study_id }`
- `GET /api/studies/:id/status`
- `GET /api/studies/:id/run-config`
- `GET /api/studies/:id/run-config-status`
- `POST /api/studies/:id/setup` (multipart form-data)
- `POST /api/studies/:id/run-config` (multipart form-data)
- `PUT /api/studies/:id/run-config` (multipart form-data)
- `POST /api/studies/:id/run`
- `POST /api/studies/:id/cancel`
- `GET /api/studies/:id/results` → URL used in iframe
- `GET /api/studies/:id/logs` → string

- `GET /api/lai` → LAI entries

## Notes
- Styles are carried over 1:1 from the provided HTML so your visual design stays intact.
- UI is split into small reusable components (Header, Modal, StatusBadge, Stepper, FileUpload, Toast).
- Network logic lives under `src/api/*` with a tiny `http` client for consistency and error handling.
- LAI page includes a map view using Leaflet with OpenStreetMap tiles.
- Routing is done with `react-router-dom`.
