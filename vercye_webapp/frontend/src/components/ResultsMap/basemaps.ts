import L from 'leaflet';
import 'maplibre-gl/dist/maplibre-gl.css';
import '@maplibre/maplibre-gl-leaflet';

/**
 * Basemaps.
 *
 * The default is OpenFreeMap's Positron style - the same backdrop stacnotator uses
 * (`shared/imagery/tileUrls.ts`). It is the flat, near-white, "papery" cartography we
 * want, served without an API key, which CARTO's own raster tiles no longer are: CARTO
 * answers a keyless `curl` happily but gates real browser traffic, so it looked fine
 * from the server and failed in the page.
 *
 * Positron is a MapLibre vector style, so it is drawn through maplibre-gl-leaflet rather
 * than as raster tiles. The raster alternatives are Esri's, which need no key either and
 * cover the cases vector styles do not (imagery, relief).
 */
export type BasemapName = 'Light' | 'Terrain' | 'Satellite' | 'Sentinel-2 imagery';

const OPENFREEMAP_POSITRON = 'https://tiles.openfreemap.org/styles/positron';
const ESRI = 'https://services.arcgisonline.com/ArcGIS/rest/services';
const ESRI_ATTR = 'Tiles &copy; Esri';

/**
 * Basemaps sit below every overlay. This is explicit rather than left to DOM order:
 * Leaflet's layer control removes and re-appends the base layer when you switch, which
 * puts its tile container last in the pane - so without a pinned z-index a freshly
 * chosen basemap paints straight over the cropmask and the yield raster.
 */
export const BASEMAP_Z = 100;
/** Dedicated pane so basemaps sit below the data tiles whatever renderer they use. */
export const BASEMAP_PANE = 'vercye-basemap';

/**
 * Create the pane once per map. A z-index alone is not enough: the MapLibre basemap
 * draws into its own canvas container rather than the tile pane, so it ignored the tile
 * layers' z-index entirely and painted straight over the yield raster and the Sentinel-2
 * imagery - the polygons survived only because they live in the overlay pane above it.
 * Leaflet's default panes are tilePane 200 / overlayPane 400, so 100 puts basemaps under
 * both.
 */
export function ensureBasemapPane(map: L.Map) {
  if (!map.getPane(BASEMAP_PANE)) {
    const pane = map.createPane(BASEMAP_PANE);
    pane.style.zIndex = String(BASEMAP_Z);
  }
}

const tile = (path: string) =>
  L.tileLayer(`${ESRI}/${path}/MapServer/tile/{z}/{y}/{x}`, {
    maxZoom: 19,
    attribution: ESRI_ATTR,
    pane: BASEMAP_PANE,
  });

/**
 * Each entry bundles its base with its own labels: the Canvas and Imagery bases carry no
 * place names, and bundling keeps the layer control to three plain choices rather than a
 * base list plus a stray "Labels" checkbox the reader has to work out.
 */
export function createBasemaps(): Record<BasemapName, L.Layer> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const maplibre = (L as any).maplibreGL({
    style: OPENFREEMAP_POSITRON,
    pane: BASEMAP_PANE,
    attribution: '&copy; <a href="https://openfreemap.org">OpenFreeMap</a> &copy; OpenStreetMap contributors',
  }) as L.Layer;

  return {
    Light: maplibre,
    Terrain: L.layerGroup([tile('World_Topo_Map')]),
    Satellite: L.layerGroup([tile('World_Imagery'), tile('Reference/World_Boundaries_and_Places')]),
    // Placeholder: the Sentinel-2 mosaic changes with the selected month, so the page
    // builds and swaps that layer itself. Present here so it is one of the basemap choices.
    'Sentinel-2 imagery': L.layerGroup([]),
  };
}

/**
 * Data layers, bottom to top, all in the default tile pane and so above the basemap
 * pane. The yield raster is last because it is what the page exists to show: with the
 * cropmask on top instead, a national mask at 55% opacity washed the yield out entirely.
 */
export const LAYER_Z = {
  raster: 300, // the pixel layer (yield, APSIM yield or cropmask) - one at a time
} as const;
