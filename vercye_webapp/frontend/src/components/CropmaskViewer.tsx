import { useEffect, useMemo, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { CropmasksAPI } from '@/api/cropmasks';
import type { CropmaskInfo } from '@/api/cropmasks';
import { MOSAIC_MIN_ZOOM, monthsBetween, mosaicTileUrl, registerMonth } from '@/api/planetary';
import type { Month } from '@/api/planetary';
import { CloseIcon } from '@/components/Icons';
import TimeSlider from '@/components/ResultsMap/TimeSlider';
import { BASEMAP_PANE, LAYER_Z, createBasemaps, ensureBasemapPane } from '@/components/ResultsMap/basemaps';

type Props = { name: string; onClose: () => void };

/**
 * Fullscreen cropmask inspector.
 *
 * A mask uploaded with a season gets a month slider over Sentinel-2 first-valid
 * composites for that season, so you can check the mask against what the imagery
 * actually shows through the growing season. A mask without season metadata - every mask
 * uploaded before that field existed - falls back to the two basemaps, which is the whole
 * reason the metadata is optional rather than required.
 */
export default function CropmaskViewer({ name, onClose }: Props) {
  const mapEl = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const maskLayer = useRef<L.TileLayer | null>(null);
  const s2Layer = useRef<L.TileLayer | null>(null);

  const [info, setInfo] = useState<CropmaskInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [monthIdx, setMonthIdx] = useState(0);
  const [s2State, setS2State] = useState<'off' | 'loading' | 'on' | 'error'>('off');
  const [showMask, setShowMask] = useState(true);
  const [preparing, setPreparing] = useState(false);
  // The cropmask is the subject of this view, so it starts fully opaque; the slider is
  // there for checking what is underneath, not for making it legible in the first place.
  const [maskOpacity, setMaskOpacity] = useState(1);

  const months: Month[] = useMemo(
    () => (info?.season_start && info?.season_end ? monthsBetween(info.season_start, info.season_end) : []),
    [info],
  );
  const hasSeason = months.length > 0;

  useEffect(() => {
    let alive = true;
    let timer: number | undefined;

    /* Masks are uploaded at a single resolution, so a zoomed-out tile would otherwise
       scan most of the raster (measured ~11 s on a national mask). The server builds
       overviews in the background; poll until they land, then let the layer refresh. */
    const poll = (i: CropmaskInfo) => {
      if (!alive) return;
      setInfo(i);
      if (i.has_overviews) {
        setPreparing(false);
        return;
      }
      setPreparing(true);
      CropmasksAPI.prepare(name).catch(() => undefined);
      timer = window.setTimeout(() => {
        CropmasksAPI.info(name).then(poll).catch(() => undefined);
      }, 3000);
    };

    CropmasksAPI.info(name)
      .then(poll)
      .catch((e) => alive && setError(e.message ?? String(e)));

    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [name]);

  // Once overviews exist, force the tile layer to refetch the placeholders it cached.
  useEffect(() => {
    if (info?.has_overviews) maskLayer.current?.redraw();
  }, [info?.has_overviews]);

  /* --------------------------------------------------------------- map init */
  useEffect(() => {
    if (!mapEl.current || mapRef.current) return;
    const map = L.map(mapEl.current, { zoomControl: false, preferCanvas: true }).setView([0, 20], 3);
    ensureBasemapPane(map);
    const base = createBasemaps();
    base.Light.addTo(map);
    L.control.layers(base, {}, { position: 'topright', collapsed: false }).addTo(map);
    L.control.zoom({ position: 'topright' }).addTo(map);
    L.control.scale({ imperial: false, position: 'bottomright' }).addTo(map);
    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  /* ------------------------------------------------------- mask layer + fit */
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !info) return;
    if (maskLayer.current) map.removeLayer(maskLayer.current);
    const layer = L.tileLayer(CropmasksAPI.tileUrl(name), { opacity: maskOpacity, zIndex: LAYER_Z.raster, maxZoom: 19 });
    if (showMask) layer.addTo(map);
    maskLayer.current = layer;
    const [w, s, e, n] = info.bounds;
    map.fitBounds(
      [
        [s, w],
        [n, e],
      ],
      { padding: [20, 20], animate: false },
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [info, name]);

  useEffect(() => {
    const map = mapRef.current;
    const layer = maskLayer.current;
    if (!map || !layer) return;
    if (showMask && !map.hasLayer(layer)) layer.addTo(map);
    if (!showMask && map.hasLayer(layer)) map.removeLayer(layer);
  }, [showMask]);

  useEffect(() => {
    maskLayer.current?.setOpacity(maskOpacity);
  }, [maskOpacity]);

  /* ------------------------------------------------------- S2 month composite */
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !hasSeason) return;
    const month = months[Math.min(monthIdx, months.length - 1)];
    if (!month) return;

    let cancelled = false;
    setS2State('loading');
    registerMonth(month, info?.bounds)
      .then((searchid) => {
        if (cancelled || !mapRef.current) return;
        if (s2Layer.current) mapRef.current.removeLayer(s2Layer.current);
        const layer = L.tileLayer(mosaicTileUrl(searchid), {
          maxZoom: 19,
          minZoom: MOSAIC_MIN_ZOOM,
          pane: BASEMAP_PANE,
          attribution: 'Sentinel-2 L2A &copy; ESA, via Microsoft Planetary Computer',
        });
        layer.addTo(mapRef.current);
        s2Layer.current = layer;
        setS2State('on');
      })
      .catch(() => !cancelled && setS2State('error'));

    return () => {
      cancelled = true;
    };
  }, [months, monthIdx, hasSeason, info]);

  const current = months[Math.min(monthIdx, Math.max(0, months.length - 1))];

  return (
    <div className="cropmask-viewer motion-fade-in">
      <div className="map-toolbar">
        <button className="btn btn-sm btn-secondary map-toolbar-back" onClick={onClose} title="Close" aria-label="Close">
          <CloseIcon size={15} />
        </button>
        <div className="map-toolbar-title">
          <strong>{name}</strong>
          {info?.year && <span className="map-toolbar-run">season {info.season_start} → {info.season_end}</span>}
        </div>

        <div className="map-toolbar-group">
          <label className="map-toolbar-label">
            <input type="checkbox" checked={showMask} onChange={(e) => setShowMask(e.target.checked)} /> Cropmask
          </label>
          <input
            type="range"
            min={0}
            max={100}
            value={Math.round(maskOpacity * 100)}
            onChange={(e) => setMaskOpacity(Number(e.target.value) / 100)}
            className="map-toolbar-opacity"
            title="Cropmask opacity"
            aria-label="Cropmask opacity"
          />
        </div>

        {preparing && (
          <div className="map-toolbar-status busy" title="Building a multi-resolution index so zoomed-out views are fast">
            preparing overviews…
          </div>
        )}

        {hasSeason ? (
          <div className="map-toolbar-status">Sentinel-2 · {current?.label}</div>
        ) : (
          <div className="map-toolbar-status" title="Add a year and season to this cropmask to browse Sentinel-2 imagery">
            no season set - basemaps only
          </div>
        )}
      </div>

      {error && <div className="map-error">{error}</div>}
      {!info && !error && <div className="map-error" style={{ background: 'var(--neutral-100)', color: 'var(--neutral-800)' }}>Loading…</div>}

      <div className="results-map-body">
        <div ref={mapEl} className="results-map-canvas" />
        {hasSeason && (
          <div className="time-slider-dock">
            <TimeSlider
              months={months}
              index={Math.min(monthIdx, months.length - 1)}
              onSelect={setMonthIdx}
              status={s2State === 'loading' ? 'loading' : s2State === 'error' ? 'error' : null}
            />
          </div>
        )}
      </div>
    </div>
  );
}
