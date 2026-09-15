import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { METRICS, ResultsAPI, withDerivedMetrics } from '@/api/results';
import { CropmasksAPI } from '@/api/cropmasks';
import type {
  EvaluationResponse,
  GeometryCollection,
  LaiSeries,
  MultiyearResponse,
  RegionStats,
  ResultsManifest,
  StatsResponse,
} from '@/api/results';
import MapToolbar from '@/components/ResultsMap/MapToolbar';
import RegionPanel from '@/components/ResultsMap/RegionPanel';
import Legend from '@/components/ResultsMap/Legend';
import { colorFor, formatNumber, isDiverging, sequentialDomain } from '@/components/ResultsMap/scale';
import { BASEMAP_PANE, LAYER_Z, createBasemaps, ensureBasemapPane } from '@/components/ResultsMap/basemaps';
import TimeSlider from '@/components/ResultsMap/TimeSlider';
import StatsPanel from '@/components/ResultsMap/StatsPanel';
import { MOSAIC_MIN_ZOOM, monthsBetween, mosaicTileUrl, registerMonth } from '@/api/planetary';

/** Wait for the map to settle before refetching geometry, so a drag is one request. */
const VIEWPORT_DEBOUNCE_MS = 250;

type RasterRange = { min: number; max: number } | null;

/**
 * Fullscreen interactive results map.
 *
 * Replaces opening a prebuilt per-(year, timepoint) HTML. The three things that make it
 * scale to a study with thousands of regions:
 *   - polygons are fetched for the current viewport only, simplified for the zoom;
 *   - the raster is served as XYZ tiles off the COG's overviews, never downloaded whole;
 *   - switching year refetches only the small stats JSON and re-points the tile layer.
 *
 * All view state lives in the URL, so a particular year/level/region is linkable and the
 * browser's back button behaves.
 */
export default function ResultsMapPage() {
  const { studyId = '', runId, year: routeYear, timepoint: routeTimepoint } = useParams();
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();

  const scope = useMemo(() => ({ studyId, runId }), [studyId, runId]);

  const [manifest, setManifest] = useState<ResultsManifest | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [rasterRange, setRasterRange] = useState<RasterRange>(null);
  const [lai, setLai] = useState<LaiSeries | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null);
  const [multiyear, setMultiyear] = useState<MultiyearResponse | null>(null);
  const [search, setSearch] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const level = params.get('level') ?? '';
  /** One pixel layer at a time: '' | 'yield' | 'apsim_yield' | 'cropmask'. They occupy the
   *  same place on screen, so presenting them as independent toggles only produced
   *  stacks where one silently hid another. */
  const pixel = params.get('pixel') ?? '';
  const metric = params.get('metric') ?? METRICS[0].key;
  const basemap = params.get('basemap') ?? 'Light';
  const selected = params.get('region');
  const monthIdx = Number(params.get('month') ?? '0');
  const compare = useMemo(() => (params.get('compare') ?? '').split(',').filter(Boolean), [params]);
  const showStats = params.get('stats') !== '0';
  /** Summary panel view: 'year' | 'multiyear'. */
  const statsView = params.get('summary') ?? 'year';
  const showImagery = basemap === 'Sentinel-2 imagery';

  // Opacity is local: writing it to the URL on every input event re-rendered the page
  // mid-drag, which tore the slider out from under the pointer.
  const [opacity, setOpacity] = useState(1);

  const year = routeYear ?? '';
  const timepoint = routeTimepoint ?? '';

  const mapEl = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const geoLayer = useRef<L.GeoJSON | null>(null);
  const tileLayer = useRef<L.TileLayer | null>(null);
  const s2Layer = useRef<L.TileLayer | null>(null);
  const basemaps = useRef<Record<string, L.Layer> | null>(null);
  const fittedFor = useRef<string>('');
  // Latest values, readable from Leaflet callbacks without re-binding handlers.
  const regions: Record<string, RegionStats> = useMemo(
    () => (stats ? withDerivedMetrics(stats.regions) : {}),
    [stats],
  );
  const view = useRef({ level, year, timepoint, scope, regions, metric, pixel, compare, selected });
  view.current = { level, year, timepoint, scope, regions, metric, pixel, compare, selected };

  const patch = useCallback(
    (next: Record<string, string | undefined>) => {
      const p = new URLSearchParams(params);
      Object.entries(next).forEach(([k, v]) => {
        if (v === undefined || v === '') p.delete(k);
        else p.set(k, v);
      });
      setParams(p, { replace: true });
    },
    [params, setParams],
  );

  /* ---------------------------------------------------------------- manifest */
  useEffect(() => {
    let alive = true;
    ResultsAPI.manifest(scope)
      .then((m) => {
        if (!alive) return;
        setManifest(m);
        const y = routeYear && m.years[routeYear] ? routeYear : m.default.year;
        const tps = m.years[y] ?? [];
        const tp = routeTimepoint && tps.includes(routeTimepoint) ? routeTimepoint : m.default.timepoint;

        // One navigation, not two. setSearchParams only replaces the *search* and keeps
        // whatever pathname was current at render time, so calling it alongside navigate()
        // silently dropped the year/timepoint from the path and left the page with no
        // year to fetch - a permanently blank map.
        const next = new URLSearchParams(params);
        if (!next.get('level')) next.set('level', m.default.level);
        // The pixel-level yield is the thing this page exists to show, so it starts on
        // when the study has one. `raster=` (empty) is a deliberate "off" and is kept.
        if (next.get('raster') === null && m.rasters.length > 0) next.set('raster', m.rasters[0].kind);
        if (y !== routeYear || tp !== routeTimepoint || next.toString() !== params.toString()) {
          const bp = runId ? `/studies/${studyId}/runs/${runId}/results` : `/studies/${studyId}/results`;
          navigate(`${bp}/${y}/${tp}?${next}`, { replace: true });
        }
      })
      .catch((e) => alive && setError(e.message ?? String(e)));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [studyId, runId]);

  /* ------------------------------------------------------------------- map */
  useEffect(() => {
    if (!mapEl.current || mapRef.current) return;
    // Zoom control on the right: the summary panel is docked to the left edge and the
    // default top-left control sat on top of its heading.
    const map = L.map(mapEl.current, { zoomControl: false, preferCanvas: true, fadeAnimation: true }).setView(
      [0, 20],
      3,
    );

    ensureBasemapPane(map);
    basemaps.current = createBasemaps();
    L.control.zoom({ position: 'topright' }).addTo(map);
    L.control.scale({ imperial: false, position: 'bottomright' }).addTo(map);

    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  /* --------------------------------------------------------------- basemap */
  useEffect(() => {
    const map = mapRef.current;
    const all = basemaps.current;
    if (!map || !all) return;
    Object.entries(all).forEach(([name, layer]) => {
      const wanted = name === basemap && name !== 'Sentinel-2 imagery';
      if (wanted && !map.hasLayer(layer)) layer.addTo(map);
      if (!wanted && map.hasLayer(layer)) map.removeLayer(layer);
    });
    // Sentinel-2 has no static basemap of its own; keep Light beneath it so gaps between
    // scenes and the area outside the mosaic are not simply blank.
    if (basemap === 'Sentinel-2 imagery' && all.Light && !map.hasLayer(all.Light)) all.Light.addTo(map);
  }, [basemap]);

  /* ------------------------------------------------- frame the study on load */
  // Do this from the manifest's extent rather than from returned geometry: geometry is
  // viewport-filtered, so a default world view that misses the study returns no features
  // and there would be nothing to fit to.
  useEffect(() => {
    const map = mapRef.current;
    const b = manifest?.bounds;
    if (!map || !b || fittedFor.current) return;
    map.fitBounds(
      [
        [b[1], b[0]],
        [b[3], b[2]],
      ],
      // No animation: this is the first framing, not a user-initiated move, and
      // watching it fly in from the whole world reads as the page still loading.
      { padding: [30, 30], animate: false },
    );
    fittedFor.current = `${studyId}|${runId ?? ''}`;
  }, [manifest, studyId, runId]);

  /* --------------------------------------------------------------- stats */
  useEffect(() => {
    if (!year || !timepoint || !level) return;
    let alive = true;
    setLoading(true);
    ResultsAPI.stats(scope, year, timepoint, level)
      .then((s) => alive && setStats(s))
      .catch((e) => alive && setError(e.message ?? String(e)))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [scope, year, timepoint, level]);

  /* --------------------------------------------------- pixel layer + tiles */
  const isRasterPixel = pixel === 'yield' || pixel === 'apsim_yield';

  useEffect(() => {
    if (!isRasterPixel || !year || !timepoint) {
      setRasterRange(null);
      return;
    }
    let alive = true;
    const b = runId
      ? `/studies/${studyId}/runs/${runId}/results/${year}/${timepoint}/rasters/${pixel}/stats`
      : `/studies/${studyId}/results/${year}/${timepoint}/rasters/${pixel}/stats`;
    fetch(`/api${b}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('raster stats unavailable'))))
      .then((s) => alive && setRasterRange({ min: s.p2, max: s.p98 }))
      .catch(() => alive && setRasterRange(null));
    return () => {
      alive = false;
    };
  }, [studyId, runId, pixel, year, timepoint, isRasterPixel]);

  const cropmaskName = manifest?.cropmasks?.[year];

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (tileLayer.current) {
      map.removeLayer(tileLayer.current);
      tileLayer.current = null;
    }
    if (!year || !timepoint) return;

    let layer: L.TileLayer | null = null;
    if (isRasterPixel && rasterRange) {
      layer = L.tileLayer(ResultsAPI.tileUrl(scope, year, timepoint, pixel, [rasterRange.min, rasterRange.max]), {
        opacity,
        maxNativeZoom: 14,
        maxZoom: 19,
        zIndex: LAYER_Z.raster,
      });
    } else if (pixel === 'cropmask' && cropmaskName) {
      layer = L.tileLayer(CropmasksAPI.tileUrl(cropmaskName), {
        opacity,
        // Stop fetching new tiles past the mask's own resolution and upscale instead:
        // beyond this every zoom step re-requested a full set of tiles that carried no
        // more detail, so the layer blanked and repainted on the way in.
        maxNativeZoom: 15,
        maxZoom: 19,
        // Keep the previous zoom's tiles up until the new ones arrive, and hold a wider
        // ring around the viewport, so panning and zooming do not flash through gaps.
        updateWhenZooming: false,
        keepBuffer: 4,
        zIndex: LAYER_Z.raster,
      });
    }
    if (layer) {
      layer.addTo(map);
      tileLayer.current = layer;
    }
    // `opacity` deliberately excluded: it is applied by the effect below, so dragging the
    // slider adjusts the live layer instead of tearing it down and refetching every tile.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scope, pixel, year, timepoint, rasterRange, cropmaskName, isRasterPixel]);

  useEffect(() => {
    tileLayer.current?.setOpacity(opacity);
  }, [opacity]);

  // A new pixel layer starts fully opaque - it was just asked for, so it should be visible.
  useEffect(() => {
    setOpacity(1);
  }, [pixel]);

  /* ------------------------------------------------------------------ LAI */
  // Observed-LAI curves for every region at this level, used by both the summary panel
  // (level mean, or the selected regions overlaid) and the per-region detail pane.
  useEffect(() => {
    if (!year || !timepoint || !level) return;
    let alive = true;
    setLai(null);
    ResultsAPI.lai(scope, year, timepoint, level)
      .then((d) => alive && setLai(d))
      // Not every level has a timeseries - the per-region "primary" level has none - so a
      // miss is normal and simply hides the charts rather than surfacing an error.
      .catch(() => alive && setLai(null));
    return () => {
      alive = false;
    };
  }, [scope, year, timepoint, level]);

  /* ------------------------------------------------- pipeline evaluation */
  // Accuracy metrics come from the pipeline's evaluation step rather than being derived
  // in the browser, so this panel and the PDF reports can never disagree.
  useEffect(() => {
    if (!year || !timepoint || !level) return;
    let alive = true;
    setEvaluation(null);
    ResultsAPI.evaluation(scope, year, timepoint, level)
      .then((d) => alive && setEvaluation(d))
      .catch(() => alive && setEvaluation(null));
    return () => {
      alive = false;
    };
  }, [scope, year, timepoint, level]);

  // Every year at once, fetched only when the multiyear view is actually open.
  useEffect(() => {
    if (statsView !== 'multiyear' || !timepoint || !level) return;
    let alive = true;
    setMultiyear(null);
    ResultsAPI.multiyear(scope, timepoint, level)
      .then((d) => alive && setMultiyear(d))
      .catch(() => alive && setMultiyear(null));
    return () => {
      alive = false;
    };
  }, [scope, timepoint, level, statsView]);

  /* --------------------------------------------------- Sentinel-2 imagery */
  // Months come from the study's own LAI window, so you can only scrub the period the
  // yield was derived from.
  const months = useMemo(() => {
    const w = manifest?.seasons?.[year]?.[timepoint];
    return w ? monthsBetween(w[0], w[1]) : [];
  }, [manifest, year, timepoint]);

  const [s2State, setS2State] = useState<'loading' | 'error' | null>(null);
  const [zoom, setZoom] = useState<number>(0);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (!showImagery || months.length === 0) {
      if (s2Layer.current) {
        map.removeLayer(s2Layer.current);
        s2Layer.current = null;
      }
      setS2State(null);
      return;
    }
    const month = months[Math.min(monthIdx, months.length - 1)];
    if (!month) return;

    let cancelled = false;
    setS2State('loading');
    registerMonth(month, manifest?.bounds ?? undefined)
      .then((searchid) => {
        if (cancelled || !mapRef.current) return;
        if (s2Layer.current) mapRef.current.removeLayer(s2Layer.current);
        const layer = L.tileLayer(mosaicTileUrl(searchid), {
          maxZoom: 19,
          // The mosaic answers 204 below this, so asking for those tiles paints nothing.
          minZoom: MOSAIC_MIN_ZOOM,
          pane: BASEMAP_PANE, // it is a basemap choice, so it sits with the basemaps
          attribution: 'Sentinel-2 L2A &copy; ESA, via Microsoft Planetary Computer',
        });
        layer.addTo(mapRef.current);
        s2Layer.current = layer;
        setS2State(null);
      })
      .catch(() => !cancelled && setS2State('error'));
    return () => {
      cancelled = true;
    };
  }, [showImagery, months, monthIdx, manifest]);

  /* ------------------------------------------------------------- geometry */
  /** Style for one feature, from the current metric and pixel-layer state. */
  const styleFor = useCallback(
    (id: string, domain: { min: number; max: number }) => {
      const cur = view.current;
      // A filled choropleth would sit on top of the pixel layer (polygons live in the
      // overlay pane, above every tile) and hide it, so boundaries only in that case.
      const outlineOnly = !!cur.pixel;
      const picked = cur.compare.includes(id) || cur.selected === id;
      return {
        color: picked ? 'rgb(50 98 71)' : outlineOnly ? 'rgb(37 36 35)' : 'rgb(118 115 111)',
        weight: picked ? 2.6 : outlineOnly ? 1.1 : 0.7,
        fillColor: colorFor(cur.regions[id]?.[cur.metric], domain.min, domain.max, cur.metric),
        fillOpacity: outlineOnly ? 0 : 0.7,
      };
    },
    [],
  );

  /**
   * Colour domain for the choropleth.
   *
   * Preference order: a range the user pinned, then the study-wide range the manifest
   * carries (computed once across every level and every year, so a colour keeps its
   * meaning as you move around), then - only if the study is too old to have one - the
   * current selection's own values.
   */
  const pinnedDomain = useMemo(() => {
    const raw = params.get('range');
    if (!raw) return null;
    const [a, b] = raw.split(',').map(Number);
    return Number.isFinite(a) && Number.isFinite(b) && b > a ? { min: a, max: b } : null;
  }, [params]);

  const domain = useMemo(() => {
    if (pinnedDomain) return pinnedDomain;
    const shared = manifest?.scales?.[metric];
    if (shared && Number.isFinite(shared.min) && Number.isFinite(shared.max) && shared.max > shared.min) {
      return shared;
    }
    const vals = Object.values(regions)
      .map((r) => r[metric])
      .filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
    if (!vals.length) return { min: 0, max: 1 };
    // Diverging metrics must keep their true extremes so the symmetric scaling about zero
    // is honest; sequential ones cap at p95 so one outlier cannot flatten the rest.
    return isDiverging(metric)
      ? { min: Math.min(...vals), max: Math.max(...vals) }
      : sequentialDomain(vals);
  }, [pinnedDomain, manifest, regions, metric]);

  // Readable from the Leaflet callbacks, which are bound once.
  const domainRef = useRef(domain);
  domainRef.current = domain;

  const drawGeometry = useCallback(
    (fc: GeometryCollection) => {
      const map = mapRef.current;
      if (!map) return;
      const cur = view.current;
      const dom = domainRef.current;

      if (geoLayer.current) map.removeLayer(geoLayer.current);

      const meta = METRICS.find((m) => m.key === cur.metric) ?? METRICS[0];
      const layer = L.geoJSON(fc as unknown as GeoJSON.GeoJsonObject, {
        style: (feat) => styleFor((feat?.properties as { region?: string })?.region ?? '', dom),
        onEachFeature: (feat, lyr) => {
          const id = (feat.properties as { region?: string }).region ?? '';
          const v = cur.regions[id]?.[cur.metric];
          lyr.bindTooltip(
            `${id}<br/>${meta.label}: ${formatNumber(v, meta.digits)}${meta.unit ? ' ' + meta.unit : ''}`,
            { sticky: true },
          );
          // Hover: lift the outline so the shape under the pointer is unambiguous. The
          // fill is left alone - with the pixel layer on there is no fill to change, and
          // dimming it would fight the layer the user is actually reading.
          lyr.on('mouseover', () => {
            const path = lyr as unknown as L.Path;
            if (view.current.compare.includes(id) || view.current.selected === id) return;
            path.setStyle({ color: 'rgb(50 98 71)', weight: 2.4 });
            path.bringToFront();
          });
          lyr.on('mouseout', () => {
            (lyr as unknown as L.Path).setStyle(styleFor(id, domainRef.current));
          });
          lyr.on('click', (e) => {
            // Shift-click builds the comparison set; a plain click opens one region.
            const orig = (e as unknown as { originalEvent?: MouseEvent }).originalEvent;
            if (orig?.shiftKey) {
              const now = view.current.compare;
              const next = now.includes(id) ? now.filter((x) => x !== id) : [...now, id];
              patch({ compare: next.join(','), stats: '1' });
            } else {
              patch({ region: id });
            }
          });
        },
      });
      layer.addTo(map);
      geoLayer.current = layer;
    },
    [patch, styleFor],
  );

  // Restyle in place when the metric, the pixel layer or the selection changes. Without
  // this the polygons kept whatever style they were built with - which is why turning the
  // pixel layer back off sometimes left the vectors invisible (fillOpacity stuck at 0).
  useEffect(() => {
    geoLayer.current?.setStyle((feat) => styleFor((feat?.properties as { region?: string })?.region ?? '', domain));
  }, [metric, pixel, selected, compare, domain, regions, styleFor]);

  const refreshGeometry = useCallback(() => {
    const map = mapRef.current;
    const { level: lv, year: y, timepoint: tp } = view.current;
    if (!map || !lv || !y || !tp) return;
    const b = map.getBounds().pad(0.2);
    setLoading(true);
    ResultsAPI.geometry(view.current.scope, y, tp, lv, map.getZoom(), [
      b.getWest(),
      b.getSouth(),
      b.getEast(),
      b.getNorth(),
    ])
      .then(drawGeometry)
      .catch((e) => setError(e.message ?? String(e)))
      .finally(() => setLoading(false));
  }, [drawGeometry]);

  // Refetch when the view settles. The handler is bound once and reads the latest
  // refreshGeometry through a ref: binding it on every change of `refreshGeometry` (which
  // changes whenever the URL does, via `patch`) tore down the pending debounce timer, so
  // the refetch at the *fitted* zoom was routinely cancelled and the map was left showing
  // whatever coarse geometry the initial world view had asked for.
  const refreshRef = useRef(refreshGeometry);
  refreshRef.current = refreshGeometry;
  const idleTimer = useRef<number | undefined>(undefined);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const onIdle = () => {
      window.clearTimeout(idleTimer.current);
      idleTimer.current = window.setTimeout(() => refreshRef.current(), VIEWPORT_DEBOUNCE_MS);
    };
    const onZoom = () => setZoom(map.getZoom());
    onZoom();
    map.on('moveend zoomend', onIdle);
    map.on('zoomend', onZoom);
    return () => {
      window.clearTimeout(idleTimer.current);
      map.off('moveend zoomend', onIdle);
      map.off('zoomend', onZoom);
    };
    // Bound once for the life of the map.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (stats) refreshGeometry();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stats, level]);

  /* ---------------------------------------------------------------- render */
  const onToolbarChange = (p: Record<string, string | undefined>) => {
    if (p.year !== undefined || p.timepoint !== undefined) {
      const y = p.year ?? year;
      const tp = p.timepoint ?? timepoint;
      const bp = runId ? `/studies/${studyId}/runs/${runId}/results` : `/studies/${studyId}/results`;
      navigate(`${bp}/${y}/${tp}?${params}`);
      return;
    }
    patch(p);
  };

  const meta = METRICS.find((m) => m.key === metric) ?? METRICS[0];

  // The same list at every level, with what this level has no values for disabled and
  // labelled with the level that does have it.
  const metricChoices = useMemo(() => {
    const byLevel = manifest?.metrics_by_level ?? {};
    const here = new Set(byLevel[level] ?? []);
    return METRICS.map((m) => ({
      key: m.key,
      label: m.label,
      available: here.size ? here.has(m.key) : Object.values(regions).some((r) => typeof r[m.key] === 'number'),
      whereAvailable: Object.entries(byLevel)
        .filter(([lv, keys]) => lv !== level && keys.includes(m.key))
        .map(([lv]) => lv),
    }));
  }, [manifest, level, regions]);
  const showPlot = !!manifest && level === manifest.plots_level && !!selected;
  const hasVectorValues = Object.values(regions).some((r) => typeof r[metric] === 'number');

  return (
    <div className="results-map-page motion-fade-in">
      {manifest && (
        <MapToolbar
          studyId={studyId}
          runId={runId}
          years={manifest.years}
          year={year}
          timepoint={timepoint}
          levels={manifest.levels}
          level={level || manifest.default.level}
          metric={metric}
          metrics={metricChoices}
          basemap={basemap}
          hasImagery={months.length > 0}
          pixel={pixel}
          pixelOptions={[
            ...manifest.rasters.map((r) => ({ value: r.kind, label: r.label })),
            ...(cropmaskName ? [{ value: 'cropmask', label: 'Cropmask' }] : []),
          ]}
          opacity={opacity}
          onOpacity={setOpacity}
          search={search}
          onSearch={setSearch}
          onSearchSubmit={() => {
            const hit = Object.keys(regions).find((r) => r.toLowerCase().includes(search.trim().toLowerCase()));
            if (hit) patch({ region: hit });
          }}
          statsOpen={showStats}
          onChange={onToolbarChange}
        />
      )}

      {error && <div className="map-error">{error}</div>}

      <div className="results-map-body">
        <div ref={mapEl} className="results-map-canvas" />

        {/* Floated over the map rather than occupying a slot in the toolbar: as a toolbar
            child it appeared and disappeared on every fetch, which re-flowed the whole
            bar and made the controls jump sideways. Here it overlays the canvas and
            costs no layout. */}
        {loading && (
          <div className="map-busy" role="status" aria-live="polite">
            Loading…
          </div>
        )}

        <div className="map-legend-stack">
          {hasVectorValues && (
            <Legend
              title={`${level} · ${meta.label}${meta.unit ? ` (${meta.unit})` : ''}`}
              min={domain.min}
              max={domain.max}
              digits={meta.digits}
              metric={metric}
              pinned={!!pinnedDomain}
              onRange={(r) => patch({ range: r ? `${r.min},${r.max}` : undefined })}
            />
          )}
          {pixel && (
            <Legend
              title={pixel === 'cropmask' ? 'Cropmask' : 'Pixel yield (kg/ha)'}
              min={pixel === 'cropmask' ? 0 : (rasterRange?.min ?? 0)}
              max={pixel === 'cropmask' ? 1 : (rasterRange?.max ?? 0)}
              binary={pixel === 'cropmask'}
            />
          )}
        </div>

        {showImagery && months.length > 0 && (
          <div className="time-slider-dock">
            <TimeSlider
              months={months}
              index={Math.min(monthIdx, months.length - 1)}
              onSelect={(i) => patch({ month: String(i) })}
              status={zoom > 0 && zoom < MOSAIC_MIN_ZOOM ? 'zoom' : s2State}
            />
          </div>
        )}

        {showStats && stats && (
          <StatsPanel
            level={level}
            metric={metric}
            regions={regions}
            lai={lai}
            evaluation={evaluation}
            multiyear={multiyear}
            view={statsView}
            onView={(v) => patch({ summary: v === 'year' ? undefined : v })}
            selected={compare}
            onClearSelection={() => patch({ compare: undefined })}
            onClose={() => patch({ stats: '0' })}
          />
        )}

        {selected && (
          <RegionPanel
            region={selected}
            stats={regions[selected]}
            lai={lai}
            plotUrl={showPlot ? ResultsAPI.regionPlotUrl(scope, year, timepoint, selected) : undefined}
            plotsLevel={manifest?.plots_level}
            referenceLevels={Object.entries(manifest?.metrics_by_level ?? {})
              .filter(([lv, keys]) => lv !== level && keys.includes('reported_mean_yield_kg_ha'))
              .map(([lv]) => lv)}
            onGoToLevel={(lv) => patch({ level: lv, region: undefined })}
            onGoToPlotsLevel={
              manifest && level !== manifest.plots_level
                ? () => patch({ level: manifest.plots_level, region: undefined })
                : undefined
            }
            inComparison={compare.includes(selected)}
            onToggleCompare={() => {
              const next = compare.includes(selected)
                ? compare.filter((x) => x !== selected)
                : [...compare, selected];
              patch({ compare: next.join(','), stats: '1' });
            }}
            onClose={() => patch({ region: undefined })}
          />
        )}
      </div>
    </div>
  );
}
