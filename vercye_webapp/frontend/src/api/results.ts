/**
 * Client for the results-map data endpoints.
 *
 * The map is assembled from small, independently cacheable pieces rather than one
 * prebuilt HTML, so switching year only refetches `stats` (tens of KB) and swaps the
 * tile URL. Geometry is keyed by year in the URL but carries a content-derived ETag,
 * so when a study's boundaries are unchanged between years the browser reuses its copy.
 */
import { http } from './client';

export type ResultsScope = { studyId: string; runId?: string };

export type RasterKind = { kind: string; label: string };

export type ResultsManifest = {
  study_id: string;
  run_id: string | null;
  /** year -> timepoints */
  years: Record<string, string[]>;
  levels: string[];
  primary_level: string;
  rasters: RasterKind[];
  /** Level whose region ids have per-region report plots. */
  plots_level: string;
  /** year -> cropmask name, from the run config. Drawn via the cropmask tile endpoint. */
  cropmasks: Record<string, string>;
  /** [west, south, east, north] of the study's regions, used to frame the map on load. */
  bounds: [number, number, number, number] | null;
  /** year -> timepoint -> [start, end]: the observed-LAI window the yield came from. */
  seasons: Record<string, Record<string, [string, string]>>;
  /**
   * Default colour domain per metric, computed once over every level and every year.
   * Shared like this a colour means the same value wherever you are, instead of the
   * legend quietly rescaling each time you switch year or admin level.
   */
  scales: Record<string, { min: number; max: number }>;
  /** level -> metric keys that level actually carries values for. */
  metrics_by_level: Record<string, string[]>;
  default: { year: string; timepoint: string; level: string };
};

export type RegionStats = Record<string, number | string | null>;

export type StatsResponse = {
  level: string;
  year: string;
  timepoint: string;
  source: string;
  range: { min: number; max: number } | null;
  regions: Record<string, RegionStats>;
};

export type GeometryCollection = {
  type: 'FeatureCollection';
  features: Array<{
    type: 'Feature';
    geometry: GeoJSON.Geometry;
    properties: { region: string };
  }>;
};

/** `/studies/{id}` or `/studies/{id}/runs/{runId}` - archived runs use the same shapes. */
function base({ studyId, runId }: ResultsScope): string {
  return runId
    ? `/studies/${encodeURIComponent(studyId)}/runs/${encodeURIComponent(runId)}/results`
    : `/studies/${encodeURIComponent(studyId)}/results`;
}

/** One metric as the pipeline's evaluation step reports it. */
export type EvalField = { key: string; label: string; digits: number };

export type EvaluationResponse = {
  level: string;
  year: string;
  timepoint: string;
  fields: EvalField[];
  /** null when the level has no reference data for that year. */
  metrics: Record<string, number> | null;
};

export type MultiyearResponse = {
  level: string;
  timepoint: string;
  years: string[];
  fields: EvalField[];
  /** One entry per year that has reference data, straight from evaluation_{level}.csv. */
  metrics: Array<Record<string, number | string>>;
  source: string | null;
  /** region -> per-year predicted and reported yields, from all_predictions_*.csv. */
  regions: Record<string, Array<{ year: string; predicted: number | null; reported: number | null }>>;
};

export type LaiSeries = {
  level: string;
  column: string;
  dates: string[];
  regions: Record<string, Array<number | null>>;
};

/** Values that can colour the choropleth. Mirrors the old prebuilt map's heatmap selector. */
export const METRICS: Array<{ key: string; label: string; unit: string; digits: number }> = [
  { key: 'mean_yield_kg_ha', label: 'Mean yield', unit: 'kg/ha', digits: 0 },
  { key: 'median_yield_kg_ha', label: 'Median yield', unit: 'kg/ha', digits: 0 },
  { key: 'mean_yield_kg_ha_apsim', label: 'Mean yield (APSIM only)', unit: 'kg/ha', digits: 0 },
  { key: 'reported_mean_yield_kg_ha', label: 'Reference yield', unit: 'kg/ha', digits: 0 },
  { key: 'abs_error', label: 'Absolute error', unit: 'kg/ha', digits: 0 },
  { key: 'rel_error', label: 'Relative error', unit: '%', digits: 1 },
  { key: 'total_production_ton', label: 'Production', unit: 't', digits: 0 },
  { key: 'total_area_ha', label: 'Cropland area', unit: 'ha', digits: 0 },
  { key: 'max_rs_lai', label: 'Peak observed LAI', unit: '', digits: 2 },
];

/** Error metrics are derived here rather than stored, so they follow the source values. */
export function withDerivedMetrics(regions: Record<string, RegionStats>): Record<string, RegionStats> {
  const out: Record<string, RegionStats> = {};
  for (const [id, r] of Object.entries(regions)) {
    const pred = r.mean_yield_kg_ha;
    const ref = r.reported_mean_yield_kg_ha;
    const both = typeof pred === 'number' && typeof ref === 'number' && Number.isFinite(pred) && Number.isFinite(ref);
    out[id] = {
      ...r,
      abs_error: both ? Math.abs((pred as number) - (ref as number)) : null,
      rel_error: both && (ref as number) !== 0 ? (((pred as number) - (ref as number)) / (ref as number)) * 100 : null,
    };
  }
  return out;
}

export const ResultsAPI = {
  manifest: (scope: ResultsScope) => http.get<ResultsManifest>(`${base(scope)}/manifest`),

  stats: (scope: ResultsScope, year: string, timepoint: string, level: string) =>
    http.get<StatsResponse>(
      `${base(scope)}/${year}/${encodeURIComponent(timepoint)}/levels/${encodeURIComponent(level)}/stats`,
    ),

  /**
   * Polygons for the current viewport, simplified for `zoom`. Sending the bbox is what
   * keeps a study with thousands of regions responsive - the server never ships more
   * than what is on screen.
   */
  geometry: (
    scope: ResultsScope,
    year: string,
    timepoint: string,
    level: string,
    zoom: number,
    bbox?: [number, number, number, number],
  ) => {
    const q = new URLSearchParams({ zoom: String(Math.round(zoom)) });
    if (bbox) q.set('bbox', bbox.map((v) => v.toFixed(4)).join(','));
    return http.get<GeometryCollection>(
      `${base(scope)}/${year}/${encodeURIComponent(timepoint)}/levels/${encodeURIComponent(level)}/geometry?${q}`,
    );
  },

  /** XYZ template for Leaflet. Tiles are immutable per run, so they cache hard. */
  tileUrl: (scope: ResultsScope, year: string, timepoint: string, kind: string, rescale: [number, number]) =>
    `/api${base(scope)}/${year}/${encodeURIComponent(timepoint)}/tiles/${encodeURIComponent(kind)}/{z}/{x}/{y}.png` +
    `?rescale=${rescale[0]},${rescale[1]}`,

  /** Observed-LAI curves for every region at this level. */
  lai: (scope: ResultsScope, year: string, timepoint: string, level: string) =>
    http.get<LaiSeries>(
      `${base(scope)}/${year}/${encodeURIComponent(timepoint)}/levels/${encodeURIComponent(level)}/lai`,
    ),

  /**
   * Accuracy metrics for one level/year, as computed by the pipeline's evaluation step.
   * Reported rather than recomputed so the panel and the PDF reports cannot disagree.
   */
  evaluation: (scope: ResultsScope, year: string, timepoint: string, level: string) =>
    http.get<EvaluationResponse>(
      `${base(scope)}/${year}/${encodeURIComponent(timepoint)}/levels/${encodeURIComponent(level)}/evaluation`,
    ),

  /** Every year of a level at once: per-year metrics plus per-region yield series. */
  multiyear: (scope: ResultsScope, timepoint: string, level: string) =>
    http.get<MultiyearResponse>(
      `${base(scope)}/multiyear/${encodeURIComponent(timepoint)}/levels/${encodeURIComponent(level)}`,
    ),

  /** Per-region report plot, fetched only when a region is actually opened. */
  regionPlotUrl: (scope: ResultsScope, year: string, timepoint: string, region: string) =>
    `/api${base(scope)}/${year}/${encodeURIComponent(timepoint)}/regions/${encodeURIComponent(region)}/plot.png`,
};
