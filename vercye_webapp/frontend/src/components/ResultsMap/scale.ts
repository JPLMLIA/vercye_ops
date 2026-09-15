/**
 * Colour ramp shared by the choropleth and the raster legend.
 *
 * This is deliberately the same formula the tile endpoint uses server-side
 * (`_colormap` in routers/maps.py). Keeping them in step is what lets a user compare
 * a region's fill against the pixels inside it without being misled.
 */
export const NO_DATA_FILL = 'rgb(218 217 216)'; // neutral-200, matching the theme

/** Cropland fill. Must match CROPLAND_COLOR in routers/cropmasks.py, which renders the
 *  tiles; magenta because green cropland over green imagery is invisible. */
export const CROPLAND_COLOR = '#ff2bbf';

/**
 * Colour ramps - the ones the original prebuilt map used, because they were clearer.
 *
 * Sequential is **viridis**, for magnitude (yield, production, area, LAI). It is
 * perceptually uniform and strictly monotonic in lightness, so equal steps in value look
 * like equal steps on screen; it separates neighbouring values far better than either the
 * muted warm-sand ramp or the single-hue green that briefly replaced it. It is multi-hue
 * but it is not a rainbow - lightness never doubles back, which is the property that makes
 * rainbows unreadable.
 *
 * Diverging is deep blue -> white -> deep red for values whose sign matters (relative
 * error: under- vs over-prediction), again matching the old map. Scaled symmetrically so
 * white always sits exactly at zero.
 */
const SEQ: Array<[number, number, number]> = [
  [68, 1, 84],
  [72, 40, 120],
  [62, 74, 137],
  [49, 104, 142],
  [38, 130, 142],
  [31, 158, 137],
  [53, 183, 121],
  [110, 206, 88],
  [181, 222, 43],
  [253, 231, 37],
];

const DIV_LOW: [number, number, number] = [49, 54, 149]; // #313695 - under-prediction
const DIV_MID: [number, number, number] = [255, 255, 255]; // white, so zero reads as "no bias"
const DIV_HIGH: [number, number, number] = [165, 0, 38]; // #a50026 - over-prediction

const lerp = (a: [number, number, number], b: [number, number, number], t: number) =>
  `rgb(${Math.round(a[0] + (b[0] - a[0]) * t)},${Math.round(a[1] + (b[1] - a[1]) * t)},${Math.round(
    a[2] + (b[2] - a[2]) * t,
  )})`;

/** t in [0,1] along the sequential ramp. */
export function rampColor(t: number): string {
  const c = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0));
  const x = c * (SEQ.length - 1);
  const i = Math.min(SEQ.length - 2, Math.floor(x));
  return lerp(SEQ[i], SEQ[i + 1], x - i);
}

/** t in [-1,1]; 0 is the neutral midpoint. */
export function divergingColor(t: number): string {
  const c = Math.min(1, Math.max(-1, Number.isFinite(t) ? t : 0));
  return c < 0 ? lerp(DIV_MID, DIV_LOW, -c) : lerp(DIV_MID, DIV_HIGH, c);
}

/** Metrics whose sign carries meaning, so they get the diverging ramp. */
export const DIVERGING_METRICS = new Set(['rel_error']);

export function isDiverging(metric: string): boolean {
  return DIVERGING_METRICS.has(metric);
}

/**
 * Colour for a value. Diverging metrics are scaled symmetrically about zero so that the
 * neutral midpoint always sits at zero - scaling them to their own min/max would put the
 * midpoint at an arbitrary value and invert the meaning of the two hues.
 */
/**
 * Upper bound for a sequential domain: the 95th percentile, not the maximum. One outlying
 * region otherwise squeezes every other value into the bottom of the ramp - this is what
 * the prebuilt map did, and it is most of why it read more clearly.
 */
export function sequentialDomain(values: number[]): { min: number; max: number } {
  const v = values.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return { min: 0, max: 1 };
  const p95 = v[Math.min(v.length - 1, Math.floor(v.length * 0.95))];
  return { min: v[0], max: p95 > v[0] ? p95 : v[v.length - 1] };
}

export function colorFor(value: unknown, min: number, max: number, metric?: string): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return NO_DATA_FILL;
  if (metric && isDiverging(metric)) {
    const bound = Math.max(Math.abs(min), Math.abs(max)) || 1;
    return divergingColor(value / bound);
  }
  if (max <= min) return rampColor(0.5);
  return rampColor((value - min) / (max - min));
}

export function formatNumber(v: unknown, digits = 0): string {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '-';
  return v.toLocaleString(undefined, { maximumFractionDigits: digits });
}

/** Human labels for the stat keys worth surfacing, in display order. */
export const STAT_LABELS: Array<[string, string, number]> = [
  ['mean_yield_kg_ha', 'Mean yield (kg/ha)', 0],
  ['median_yield_kg_ha', 'Median yield (kg/ha)', 0],
  ['reported_mean_yield_kg_ha', 'Reference yield (kg/ha)', 0],
  ['total_production_ton', 'Production (t)', 0],
  ['total_area_ha', 'Cropland area (ha)', 0],
  ['coverage_pct', 'Coverage (%)', 1],
  ['abs_error', 'Absolute error (kg/ha)', 0],
  ['rel_error', 'Relative error (%)', 1],
  ['mean_yield_kg_ha_apsim', 'Mean yield, APSIM only (kg/ha)', 0],
  ['max_rs_lai', 'Peak observed LAI', 2],
  ['n_days_with_rs_data_valid', 'Valid LAI days', 0],
];
