/**
 * Sentinel-2 monthly composites from Microsoft Planetary Computer.
 *
 * Called straight from the browser: the Data API serves `Access-Control-Allow-Origin: *`
 * and signs blob access internally, so no token and no backend proxy are needed.
 *
 * "First valid composite" is the mosaic API's own behaviour - register a STAC search for
 * a month, and its tiler composites the ordered results taking the first non-masked pixel
 * per location. Ordering by cloud cover ascending therefore yields the clearest available
 * observation for each pixel in that month.
 */
const PC = 'https://planetarycomputer.microsoft.com/api/data/v1';

/**
 * The collection's prepared true-colour asset rather than a hand-rolled B04/B03/B02
 * stretch: it is already tone-mapped by the data provider, so it looks right without a
 * colour formula, and the tiler reads one COG instead of three - measured 560 ms per
 * tile against 1667 ms for the band combination.
 */
const RENDER = 'collection=sentinel-2-l2a&assets=visual';

export type Month = { key: string; label: string; start: string; end: string };

/** Inclusive list of months spanned by a season, e.g. 2024-03 .. 2024-09. */
export function monthsBetween(startISO: string, endISO: string): Month[] {
  const start = new Date(`${startISO}T00:00:00Z`);
  const end = new Date(`${endISO}T00:00:00Z`);
  if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime()) || end < start) return [];
  const out: Month[] = [];
  const cur = new Date(Date.UTC(start.getUTCFullYear(), start.getUTCMonth(), 1));
  while (cur <= end && out.length < 60) {
    const y = cur.getUTCFullYear();
    const m = cur.getUTCMonth();
    const first = new Date(Date.UTC(y, m, 1));
    const last = new Date(Date.UTC(y, m + 1, 0));
    out.push({
      key: `${y}-${String(m + 1).padStart(2, '0')}`,
      label: first.toLocaleString(undefined, { month: 'short', year: 'numeric', timeZone: 'UTC' }),
      start: first.toISOString().slice(0, 10),
      end: last.toISOString().slice(0, 10),
    });
    cur.setUTCMonth(cur.getUTCMonth() + 1);
  }
  return out;
}

/**
 * Below this zoom the mosaic endpoint answers 204 No Content for every tile: it will not
 * composite a whole region's worth of scenes at once. The published tilejson claims
 * minzoom 0, which is simply wrong - measured, z9 is the first zoom that returns pixels.
 * Layers must carry this as their own minZoom or the map silently shows nothing.
 */
export const MOSAIC_MIN_ZOOM = 9;

const searchCache = new Map<string, Promise<string>>();

/**
 * Register a month's STAC search and return its mosaic searchid (cached per month+area).
 *
 * `bbox` matters: the tiler scans a bounded number of items from the search, and an
 * unconstrained global search sorted by cloud cover never reaches the ones over the study.
 */
export function registerMonth(month: Month, bbox?: [number, number, number, number], maxCloud = 50): Promise<string> {
  const cacheKey = `${month.key}|${maxCloud}|${bbox ? bbox.map((v) => v.toFixed(2)).join(',') : 'global'}`;
  const hit = searchCache.get(cacheKey);
  if (hit) return hit;

  const body: Record<string, unknown> = {
    collections: ['sentinel-2-l2a'],
    datetime: `${month.start}T00:00:00Z/${month.end}T23:59:59Z`,
    ...(bbox ? { bbox } : {}),
    query: { 'eo:cloud_cover': { lt: maxCloud } },
    // Clearest scene first, so the mosaic's first-valid pixel is the least cloudy one.
    sortby: [{ field: 'properties.eo:cloud_cover', direction: 'asc' }],
  };

  const p = fetch(`${PC}/mosaic/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
    .then((r) => {
      if (!r.ok) throw new Error(`Planetary Computer register failed (${r.status})`);
      return r.json();
    })
    .then((j) => j.searchid as string)
    .catch((e) => {
      searchCache.delete(cacheKey); // let a later attempt retry
      throw e;
    });

  searchCache.set(cacheKey, p);
  return p;
}

/** XYZ template for a registered month. `collection` must be on the tile request itself. */
export function mosaicTileUrl(searchid: string): string {
  return `${PC}/mosaic/${searchid}/tiles/WebMercatorQuad/{z}/{x}/{y}?${RENDER}`;
}
