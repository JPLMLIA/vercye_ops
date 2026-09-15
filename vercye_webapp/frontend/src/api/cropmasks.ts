import { http } from './client';

export type Cropmask = {
  id: string;
  name: string;
  size_bytes: number | null;
  /** Present only when the mask was uploaded with season metadata. Without it the
   *  viewer shows basemaps only - there is no season to build a month slider from. */
  year: number | null;
  season_start: string | null;
  season_end: string | null;
  has_overviews: boolean;
};

export type CropmaskInfo = Cropmask & {
  bounds: [number, number, number, number];
  width: number;
  height: number;
  crs: string;
};

export type SeasonFields = { year?: number | null; season_start?: string | null; season_end?: string | null };

function seasonForm(fd: FormData, season?: SeasonFields) {
  if (season?.year) fd.append('year', String(season.year));
  if (season?.season_start) fd.append('season_start', season.season_start);
  if (season?.season_end) fd.append('season_end', season.season_end);
  return fd;
}

export const CropmasksAPI = {
  list: () => http.get<Cropmask[]>('/cropmasks'),

  info: (name: string) => http.get<CropmaskInfo>(`/cropmasks/${encodeURIComponent(name)}/info`),

  create: (cropmaskFile: File, cropmaskName: string, season?: SeasonFields) => {
    const fd = new FormData();
    fd.append('cropmask_file', cropmaskFile);
    seasonForm(fd, season);
    return http.post<void, FormData>(`/cropmasks/${cropmaskName}`, fd);
  },

  /** Attach or clear season metadata on an existing mask (lets legacy masks gain a slider). */
  setMetadata: (name: string, season?: SeasonFields) =>
    http.put<void, FormData>(`/cropmasks/${encodeURIComponent(name)}/metadata`, seasonForm(new FormData(), season)),

  /** Ask the server to build overviews. Idempotent; returns {ready} immediately when done. */
  prepare: (name: string) =>
    http.post<{ ready: boolean; building: boolean }>(`/cropmasks/${encodeURIComponent(name)}/prepare`),

  tileUrl: (name: string) => `/api/cropmasks/${encodeURIComponent(name)}/tiles/{z}/{x}/{y}.png`,
};
