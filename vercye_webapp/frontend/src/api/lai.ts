
import { AddDatesPayload, GenerateLAIPayload } from '@/components/Forms/GenerateLAIForm';
import { http } from './client';
import type { LAIEntry } from '@/types';

export const LAIAPI = {
  list: () => http.get<LAIEntry[]>('/lai'),
  create: (payload: GenerateLAIPayload) => {
    const fd = new FormData();
    fd.append('region_shapefile', payload.shapefile);

    const laiConfigSnakeCase = {
      resolution: payload.resolution,
      keep_imagery: payload.keep_imagery,
      name: payload.name,
      date_ranges: payload.date_ranges.map(dr => ({
        start_date: dr.start_date,
        end_date: dr.end_date
      })),
      imagery_src: payload.imagery_src,
      chunk_days: payload.chunk_days
    };

    fd.append("lai_config", JSON.stringify(laiConfigSnakeCase));

    return http.post<void, FormData>(`/lai/actions/generate`, fd);
  },
  addDates: (payload: AddDatesPayload) => {
    const laiConfig = {
      resolution: payload.resolution,
      keep_imagery: payload.keep_imagery,
      name: payload.name,
      date_ranges: payload.date_ranges.map(dr => ({
        start_date: dr.start_date,
        end_date: dr.end_date
      })),
      chunk_days: payload.chunk_days
    }
    const fd = new FormData()
    fd.append("config", JSON.stringify(laiConfig))
    return http.post<void, FormData>(`/lai/actions/add`, fd)
  },
  retryCreate: (laiID: string, resolution: number) => {
    return http.post<void, FormData>(`/lai/${laiID}/${resolution}/actions/regenerate`);
  },

  logs: (id: string) =>
    http.get<string>(`/lai/${id}/logs`),

  cancelGeneration: (id: string, resolution: number) =>
    http.post<string>(`/lai/${id}/${resolution}/actions/cancel`),
};
