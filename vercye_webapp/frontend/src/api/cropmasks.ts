
import { http } from './client';

export const CropmasksAPI = {
  list: () => http.get<{id: string}[]>('/cropmasks'),
  create: (cropmaskFile: File, cropmaskName: string) => {
    const fd = new FormData();
    fd.append('cropmask_file',cropmaskFile);
    return http.post<void, FormData>(`/cropmasks/${cropmaskName}`, fd);
  }
};
