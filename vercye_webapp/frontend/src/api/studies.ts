import { SetupSubmissionsPayload } from '@/components/Forms/SetupStudyForm';
import { http } from './client';
import type { StudyId, StudyStatusResponse, RunConfigStatusResponse, SetupConfigTemplate, RunConfigFormParams } from '@/types';
import { RunParamsSubmissionsPayload } from '@/components/Forms/RunParamsForm';

export type PagedStudies = { items: StudyId[]; total: number; page: number; page_size: number };

export const StudiesAPI = {
  listPaged: (page: number, pageSize: number) =>
    http.get<PagedStudies>(`/studies?page=${page}&page_size=${pageSize}`),

  create: (study_id: string) =>
    http.post<void, { study_id: string }>('/studies', { study_id }),

  statusMany: (ids: StudyId[]) =>
    http.get<Record<string, StudyStatusResponse["status"]>>(
      `/studies/status?${ids.map(id => `ids=${encodeURIComponent(id)}`).join("&")}`
    ),

  runConfig: (id: StudyId) =>
    http.get<Blob>(`/studies/${id}/run-config`),

  runConfigStatus: (id: StudyId) =>
    http.get<RunConfigStatusResponse>(`/studies/${id}/run-config-status`),

  runConfigFormParams: (id: StudyId) =>
    http.get<RunConfigFormParams | null>(`/studies/${id}/run-config-formdata`),

  downloadRunConfigTemplate: (id: StudyId) =>
    http.download(`/studies/${id}/run-config`),

  getSetupConfig: (id: StudyId) =>
    http.get<SetupConfigTemplate | null>(`/studies/${id}/setup-config`),

  uploadSetup: (id: StudyId, payload: SetupSubmissionsPayload) => {
    const fd = new FormData();

    fd.append("shapefile", payload.shapefile);
    payload.apsimFiles?.forEach((f) => fd.append("apsim_files", f));
    payload.referenceFiles?.forEach((f) => fd.append("reference_files", f));

    const { shapefile: _drop1, apsimFiles: _drop2, referenceFiles: _drop3, ...meta } = payload;

    // Convert camelCase keys in meta to snake_case here before sending
    const metaSnakeCase = {
      region_extraction: {
        admin_name_column: meta.regionExtraction.adminNameColumn,
        target_projection: meta.regionExtraction.targetProjection,
        filter: meta.regionExtraction.filter
          ? {
              column: meta.regionExtraction.filter.column,
              allow: meta.regionExtraction.filter.allow
            }
          : undefined
      },
      apsim_column: meta.apsimColumn,
      apsim_mapping: meta.apsimMapping,
      reference_mapping: meta.referenceMapping,
      reference_years_mapping: meta.referenceYearsMapping,
      years: meta.years,
      timepoints: meta.timepoints,
      simulation_windows: meta.simulationWindows
    };

    fd.append("setup_data", JSON.stringify(metaSnakeCase));
    return http.post<void, FormData>(`/studies/${id}/setup`, fd);
  },

  uploadRunConfig: (id: StudyId, payload: RunParamsSubmissionsPayload) => {
    const fd = new FormData();
    fd.append('run_cfg_file', payload.configFile);

    if (payload.laiSourceId && payload.laiSourceResolution) {
      const laiSnakeCase = {
        lai_source_id: payload.laiSourceId,
        lai_source_resolution: payload.laiSourceResolution
      }
      fd.append("lai_config", JSON.stringify(laiSnakeCase));
    }

    let cropmaskMapping: { [year: number]: string } = {}


    for (const [year, cropmask] of Object.entries(payload.cropmaskMapping)) {
      if(Array.isArray(cropmask)) {
        throw Error("Array of states encounterd in cropmask mapping")
      }
      cropmaskMapping[Number(year)] = cropmask;
    }

    fd.append("cropmask_mapping", JSON.stringify(cropmaskMapping))

    return http.post<void, FormData>(`/studies/${id}/run-config`, fd);
  },

  updateRunConfig: (id: StudyId, file: File) => {
    const fd = new FormData();
    fd.append('config', file);
    return http.put<void, FormData>(`/studies/${id}/run-config`, fd);
  },

  run: (id: StudyId, forceRerun: boolean) =>
    http.post<void>(`/studies/${id}/actions/run?forceRerun=${forceRerun}`),

  cancel: (id: StudyId) =>
    http.post<void>(`/studies/${id}/actions/cancel`),

  forceCancel: (id: StudyId) =>
    http.post<void>(`/studies/${id}/actions/cancel?force=true`),

  resultsUrl: (id: StudyId) =>
    `/api/studies/${id}/results`,

  logs: (id: StudyId) =>
    http.get<string>(`/studies/${id}/log`),

  fullLog: (id: StudyId) =>
    http.download(`/studies/${id}/full-log`),

  resultTimepoints: (id: StudyId) =>
    http.get<{ timepoints: Record<string, string[]> }>(`/studies/${id}/result-timepoints`),

  mapResultUrl: (id: StudyId, year: string, timepoint: string) =>
    `/api/studies/${id}/map-result/${year}/${timepoint}`,

  studyYears: (id: StudyId) =>
    http.get<number[]>(`/studies/${id}/required-years`),

  multiyearSummaryReportUrl: (id: StudyId) =>  `/api/studies/${id}/multiyear-report`,

  downloadReport: (id: StudyId, year: string, timepoint: string) =>
    http.download(`/studies/${id}/report/${year}/${timepoint}`),

  duplicate: (existingStudyId: StudyId, newStudyId: StudyId) =>
    http.post<void, { new_study_id: StudyId} >(
      `/studies/${existingStudyId}/duplicate`,
      { new_study_id: newStudyId }
    ),

    delete: (id: StudyId) =>
      http.del<void>(`/studies/${id}`),
};
