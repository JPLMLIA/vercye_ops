import { Geometry } from "geojson";

export type StudyId = string;

export type StudyStatus =
  | 'pending'
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'cancelling'
  | 'validating'
  | 'ready';

export interface StudyStatusResponse {
  status: StudyStatus;
}

export interface RunConfigStatusResponse {
  status: 'valid' | 'invalid';
  details?: string;
}

export interface LAIEntry {
  id: string;
  lat: number;
  lng: number;
  dates: [string];
  status: string;
  status_details?: string | null;
  resolution: number;
  geometry: Geometry
}

export interface Feature {
  geometry: any;
  properties: Record<string, any>;
}

export interface ShapefileData {
  type: string;
  features: Feature[];
}

export interface RegionFilter {
  column: string;
  allow: string[];
}

export interface RegionExtraction {
  adminNameColumn: string;
  targetProjection: string;
  filter?: RegionFilter;
}

export interface WindowConfigNoId {
  year: string;
  timepoint: string;
  sim_start_date: string;
  sim_end_date: string;
  met_start_date: string;
  met_end_date: string;
  lai_start_date: string;
  lai_end_date: string;
}

export interface AggregationShapefileConfig {
  level_name: string;
  name_column: string;
  reference_yield_column: string | null;
  year_column: string | null;
  columns?: ShapefileColumn[];
}

export interface ShapefileColumn {
  name: string;
  dtype: string;
  is_numeric: boolean;
}

export interface SetupConfigTemplate {
  regionExtraction: RegionExtraction;
  apsimColumn: string;
  apsimMapping: Record<string, string[]>;
  apsimFiles: string[];
  aggregationShapefiles: AggregationShapefileConfig[];
  aggregationShapefileNames: string[];
  years: string[];
  timepoints: string[];
  simulationWindows: WindowConfigNoId[];
  shapefileData: ShapefileData;
  shapefileName: string
}

export interface RunConfigFormParams {
  laiId: string
  laiResolution: number
  cropmasks: Record<string, string>
}

export type RunID = string;

export interface StudyRun {
  run_id: RunID;
  created_at: string | null;
  file_count: number | null;
  uploaded_to: string | null;
  has_multiyear_report: boolean;
  timepoints: Record<string, string[]>;
  size_bytes: number | null;
}
