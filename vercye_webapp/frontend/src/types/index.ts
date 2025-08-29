import { Geometry } from "geojson";

export type StudyId = string;

export type StudyStatus =
  | 'pending'
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
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
  resolution: number;
  geometry: Geometry
}
