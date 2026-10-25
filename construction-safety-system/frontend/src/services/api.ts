import axios from 'axios';
import { Zone, Incident, KPIStats } from '../types';

const apiClient = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // Zones
  getZones: async (): Promise<Zone[]> => {
    const res = await apiClient.get<Zone[]>('/zones');
    return res.data;
  },
  createZone: async (zone: Omit<Zone, 'id' | 'created_at' | 'is_active'>): Promise<Zone> => {
    const res = await apiClient.post<Zone>('/zones', zone);
    return res.data;
  },
  deleteZone: async (id: number): Promise<void> => {
    await apiClient.delete(`/zones/${id}`);
  },

  // Incidents
  getIncidents: async (status?: string, severity?: string): Promise<Incident[]> => {
    const params: Record<string, string> = {};
    if (status) params.status = status;
    if (severity) params.severity = severity;
    const res = await apiClient.get<Incident[]>('/incidents', { params });
    return res.data;
  },
  updateIncidentStatus: async (id: number, newStatus: string): Promise<Incident> => {
    const res = await apiClient.patch<Incident>(`/incidents/${id}/status`, { status: newStatus });
    return res.data;
  },
  getSnapshotUrl: (id: number): string => `/api/incidents/${id}/snapshot`,

  // Telemetry & Stream
  getKPIStats: async (): Promise<KPIStats> => {
    const res = await apiClient.get<KPIStats>('/stats');
    return res.data;
  },
  getStreamStatus: async () => {
    const res = await apiClient.get('/stream/status');
    return res.data;
  },
  uploadVideo: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await apiClient.post('/stream/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },
  setStreamSource: async (sourceType: string, sourceUrl?: string) => {
    const formData = new FormData();
    formData.append('source_type', sourceType);
    if (sourceUrl) formData.append('source_url', sourceUrl);
    const res = await apiClient.post('/stream/source', formData);
    return res.data;
  }
};
