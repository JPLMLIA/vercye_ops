import { useEffect, useMemo, useRef, useState } from 'react';
import Header from '@/components/Header';
import { LAIAPI } from '@/api/lai';
import type { LAIEntry } from '@/types';
import L from 'leaflet';
import Modal from '@/components/Modal';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend,
  ChartOptions,
  ChartData
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import GenerateLAIForm, { GenerateLAIPayload, GenerationMode, AddDatesPayload } from '@/components/Forms/GenerateLAIForm';
import useToast from '@/components/Toast';
import { ApiError } from '@/api/client';
import { ansiToHtml } from '@/utils/utils';

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

const LAIPage = () => {
  const [data, setData] = useState<LAIEntry[] | null>(null);
  const [view, setView] = useState<'table' | 'map'>('table');
  const mapRef = useRef<L.Map | null>(null);

  const [detailOpen, setDetailOpen] = useState(false);
  const [detailsEntry, setDetailsEntry] = useState<LAIEntry | null>(null);

  const [createOpen, setCreateOpen] = useState(false);

  const [addDatesOpen, setAddDatesOpen] = useState(false);
  const [addDatesTarget, setAddDatesTarget] = useState<LAIEntry | null>(null);

  const [logsOpen, setLogsOpen] = useState(false);
  const [logs, setLogs] = useState<string>('Loading logs...');

  const { show, Toast } = useToast();

  const loadLaiData = async () => {
    try {
      const res = await LAIAPI.list();
      setData(res);
    } catch {
      setData([]);
      show('Failed to load lai entries', 'error');
    }
  };

  useEffect(() => {
    loadLaiData();
    const iv = setInterval(async () => {
      loadLaiData();
    }, 10000);
    return () => clearInterval(iv);
  }, []);

  useEffect(() => {
    if (view !== 'map') return;
    if (mapRef.current) return;
    const map = L.map('laiMap').setView([0, 0], 2);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap',
    }).addTo(map);
    mapRef.current = map;
  }, [view]);

  useEffect(() => {
    if (view !== 'map' || !mapRef.current) return;

    const map = mapRef.current;
    const layer = L.layerGroup().addTo(map);

    if (data && data.length) {
      const mergedLAIData: Record<string, {
        id: string;
        resolutions: number[];
        dates: string[];
        status: string;
        geometry: GeoJSON.Geometry;
        lat: number;
        lng: number;
      }> = {};

      for (const e of data) {
        if (mergedLAIData[e.id]) {
          mergedLAIData[e.id].resolutions.push(e.resolution);
        } else {
          mergedLAIData[e.id] = {
            id: e.id,
            resolutions: [e.resolution],
            dates: e.dates,
            status: e.status,
            geometry: e.geometry,
            lat: e.lat,
            lng: e.lng
          };
        }
      }

      let totalBounds = L.latLngBounds([]);

      for (const e of Object.values(mergedLAIData)) {
        try {
          const marker = L.circleMarker([e.lat, e.lng], {
            radius: 8,
            weight: 2,
            color: '#4d688c',
            fillOpacity: 0.9
          }).bindPopup(`
            <div class="custom-popup">
              <div class="popup-title">${e.id}</div>
              <div class="popup-info"><b>Total days:</b> ${e.dates.length}</div>
              <div class="popup-info"><b>Resolutions:</b> ${e.resolutions.join(', ')}m</div>
            </div>
          `);

          marker.addTo(layer);
          totalBounds.extend([e.lat, e.lng]);
        } catch (err) {
          console.log(err);
        }
      }

      if (!map.hasLayer(layer) && totalBounds.isValid()) {
        map.fitBounds(totalBounds, { padding: [40, 40] });
      }
    }

    return () => {
      map.removeLayer(layer);
    };
  }, [view, data]);

  const showLogs = async (laiId: string) => {
    setLogs('Loading logs...');
    setLogsOpen(true);
    try {
      const res = await LAIAPI.logs(laiId);
      setLogs(res);
    } catch (err) {
      if (err instanceof ApiError) {
        show(err.message, 'error');
      } else {
        show('Failed to load logs', 'error');
      }
    }
  };

  const cancelGeneration = async (laiId: string, resolution: number) => {
    try {
      await LAIAPI.cancelGeneration(laiId, resolution);
      show('LAI generation successfully cancelled.', 'success');
      loadLaiData();
    } catch (err) {
      if (err instanceof ApiError) {
        show(err.message, 'error');
      } else {
        show('Failed to cancel generation', 'error');
      }
    }
  };

  const table = useMemo(() => {
    if (!data)
      return (
        <div className="empty-state">
          <div className="loading"></div>
          <p style={{ marginTop: '1rem' }}>Loading available LAI data...</p>
        </div>
      );
    if (!data.length)
      return (
        <div className="empty-state">
          <h3>No LAI entries</h3>
          <p>Check back later.</p>
        </div>
      );
    return (
      <div className="table-container">
        <table className="table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Resolution</th>
              <th>Centroid</th>
              <th>Total days</th>
              <th>Status</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {data.map((e) => (
              <tr key={`${e.id}-${e.resolution}`}>
                <td>{e.id}</td>
                <td>{e.resolution}m</td>
                <td>{e.lat}, {e.lng}</td>
                <td>{e.dates.length}</td>
                <td>
                  <span className={`status-badge status-${e.status}`}>
                    {e.status}
                  </span>
                </td>
                <td>
                  <div className="actions-cell">
                    <button
                      className="btn btn-sm btn-primary"
                      onClick={() => {
                        setDetailsEntry(e);
                        setDetailOpen(true);
                      }}
                      disabled={e.status !== 'completed'}
                    >
                      Details
                    </button>

                    {e.status !== "pending" && (
                      <button
                        className="btn btn-sm btn-primary"
                        onClick={() => {
                          showLogs(e.id);
                        }}
                      >
                        Logs
                      </button>
                    )}

                    {["generating", "standardizing", "finalizing"].includes(e.status) && (
                      <button
                        className="btn btn-sm btn-danger"
                        onClick={() => {
                          cancelGeneration(e.id, e.resolution);
                        }}
                      >
                        Cancel
                      </button>
                    )}

                    {(e.status === "completed" || e.status === "cancelled" || e.status === "failed") && (
                      <button
                        className="btn btn-sm btn-primary"
                        onClick={() => {
                          setAddDatesTarget(e);
                          setAddDatesOpen(true);
                        }}
                      >
                        + Dates
                      </button>
                    )}

                    {(e.status === "failed" || e.status === "cancelled") && (
                      <button
                        className="btn btn-sm btn-primary"
                        onClick={() => {
                          retryLAIGeneration(e.id, e.resolution);
                        }}
                      >
                        ↻  Retry
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }, [data]);

  const chartData: ChartData<'bar'> | null = useMemo(() => {
    if (!detailsEntry) return null;

    const availableDates = new Set(detailsEntry.dates);
    const minDate = new Date(Math.min(...detailsEntry.dates.map((d) => new Date(d).getTime())));
    const maxDate = new Date(Math.max(...detailsEntry.dates.map((d) => new Date(d).getTime())));

    const allDates: string[] = [];
    const values: number[] = [];

    let current = new Date(minDate);
    while (current <= maxDate) {
      const dateStr = current.toISOString().split('T')[0];
      allDates.push(dateStr);
      values.push(availableDates.has(dateStr) ? 1 : 0);
      current.setDate(current.getDate() + 1);
    }

    return {
      labels: allDates,
      datasets: [
        {
          label: 'Data Available',
          data: values,
          backgroundColor: values.map((v) => (v ? 'green' : 'lightgray')),
        },
      ],
    };
  }, [detailsEntry]);

  const chartOptions: ChartOptions<'bar'> = useMemo(
    () => ({
      scales: {
        y: {
          beginAtZero: true,
          max: 1,
          ticks: {
            stepSize: 1,
            callback: (value) => (value ? 'Yes' : ''),
          },
        },
        x: {
          ticks: {
            maxTicksLimit: 15,
            autoSkip: true,
          },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => (ctx.raw ? 'Available' : 'Missing'),
          },
        },
      },
    }),
    []
  );

  const handleCreateLAI = async (payload: GenerateLAIPayload) => {
    try {
      await LAIAPI.create(payload);
      show('LAI generation started', 'success');
      loadLaiData();
      setCreateOpen(false);
    } catch (err) {
      if (err instanceof ApiError) {
        show(err.message, 'error');
      } else {
        show('Failed to create lai data', 'error');
      }
    }
  };

  const handleAddDates = async (payload: AddDatesPayload) => {
    try {
      await LAIAPI.addDates(payload);
      show('Dates added to LAI entry', 'success');
      loadLaiData();
      setAddDatesOpen(false);
      setAddDatesTarget(null);
    } catch (err) {
      if (err instanceof ApiError) {
        show(err.message, 'error');
      } else {
        show('Failed to add dates', 'error');
      }
    }
  };

  const retryLAIGeneration = async (laiID: string, resolution: number) => {
    try {
      await LAIAPI.retryCreate(laiID, resolution);
      show('Retrying LAI generation', 'success');
      loadLaiData();
      setCreateOpen(false);
    } catch (err) {
      if (err instanceof ApiError) {
        show(err.message, 'error');
      } else {
        show('Failed to retry lai generation', 'error');
      }
    }
  };

  return (
    <div className="container">
      <Header />
      <div className="actions">
        <button className="btn btn-primary" onClick={() => setCreateOpen(true)}>
          ✚ Create New LAI Entry
        </button>
        <button className="btn btn-secondary" onClick={() => loadLaiData()}>
          ↻ Refresh
        </button>
      </div>
      <div className="view-toggle">
        <span style={{ color: 'var(--gray-700)', fontWeight: 500, marginRight: 10 }}>View:</span>
        <button className={`toggle-btn ${view === 'table' ? 'active' : ''}`} onClick={() => setView('table')}>
          Table View
        </button>
        <button className={`toggle-btn ${view === 'map' ? 'active' : ''}`} onClick={() => setView('map')}>
          Map View
        </button>
      </div>
      <div id="tableView" style={{ display: view === 'table' ? 'block' : 'none' }}>
        {table}
      </div>
      <div id="mapView" style={{ display: view === 'map' ? 'block' : 'none' }}>
        <div id="mapContainer"><div id="laiMap"></div></div>
      </div>

      <Modal
        open={detailOpen}
        onClose={() => setDetailOpen(false)}
        title={`LAI Details ${detailsEntry ? `- ${detailsEntry.id}` : ''}`}
        width={800}
      >
        {chartData ? (
          <div>
            <h3>LAI Data Availability</h3>
            <Bar data={chartData} options={chartOptions} />
          </div>
        ) : (
          <p>No data available for chart.</p>
        )}
      </Modal>

      {/* Create */}
      <Modal open={createOpen} onClose={() => setCreateOpen(false)} title="Create New LAI Entry" width={800}>
        <GenerateLAIForm
          mode={GenerationMode.InitialCreate}
          onSubmitCreate={handleCreateLAI}
        />
      </Modal>

      {/* Add Dates */}
      <Modal
        open={addDatesOpen}
        onClose={() => {
          setAddDatesOpen(false);
          setAddDatesTarget(null);
        }}
        title={`Add Dates${addDatesTarget ? ` - ${addDatesTarget.id} (${addDatesTarget.resolution}m)` : ''}`}
        width={800}
      >
        {addDatesTarget && (
          <GenerateLAIForm
            mode={GenerationMode.AddDates}
            initialName={addDatesTarget.id}
            initialResolution={addDatesTarget.resolution}
            onSubmitAddDates={handleAddDates}
          />
        )}
      </Modal>

      <Modal open={logsOpen} onClose={() => setLogsOpen(false)} title="Logs" width={1100}>
        <div className="logs-container" style={{ marginTop: '2rem' }}>
          <pre dangerouslySetInnerHTML={{ __html: ansiToHtml(logs || 'No logs available') }} />
        </div>
      </Modal>

      <Toast />
    </div>
  );
};

export default LAIPage;
