import { CropmasksAPI } from "@/api/cropmasks";
import type { Cropmask } from "@/api/cropmasks";
import CropmaskViewer from "@/components/CropmaskViewer";
import FileUpload from "@/components/FileUpload"
import Header from "@/components/Header"
import { MapIcon } from "@/components/Icons";
import Modal from "@/components/Modal"
import useToast from "@/components/Toast";
import { useEffect, useState } from "react";

/** Season metadata is optional so masks uploaded before it existed stay valid; when it is
 *  present it bounds the Sentinel-2 month slider in the viewer. */
type SeasonDraft = { year: string; start: string; end: string };
const EMPTY_SEASON: SeasonDraft = { year: '', start: '', end: '' };

function formatSize(bytes: number | null): string {
  if (!bytes) return '-';
  const mb = bytes / 1048576;
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb.toFixed(0)} MB`;
}

const CropmasksPage = () => {
  const [createOpen, setCreateOpen] = useState<boolean>(false)
  const [cropmaskCreateName, setCropmaskCreateName] = useState<string | null>(null)
  const [cropmaskFiles, setCropmaskFiles] = useState<File[]>([])
  const [cropmasks, setCropmasks] = useState<Cropmask[] | null>(null)
  const [season, setSeason] = useState<SeasonDraft>(EMPTY_SEASON)
  const [viewing, setViewing] = useState<string | null>(null)
  const [editing, setEditing] = useState<Cropmask | null>(null)
  const [editSeason, setEditSeason] = useState<SeasonDraft>(EMPTY_SEASON)
  const [uploading, setUploading] = useState(false)

  const { show, Toast } = useToast();

  const loadCropmasks = async () => {
    try {
      const res = await CropmasksAPI.list();
      setCropmasks(res);
    } catch {
      setCropmasks([]);
      show('Failed to load cropmasks', 'error');
    }
  };

  useEffect(() => {
    loadCropmasks();
    const iv = setInterval(() => { loadCropmasks() }, 10000);
    return () => clearInterval(iv);
  }, []);

  /** All three season fields or none - the backend rejects a partial range. */
  const seasonPayload = (s: SeasonDraft) => {
    const filled = [s.year, s.start, s.end].filter(Boolean).length;
    if (filled === 0) return undefined;
    if (filled < 3) return 'incomplete' as const;
    return { year: Number(s.year), season_start: s.start, season_end: s.end };
  };

  const handleCropmaskSubmit = async () => {
    if (cropmaskFiles.length === 0) { show('Must select a cropmask file first.', 'error'); return }
    if (!cropmaskCreateName) { show('Must set a cropmask name first.', 'error'); return }
    const payload = seasonPayload(season);
    if (payload === 'incomplete') {
      show('Give year, season start and season end together, or leave all three empty.', 'error');
      return;
    }
    setUploading(true);
    try {
      await CropmasksAPI.create(cropmaskFiles[0], cropmaskCreateName, payload);
      show('Cropmask uploaded', 'success');
      setCropmaskFiles([]);
      setCropmaskCreateName("");
      setSeason(EMPTY_SEASON);
      setCreateOpen(false);
      loadCropmasks();
    } catch (e) {
      show(e instanceof Error ? e.message : 'Failed to upload cropmask', 'error');
    } finally {
      setUploading(false);
    }
  }

  const saveMetadata = async () => {
    if (!editing) return;
    const payload = seasonPayload(editSeason);
    if (payload === 'incomplete') {
      show('Give year, season start and season end together, or leave all three empty.', 'error');
      return;
    }
    try {
      await CropmasksAPI.setMetadata(editing.name, payload);
      show('Season updated', 'success');
      setEditing(null);
      loadCropmasks();
    } catch (e) {
      show(e instanceof Error ? e.message : 'Failed to save', 'error');
    }
  };

  const seasonFields = (s: SeasonDraft, set: (v: SeasonDraft) => void) => (
    <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
      <div style={{ flex: '0 0 110px' }}>
        <label className="form-label">Year</label>
        <input className="form-input" type="number" placeholder="2024" value={s.year}
          onChange={e => set({ ...s, year: e.target.value })} />
      </div>
      <div style={{ flex: '1 1 160px' }}>
        <label className="form-label">Season start</label>
        <input className="form-input" type="date" value={s.start} onChange={e => set({ ...s, start: e.target.value })} />
      </div>
      <div style={{ flex: '1 1 160px' }}>
        <label className="form-label">Season end</label>
        <input className="form-input" type="date" value={s.end} onChange={e => set({ ...s, end: e.target.value })} />
      </div>
    </div>
  );

  const cropMasksTable = () => {
    if (!cropmasks)
      return (
        <div className="empty-state">
          <div className="loading"></div>
          <p style={{ marginTop: '1rem' }}>Loading available cropmasks data...</p>
        </div>
      );
    if (!cropmasks.length)
      return (
        <div className="empty-state">
          <h3>No Cropmasks available</h3>
          <p>Check back later.</p>
        </div>
      );
    return (
      <div className="table-container">
        <table className="table">
          <thead>
            <tr>
              <th>Cropmask Name</th>
              <th>Year</th>
              <th>Season</th>
              <th>Size</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {cropmasks.map((e) => (
              <tr key={e.id}>
                <td>{e.name}</td>
                <td>{e.year ?? <span style={{ color: 'var(--neutral-400)' }}>-</span>}</td>
                <td style={{ fontSize: 13 }}>
                  {e.season_start
                    ? `${e.season_start} → ${e.season_end}`
                    : <span style={{ color: 'var(--neutral-400)' }}>not set</span>}
                </td>
                <td style={{ fontSize: 13 }}>{formatSize(e.size_bytes)}</td>
                <td>
                  <div className="actions-cell">
                    <button className="btn btn-sm btn-primary btn-icon" title="View on map" aria-label="View on map"
                      onClick={() => setViewing(e.name)}><MapIcon /></button>
                    <button className="btn btn-sm btn-secondary" title="Set year and season"
                      onClick={() => {
                        setEditing(e);
                        setEditSeason({
                          year: e.year ? String(e.year) : '',
                          start: e.season_start ?? '',
                          end: e.season_end ?? '',
                        });
                      }}>{e.season_start ? 'Edit season' : 'Set season'}</button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }

  if (viewing) return <CropmaskViewer name={viewing} onClose={() => setViewing(null)} />;

  return (
    <div className="container">
      <Header />
      <div className="page-header">
        <h1 className="title">Cropmasks</h1>
        <p className="subtitle">Upload, inspect and season-tag cropland masks.</p>
      </div>
      <div className="actions">
        <button className="btn btn-primary" onClick={() => setCreateOpen(true)}>
          ✚ Upload Cropmask
        </button>
      </div>

      {cropMasksTable()}

      <Modal open={createOpen} onClose={() => setCreateOpen(false)} title="Upload a Cropmask" width={800}>
        <div className="form-group">
          <label htmlFor="studyName" className="form-label">Cropmask Name</label>
          <input id="studyName" className="form-input" placeholder="Enter cropmask name" value={cropmaskCreateName ? cropmaskCreateName : ''} onChange={e => setCropmaskCreateName(e.target.value)} />
          <p className="subtitle" style={{ marginTop: 4 }}>
            The Cropmask must be a binary (0/1) geotiff. Use 0 for the non-crop class and 1 for the pixels containing the crop of interest. It is reccomended to upload a compressed file.
          </p>
        </div>

        <div className="form-group">
          <label className="form-label">Cropmask File (.tif)</label>
          <FileUpload
            id="cropmaskFile"
            accept=".tif"
            label="📁 Choose cropmask file (.tif)"
            value={cropmaskFiles}
            onChange={setCropmaskFiles}
          />
        </div>

        <div className="form-group">
          <label className="form-label">Year &amp; growing season <span style={{ fontWeight: 400, color: 'var(--neutral-500)' }}>(optional)</span></label>
          {seasonFields(season, setSeason)}
          <p className="subtitle" style={{ marginTop: 4 }}>
            Used to bound the Sentinel-2 imagery time slider when inspecting this mask.
            Leave empty to skip - the viewer then shows basemaps only.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end' }}>
          <button className="btn btn-secondary" onClick={() => setCreateOpen(false)}>Cancel</button>
          <button className="btn btn-primary" onClick={handleCropmaskSubmit} disabled={uploading}>
            {uploading ? 'Uploading…' : 'Upload'}
          </button>
        </div>
      </Modal>

      <Modal open={!!editing} onClose={() => setEditing(null)} title={`Season for ${editing?.name ?? ''}`} width={620}>
        {seasonFields(editSeason, setEditSeason)}
        <p className="subtitle" style={{ marginTop: 8 }}>
          Clear all three fields to remove the season; the viewer then falls back to basemaps only.
        </p>
        <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end', marginTop: 12 }}>
          <button className="btn btn-secondary" onClick={() => setEditing(null)}>Cancel</button>
          <button className="btn btn-primary" onClick={saveMetadata}>Save</button>
        </div>
      </Modal>

      <Toast />
    </div>
  )
}

export default CropmasksPage
