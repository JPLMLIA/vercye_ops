import { Fragment, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Header from '@/components/Header';
import Modal from '@/components/Modal';
import StatusBadge from '@/components/StatusBadge';
import Stepper from '@/components/Stepper';
import useToast from '@/components/Toast';
import { StudiesAPI } from '@/api/studies';
import type {
  SetupConfigTemplate,
  StudyId,
  StudyStatus,
  RunConfigFormParams,
  RunSummary,
  StudyRun,
  RunID,
} from '@/types';
import SetupStudyForm, { SetupSubmissionsPayload } from '@/components/Forms/SetupStudyForm';
import RunParamsForm, { RunParamsSubmissionsPayload } from '@/components/Forms/RunParamsForm';
import YamlTree from '@/components/YamlTree';
import { ApiError } from '@/api/client';
import { ansiToHtml } from '@/utils/utils';
import { ChevronDownIcon, ChevronRightIcon, CopyIcon, PlusIcon, TrashIcon } from '@/components/Icons';

enum Result {
  Report = 'Report',
}

const SplitButton: React.FC<{
  label: React.ReactNode;
  /** Tooltip / accessible name, needed when `label` is an icon rather than text. */
  title?: string;
  /** Square icon-sized primary button, so the actions column stays on one row. */
  iconOnly?: boolean;
  onPrimary: () => void;
  disabled?: boolean;
  variant?: 'primary' | 'success' | 'secondary' | 'danger';
  size?: 'sm' | 'xs';
  menu: { label: string; onClick: () => void; tooltip?: string }[];
}> = ({ label, title, iconOnly, onPrimary, disabled, variant = 'primary', size = 'sm', menu }) => {
  const [open, setOpen] = useState(false);
  const sizeClass = `btn-${size}`;
  return (
    <div className="btn-split" onMouseLeave={() => setOpen(false)}>
      <button
        className={`btn ${sizeClass} btn-${variant} btn-part1 ${iconOnly ? 'btn-icon' : ''} ${open ? 'open' : ''}`}
        onClick={onPrimary}
        disabled={disabled}
        title={title}
        aria-label={title}
      >{label}</button>
      <button
        className={`btn ${sizeClass} btn-${variant} btn-part2 btn-caret ${open ? 'open' : ''}`}
        aria-haspopup="menu" aria-expanded={open} aria-label="More options"
        onClick={() => setOpen(o => !o)} disabled={disabled}
      > ▾</button>
      {open && (
        <div role="menu" className="dropdown-menu">
          {menu.map((m, i) => (
            <button key={i} className="dropdown-item" onClick={() => { setOpen(false); m.onClick(); }} title={m.tooltip || ''}>
              {m.label}{m.tooltip ? <span aria-hidden style={{ marginLeft: 6, fontWeight: 600 }}> ?</span> : null}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

const formatDate = (iso: string | null | undefined) => {
  if (!iso) return '-';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleString();
};

const formatSize = (bytes: number | null | undefined) => {
  if (!bytes || bytes <= 0) return '-';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let v = bytes;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
};

const StudiesPage = () => {
  const [createOpen, setCreateOpen] = useState(false);
  const [createName, setCreateName] = useState('');
  const [detailOpen, setDetailOpen] = useState(false);
  const [detailStudy, setDetailStudy] = useState<StudyId | null>(null);
  const [step, setStep] = useState<1 | 2 | 3 | 4>(1);
  const [runConfigMessage, setRunConfigMessage] = useState('');
  const [logsOpen, setLogsOpen] = useState(false);
  const [logs, setLogs] = useState<string>('Loading logs...');
  const [duplicateOpen, setDuplicateOpen] = useState(false);
  const [initialSetupData, setInitialSetupData] = useState<SetupConfigTemplate | null>(null);
  const [initialRunConfigData, setInitialRunConfigData] = useState<RunConfigFormParams | null>(null);
  const [deleteOpen, setDeleteOpen] = useState(false);
  // Confirm-before-run: the dialog shows what the configuration will actually do, so the
  // decision is about the run, not about a button label.
  const [runConfirm, setRunConfirm] = useState<{ id: StudyId; force: boolean } | null>(null);
  const [runSummary, setRunSummary] = useState<RunSummary | 'loading' | 'error' | null>(null);
  // Which studies have the "new run" block disclosed. Hidden until asked for, so the
  // panel of a finished study is its history and nothing else.
  const [newRunOpen, setNewRunOpen] = useState<Record<StudyId, boolean>>({});

  // Studies / pagination state
  const [studies, setStudies] = useState<StudyId[] | null>(null);
  const [statuses, setStatuses] = useState<Record<string, StudyStatus>>({});
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [total, setTotal] = useState(0);

  // Expansion / per-study runs state
  const [expanded, setExpanded] = useState<Record<StudyId, boolean>>({});
  const [runsByStudy, setRunsByStudy] = useState<Record<StudyId, StudyRun[] | 'loading' | 'error'>>({});

  // Per-run config viewer state (key: `${studyId}|${runId}`)
  const [configOpen, setConfigOpen] = useState<Record<string, boolean>>({});
  const [configByRun, setConfigByRun] = useState<Record<string, string | 'loading' | 'error'>>({});

  type ApsimManifest = {
    manifest: {
      filter_column: string | null;
      region_to_template: Record<string, string>;
      files: { name: string; size: number | null; regions: string[] }[];
    };
    files: { name: string; size: number }[];
  };
  const [apsimByRun, setApsimByRun] = useState<Record<string, ApsimManifest | 'loading' | 'error'>>({});

  // Result selector modal state (year/timepoint picker, scoped to a specific run)
  const [resultsSelectorOpen, setResultsSelectorOpen] = useState(false);
  const [selectorContext, setSelectorContext] = useState<{
    studyId: StudyId;
    runId: RunID;
    timepoints: Record<string, string[]>;
    type: Result;
  } | null>(null);
  const [selectedYear, setSelectedYear] = useState('');
  const [selectedTimepoint, setSelectedTimepoint] = useState('');

  // Delete-run confirmation state
  const [deleteRunCtx, setDeleteRunCtx] = useState<{ studyId: StudyId; runId: RunID } | null>(null);

  const { show, Toast } = useToast();
  const navigate = useNavigate();

  const [loadingCount, setLoadingCount] = useState(0);
  const [loadingText, setLoadingText] = useState<string | null>(null);
  const isLoading = loadingCount > 0;

  const withLoading = async <T,>(task: () => Promise<T>, message?: string): Promise<T> => {
    if (message) setLoadingText(message);
    setLoadingCount(c => c + 1);
    try {
      return await task();
    } finally {
      setLoadingCount(c => Math.max(0, c - 1));
      setLoadingText(null);
    }
  };

  useEffect(() => { withLoading(load, 'Loading studies…'); }, [page, pageSize]);

  useEffect(() => {
    if (!studies?.length) return;
    const iv = setInterval(async () => {
      try {
        const map = await StudiesAPI.statusMany(studies);
        setStatuses(prev => ({ ...prev, ...map as any }));
      } catch {}
    }, 10000);
    return () => clearInterval(iv);
  }, [studies?.join(',')]);

  // Load runs for every study on the page, not just expanded ones: the list shows each
  // study's latest run name and date, and "expand to load" made that column useless for
  // the one thing you scan the list for - when did this last run.
  useEffect(() => {
    (studies ?? []).forEach((id) => {
      if (runsByStudy[id] === undefined) loadRuns(id);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [studies?.join(',')]);

  // When a study transitions to "completed" while expanded, refresh its runs so
  // the newly archived snapshot appears without requiring a manual reload.
  useEffect(() => {
    Object.entries(statuses).forEach(([id, s]) => {
      if (s === 'completed' && expanded[id]) {
        loadRuns(id);
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [statuses]);

  const load = async () => {
    try {
      const { items, total } = await StudiesAPI.listPaged(page, pageSize);
      setStudies(items);
      setTotal(total);
      if (items.length) {
        try {
          const map = await StudiesAPI.statusMany(items);
          setStatuses(prev => ({ ...prev, ...map as any }));
        } catch {}
      }
    } catch {
      setStudies([]);
      show('Failed to load studies', 'error');
    }
  };

  const loadRuns = async (id: StudyId) => {
    setRunsByStudy(prev => ({ ...prev, [id]: prev[id] && prev[id] !== 'error' ? prev[id] : 'loading' }));
    try {
      const { items } = await StudiesAPI.listRuns(id);
      setRunsByStudy(prev => ({ ...prev, [id]: items }));
    } catch {
      setRunsByStudy(prev => ({ ...prev, [id]: 'error' }));
    }
  };

  const toggleExpand = (id: StudyId) => {
    const willExpand = !expanded[id];
    setExpanded(prev => ({ ...prev, [id]: willExpand }));
    if (willExpand && !Array.isArray(runsByStudy[id])) {
      loadRuns(id);
    }
  };

  const fetchSetInitialSetupData = async (studyID: StudyId) => {
    try {
      const setupConfigData = await StudiesAPI.getSetupConfig(studyID);
      setInitialSetupData(setupConfigData);
    } catch (err) {
      if (err instanceof ApiError) show(err.message, 'error');
      else show('Failed to load setup config template!', 'error');
    }
  };

  const fetchSetInitialRunConfigData = async (studyID: StudyId) => {
    try {
      const runConfigData = await StudiesAPI.runConfigFormParams(studyID);
      setInitialRunConfigData(runConfigData);
    } catch (err) {
      if (err instanceof ApiError) show(err.message, 'error');
      else show('Failed to load initial runconfig data!', 'error');
    }
  };

  const openDetail = (id: StudyId) => {
    setDetailStudy(id);
    withLoading((async () => {
      // These requests are independent, so fire them concurrently instead of waterfalling.
      // fetchSetInitial* swallow their own errors (and surface toasts), so they never reject.
      const [, , runConfigRes, statusRes] = await Promise.allSettled([
        fetchSetInitialSetupData(id),
        fetchSetInitialRunConfigData(id),
        StudiesAPI.runConfig(id),
        StudiesAPI.runConfigStatus(id),
      ]);

      let currentStep: 1 | 2 | 3 | 4 = 2;
      if (runConfigRes.status === 'fulfilled' && statusRes.status === 'fulfilled') {
        currentStep = statusRes.value.status === 'valid' ? 4 : 3;
        setRunConfigMessage(statusRes.value.details || '');
      }
      setStep(currentStep);
      setDetailOpen(true);
    }), 'Loading configuration…');
  };

  const createStudy = async () => {
    if (!createName.trim()) return;
    await withLoading(async () => {
      try {
        await StudiesAPI.create(createName.trim());
        show(`Study "${createName}" created successfully!`, 'success');
        setCreateOpen(false);
        setCreateName('');
        await load();
        setTimeout(() => openDetail(createName.trim()), 400);
      } catch (err) {
        if (err instanceof ApiError) show(err.message, 'error');
        else show('Failed to create study', 'error');
      }
    }, 'Creating study…');
  };

  /** Warm the run summary so the confirmation dialog opens with its contents already in
   *  hand rather than spinning on a round trip the user has to wait through. */
  const summaryCache = useRef<Record<StudyId, RunSummary>>({});
  const prefetchRunSummary = (id: StudyId) => {
    if (summaryCache.current[id]) return;
    StudiesAPI.runSummary(id)
      .then((d) => {
        summaryCache.current[id] = d;
      })
      .catch(() => undefined);
  };

  /** Open the confirmation dialog and load what the run will do. */
  const askToRun = (id: StudyId, forceRerun: boolean) => {
    setRunConfirm({ id, force: forceRerun });
    const cached = summaryCache.current[id];
    setRunSummary(cached ?? 'loading');
    if (cached) return;
    StudiesAPI.runSummary(id)
      .then((d) => {
        summaryCache.current[id] = d;
        setRunSummary(d);
      })
      .catch(() => setRunSummary('error'));
  };

  const runStudy = async (id: StudyId, forceRerun: boolean) => {
    setRunConfirm(null);
    await withLoading(async () => {
      try {
        await StudiesAPI.run(id, forceRerun);
        show('Run started', 'success');
        setStatuses((s) => ({ ...s, [id]: 'running' } as any));
        setDetailOpen(false);
      } catch (err) {
        if (err instanceof ApiError) show(err.message, 'error');
        else show('Failed to start run', 'error');
      }
    }, 'Starting run…');
  };

  const cancelRun = async (id: StudyId, force: boolean = false) => {
    await withLoading(async () => {
      try {
        if (force) {
          await StudiesAPI.forceCancel(id);
          show('Run force-killed', 'success');
        } else {
          await StudiesAPI.cancel(id);
          show('Cancelling run (graceful)...', 'success');
        }
        setStatuses((s) => ({ ...s, [id]: force ? 'cancelled' : 'cancelling' } as any));
      } catch (err) {
        if (err instanceof ApiError) show(err.message, 'error');
        else show('Failed to cancel run', 'error');
      }
    }, force ? 'Force killing…' : 'Cancelling…');
  };

  const requestDelete = (id: StudyId) => { setDetailStudy(id); setDeleteOpen(true); };
  const confirmDelete = async () => {
    if (!detailStudy) return;
    await withLoading(async () => {
      try {
        await StudiesAPI.delete(detailStudy);
        show(`Deleted "${detailStudy}"`, 'success');
        setDeleteOpen(false);
        setDetailStudy(null);
        await load();
      } catch (err) {
        if (err instanceof ApiError) show(err.message, 'error');
        else show('Failed to delete study', 'error');
      }
    }, 'Deleting…');
  };

  const openLogs = async (id: StudyId) => {
    setLogs('Loading logs...');
    setDetailStudy(id);
    setLogsOpen(true);
    try {
      const l = await StudiesAPI.logs(id);
      setLogs(l);
    } catch {
      setLogs('Failed to load logs');
    }
  };

  const downloadFullLog = async (id: StudyId) => {
    try {
      const blob = await StudiesAPI.fullLog(id);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = `${id}_log.txt`; a.click(); URL.revokeObjectURL(url);
    } catch {
      setLogs('Failed to download full log');
    }
  };

  const openRunResultSelector = (studyId: StudyId, run: StudyRun, type: Result) => {
    const years = Object.keys(run.timepoints || {}).sort();
    if (!years.length) {
      show('This run has no timepoints to open.', 'error');
      return;
    }
    const firstYear = years[0];
    const tps = run.timepoints[firstYear] || [];
    setSelectorContext({ studyId, runId: run.run_id, timepoints: run.timepoints, type });
    setSelectedYear(firstYear);
    setSelectedTimepoint(tps[0] || '');
    setResultsSelectorOpen(true);
  };

  const onYearChange = (nextYear: string) => {
    setSelectedYear(nextYear);
    const tps = selectorContext?.timepoints[nextYear] || [];
    setSelectedTimepoint(tps[0] || '');
  };

  const openSelectedResult = async () => {
    if (!selectorContext || !selectedYear || !selectedTimepoint) return;
    const { studyId, runId, type } = selectorContext;
    if (type === Result.Report) {
      try {
        const blob = await StudiesAPI.runReport(studyId, runId, selectedYear, selectedTimepoint);
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `final_report_${studyId}_${runId}_${selectedYear}_${selectedTimepoint}.pdf`;
        a.click();
        URL.revokeObjectURL(url);
      } catch { show('Download failed', 'error'); }
    }
    setResultsSelectorOpen(false);
  };

  const openRunMultiyear = (studyId: StudyId, runId: RunID) => {
    const url = StudiesAPI.runMultiyearUrl(studyId, runId);
    window.open(url, '_blank', 'noopener,noreferrer');
  };

  const downloadRunArchive = async (studyId: StudyId, runId: RunID) => {
    await withLoading(async () => {
      try {
        const blob = await StudiesAPI.runArchiveDownload(studyId, runId);
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = `${studyId}_${runId}.zip`; a.click(); URL.revokeObjectURL(url);
      } catch (err) {
        if (err instanceof ApiError) show(err.message, 'error');
        else show('Download failed', 'error');
      }
    }, 'Preparing archive…');
  };

  const downloadRunConfig = async (studyId: StudyId, runId: RunID) => {
    try {
      const blob = await StudiesAPI.runConfigDownload(studyId, runId);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = `${studyId}_${runId}_config.yaml`; a.click(); URL.revokeObjectURL(url);
    } catch (err) {
      if (err instanceof ApiError) show(err.message, 'error');
      else show('Download failed', 'error');
    }
  };

  const runKey = (studyId: StudyId, runId: RunID) => `${studyId}|${runId}`;

  const toggleRunConfig = async (studyId: StudyId, runId: RunID) => {
    const key = runKey(studyId, runId);
    const willOpen = !configOpen[key];
    setConfigOpen(prev => ({ ...prev, [key]: willOpen }));
    if (!willOpen) return;
    // Fire config + apsim fetches in parallel on first open.
    if (typeof configByRun[key] !== 'string') {
      setConfigByRun(prev => ({ ...prev, [key]: 'loading' }));
      StudiesAPI.runConfigText(studyId, runId)
        .then(text => setConfigByRun(prev => ({ ...prev, [key]: text })))
        .catch(() => setConfigByRun(prev => ({ ...prev, [key]: 'error' })));
    }
    if (!apsimByRun[key] || apsimByRun[key] === 'error') {
      setApsimByRun(prev => ({ ...prev, [key]: 'loading' }));
      StudiesAPI.runApsim(studyId, runId)
        .then(data => setApsimByRun(prev => ({ ...prev, [key]: data })))
        .catch(() => setApsimByRun(prev => ({ ...prev, [key]: 'error' })));
    }
  };

  // Top-level config keys to auto-collapse in the YAML viewer because they
  // contain hundreds of entries that drown out the more interesting params.
  const collapsedConfigKeys = [
    'region',
    'regions',
    'scripts',
    'packaging_params',
    'eval_params',
    'apsim_execution',
    'years',
    'timepoints',
  ];

  const confirmDeleteRun = async () => {
    if (!deleteRunCtx) return;
    const { studyId, runId } = deleteRunCtx;
    await withLoading(async () => {
      try {
        await StudiesAPI.deleteRun(studyId, runId);
        show(`Deleted run "${runId}"`, 'success');
        await loadRuns(studyId);
      } catch (err) {
        if (err instanceof ApiError) show(err.message, 'error');
        else show('Failed to delete run', 'error');
      } finally {
        setDeleteRunCtx(null);
      }
    }, 'Deleting run…');
  };

  const handleCopyLogs = async () => {
    try {
      await navigator.clipboard.writeText(logs || '');
      show('Logs copied to clipboard', 'success');
    } catch {
      show('Failed to copy logs', 'error');
    }
  };

  const duplicateStudy = async () => {
    if (!detailStudy || !createName) return;
    await withLoading(async () => {
      try {
        await StudiesAPI.duplicate(detailStudy, createName);
        show('Study duplicated successfully', 'success');
        setCreateName('');
      } catch (err) {
        if (err instanceof ApiError) show(err.message, 'error');
        else show('Failed to duplicate study!', 'error');
      }
      await load();
    }, 'Duplicating study…');
  };

  const renderRunsPanel = (id: StudyId) => {
    const entry = runsByStudy[id];
    if (entry === 'loading' || !entry) {
      return (
        <div className="runs-panel-empty">
          <div className="loading" />
          <span style={{ marginLeft: 10 }}>Loading runs…</span>
        </div>
      );
    }
    if (entry === 'error') {
      return (
        <div className="runs-panel-empty">
          <span>Could not load runs.</span>
          <button className="btn btn-sm btn-secondary" style={{ marginLeft: 10 }} onClick={() => loadRuns(id)}>Retry</button>
        </div>
      );
    }
    // One run = one row. A run that finished writes an archive at the very end, so a
    // completed run *is* its newest archive entry rather than a second row beside it;
    // the row above the archives exists only while the current run has no archive of its
    // own - never started, still going, or stopped before the snapshot step.
    const s = statuses[id] ?? 'pending';
    const isBusy = s === 'running' || s === 'cancelling' || s === 'queued' || s === 'validating';
    const hasFailed = s === 'failed' || s === 'cancelled';
    const orphaned = s === 'completed' && !entry.length;
    // "Create new run" does not start anything - it opens the next run as a pending row,
    // which is what the study is once its last run has been archived. So there is nothing
    // to cancel, only a configuration to finish and a run to start.
    const drafting = !!newRunOpen[id] && !isBusy && !hasFailed && !orphaned;
    const notRunYet = s === 'pending' || s === 'ready' || drafting;
    const showCurrentRow = isBusy || hasFailed || notRunYet || orphaned;

    const currentStatus: StudyStatus = drafting ? 'pending' : s;
    const currentLabel = isBusy
      ? 'in progress'
      : hasFailed
        ? 'stopped before the archive step'
        : notRunYet
          ? 'not started'
          : 'live outputs';

    const rerunMenu = [
      {
        label: 'Force full rerun',
        onClick: () => askToRun(id, true),
        tooltip: 'Ignore every existing artifact and recompute the whole pipeline from scratch.',
      },
    ];

    return (
      <div className="runs-panel">
        {/* A new run can only be opened when nothing is pending, running or half-finished:
            there is one live output directory, so there is one run at a time. */}
        {!showCurrentRow && (
          <div className="run-launcher-collapsed">
            <button
              className="btn btn-sm btn-primary"
              onClick={() => {
                setNewRunOpen((o) => ({ ...o, [id]: true }));
                prefetchRunSummary(id);
              }}
            ><PlusIcon /> Create new run</button>
          </div>
        )}

        <div className="runs-grid">
          {showCurrentRow && (
            <>
              <div className="runs-section-title">Current run</div>
              <div className={`runs-grid-row runs-grid-current ${isBusy ? 'is-running' : ''} ${hasFailed ? 'is-failed' : ''}`}>
                <div />
                <div>{currentLabel}</div>
                <div>
                  <span className="run-current-badge">Current</span>
                  <StatusBadge status={currentStatus} />
                </div>
                <div className="runs-grid-actions">
                  {isBusy && (
                    <>
                      <button className="btn btn-xs btn-secondary" onClick={() => openLogs(id)}>Logs</button>
                      <SplitButton
                        label="Cancel"
                        variant="danger"
                        size="xs"
                        onPrimary={() => cancelRun(id)}
                        menu={[{ label: 'Force kill', onClick: () => cancelRun(id, true), tooltip: 'Immediately kills all processes without cleanup. May leave incomplete output files.' }]}
                      />
                    </>
                  )}

                  {hasFailed && (
                    <>
                      <button className="btn btn-xs btn-secondary" onClick={() => openLogs(id)}>Logs</button>
                      <button className="btn btn-xs btn-secondary" onClick={() => openDetail(id)}>Configure</button>
                      <SplitButton
                        label="Run again"
                        variant="success"
                        size="xs"
                        title="Resume from where the pipeline stopped: completed steps are kept"
                        onPrimary={() => askToRun(id, false)}
                        menu={rerunMenu}
                      />
                      <span className="runs-hint">
                        No archive was written. The archive is the pipeline's last step, so running
                        again resumes from where it stopped.
                      </span>
                    </>
                  )}

                  {orphaned && (
                    <>
                      <button
                        className="btn btn-xs btn-primary"
                        onClick={() => navigate(`/studies/${id}/results`)}
                      >Results</button>
                      <button className="btn btn-xs btn-secondary" onClick={() => openLogs(id)}>Logs</button>
                      <span className="runs-hint">Completed, but its archive is missing or was deleted.</span>
                    </>
                  )}

                  {notRunYet && (
                    <>
                      <button className="btn btn-xs btn-secondary" onClick={() => openDetail(id)}>Configure</button>
                      <SplitButton
                        label="Run"
                        variant="success"
                        size="xs"
                        onPrimary={() => askToRun(id, false)}
                        menu={rerunMenu}
                      />
                      {drafting && (
                        <button
                          className="btn btn-xs btn-secondary"
                          onClick={() => setNewRunOpen((o) => ({ ...o, [id]: false }))}
                          title="Close this row; nothing has been started"
                        >Discard</button>
                      )}
                      <span className="runs-hint">
                        {drafting
                          ? "Starts from the last run's configuration. Unchanged steps reuse their existing results; a changed input is recomputed along with everything downstream of it."
                          : 'Not run yet. Configure the study, then start the run.'}
                      </span>
                    </>
                  )}
                </div>
              </div>
            </>
          )}

          <div className="runs-section-title">
            Completed runs{entry.length ? ` (${entry.length})` : ''}
          </div>
          <div className="runs-grid-head">
            <div></div>
            <div>Completed</div>
            <div>Run</div>
            <div>Actions</div>
          </div>

          {!entry.length && (
            <div className="runs-grid-row">
              <div />
              <div className="runs-hint" style={{ gridColumn: '2 / -1' }}>
                No completed runs yet. An archive is written at the end of every successful run.
              </div>
            </div>
          )}
          {entry.map((run, i) => {
            const key = runKey(id, run.run_id);
            const cfgOpen = !!configOpen[key];
            const cfg = configByRun[key];
            // Once the config viewer has ever been opened we keep its block
            // mounted and only flip visibility via CSS so toggling is instant
            // (no remount of the YAML tree's hundreds of nested nodes).
            const cfgMounted = cfgOpen || cfg !== undefined;
            return (
              <Fragment key={run.run_id}>
                <div className="runs-grid-row">
                  <div>
                    <button
                      className="row-toggle"
                      aria-label={cfgOpen ? 'Hide config' : 'Show config'}
                      aria-expanded={cfgOpen}
                      onClick={() => toggleRunConfig(id, run.run_id)}
                    >
                      <span className={`chevron ${cfgOpen ? 'open' : ''}`}>▶</span>
                    </button>
                  </div>
                  <div>{formatDate(run.created_at)}</div>
                  <div>
                    <code style={{ fontSize: 12 }}>{run.run_id}</code>
                    {i === 0 && s === 'completed' && (
                      <span className="run-latest-badge" title="This run's outputs are the study's live outputs - a plain Run builds on them">
                        latest
                      </span>
                    )}
                  </div>
                  <div className="runs-grid-actions">
                    <button
                      className="btn btn-xs btn-primary"
                      onClick={() => navigate(`/studies/${id}/runs/${run.run_id}/results`)}
                      disabled={!Object.keys(run.timepoints || {}).length}
                      title="Open the interactive results map for this archived run"
                    >Results</button>
                    <button
                      className="btn btn-xs btn-secondary"
                      onClick={() => openRunResultSelector(id, run, Result.Report)}
                      disabled={!Object.keys(run.timepoints || {}).length}
                    >Report</button>
                    <button
                      className="btn btn-xs btn-secondary"
                      onClick={() => openRunMultiyear(id, run.run_id)}
                      disabled={!run.has_multiyear_report}
                    >Multiyear</button>
                    <SplitButton
                      label="Download"
                      variant="secondary"
                      size="xs"
                      title={`Download this run - ${run.file_count ?? '?'} files, ${formatSize(run.size_bytes)}`}
                      onPrimary={() => downloadRunArchive(id, run.run_id)}
                      menu={[
                        { label: 'Run archive (.zip)', onClick: () => downloadRunArchive(id, run.run_id) },
                        { label: 'config.yaml', onClick: () => downloadRunConfig(id, run.run_id) },
                      ]}
                    />
                    <button
                      className="btn btn-xs btn-secondary btn-delete"
                      title="Delete this archived run"
                      onClick={() => setDeleteRunCtx({ studyId: id, runId: run.run_id })}
                    >🗑</button>
                  </div>
                </div>
                {cfgMounted && (
                  <div className="runs-grid-config" style={cfgOpen ? undefined : { display: 'none' }}>
                    <div className="run-config-viewer">
                      <div className="run-config-header">
                        <strong>config.yaml</strong>
                        <button
                          className="btn btn-xs btn-secondary"
                          onClick={() => downloadRunConfig(id, run.run_id)}
                        >Download</button>
                      </div>
                      {cfg === 'loading' || cfg === undefined ? (
                        <div className="run-config-empty">
                          <div className="loading" />
                          <span style={{ marginLeft: 10 }}>Loading config…</span>
                        </div>
                      ) : cfg === 'error' ? (
                        <div className="run-config-empty">
                          Could not load config.
                          <button className="btn btn-xs btn-secondary" style={{ marginLeft: 10 }} onClick={() => toggleRunConfig(id, run.run_id)}>Retry</button>
                        </div>
                      ) : (
                        <div className="run-config-tree">
                          <YamlTree text={cfg} collapsedKeys={collapsedConfigKeys} />
                        </div>
                      )}

                      <div className="run-config-header" style={{ marginTop: 16 }}>
                        <strong>APSIM templates used</strong>
                        {apsimByRun[key] && apsimByRun[key] !== 'loading' && apsimByRun[key] !== 'error' && (apsimByRun[key] as ApsimManifest).manifest.filter_column && (
                          <span style={{ color: 'var(--neutral-500)', fontSize: 11 }}>
                            mapped via column <code>{(apsimByRun[key] as ApsimManifest).manifest.filter_column}</code>
                          </span>
                        )}
                      </div>
                      {apsimByRun[key] === 'loading' || apsimByRun[key] === undefined ? (
                        <div className="run-config-empty">
                          <div className="loading" />
                          <span style={{ marginLeft: 10 }}>Loading APSIM mapping…</span>
                        </div>
                      ) : apsimByRun[key] === 'error' ? (
                        <div className="run-config-empty">Could not load APSIM mapping.</div>
                      ) : (apsimByRun[key] as ApsimManifest).files.length === 0 ? (
                        <div className="run-config-empty" style={{ minHeight: 60 }}>
                          No APSIM templates were captured for this run.
                        </div>
                      ) : (
                        <div className="apsim-files">
                          {(apsimByRun[key] as ApsimManifest).files.map(f => {
                            const fileManifest = (apsimByRun[key] as ApsimManifest).manifest.files.find(x => x.name === f.name);
                            const regions = fileManifest?.regions ?? [];
                            return (
                              <div className="apsim-file" key={f.name}>
                                <div className="apsim-file-info">
                                  <code className="apsim-file-name">{f.name}</code>
                                  <span className="apsim-file-meta">
                                    {formatSize(f.size)} · {regions.length} region{regions.length === 1 ? '' : 's'}
                                  </span>
                                </div>
                                <details className="apsim-file-regions">
                                  <summary>Show regions</summary>
                                  <div className="apsim-region-list">
                                    {regions.length === 0 ? <em>- no regions in mapping -</em> : regions.join(', ')}
                                  </div>
                                </details>
                                <a
                                  className="btn btn-xs btn-secondary"
                                  href={StudiesAPI.runApsimFileUrl(id, run.run_id, f.name)}
                                  download={f.name}
                                >Download</a>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </Fragment>
            );
          })}
        </div>
      </div>
    );
  };

  const table = useMemo(() => {
    if (!studies) return (
      <div className="empty-state">
        <div className="loading"></div>
        <p style={{ marginTop: '1rem' }}>Loading studies...</p>
      </div>
    );
    if (studies.length === 0) return (
      <div className="empty-state">
        <h3>No Studies Found</h3>
        <p>Create your first study to get started</p>
        <button className="btn btn-primary" style={{ marginTop: 15 }} onClick={() => setCreateOpen(true)}>Create Your First Study</button>
      </div>
    );
    return (
      <>
        <div className="table-container">
          <table className="table studies-table">
            <thead>
              <tr>
                <th>Study Name</th>
                <th style={{ width: 160 }}>Status</th>
                <th style={{ width: 200 }}>Latest Run</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {studies.map((id) => {
                const s = statuses[id] ?? 'pending';
                const canConfigure = s !== 'running';
                const isOpen = !!expanded[id];
                const runs = Array.isArray(runsByStudy[id]) ? (runsByStudy[id] as StudyRun[]) : null;
                const latest = runs?.[0];

                return (
                  <Fragment key={id}>
                    {/* The whole row is the target: a 4px chevron in its own column was the
                        only way to open a study, and it sat next to a row that already
                        looked clickable. The actions cell stops propagation so pressing a
                        button there does not also toggle the row. */}
                    <tr
                      className={isOpen ? 'study-row open' : 'study-row'}
                      onClick={() => toggleExpand(id)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          toggleExpand(id);
                        }
                      }}
                      tabIndex={0}
                      role="button"
                      aria-expanded={isOpen}
                      aria-label={`${isOpen ? 'Collapse' : 'Expand'} ${id}`}
                    >
                      <td>
                        <div style={{ fontWeight: 500 }}>{id}</div>
                        {runs && runs.length > 0 && (
                          <div style={{ fontSize: 12, color: 'var(--neutral-500)', marginTop: 2 }}>
                            {runs.length} archived run{runs.length === 1 ? '' : 's'}
                          </div>
                        )}
                      </td>
                      <td><StatusBadge status={s} /></td>
                      <td style={{ fontSize: 13, color: 'var(--neutral-600)' }}>
                        {latest ? (
                          <>
                            <code style={{ fontSize: 11.5 }}>{latest.run_id}</code>
                            <div style={{ fontSize: 11.5, color: 'var(--neutral-500)' }}>{formatDate(latest.created_at)}</div>
                          </>
                        ) : runs ? (
                          <span style={{ color: 'var(--neutral-400)' }}>
                            {s === 'running' ? 'running now' : 'never run'}
                          </span>
                        ) : (
                          <span style={{ color: 'var(--neutral-400)' }}>expand to load</span>
                        )}
                      </td>
                      <td>
                        {/* Icons rather than text labels: the six actions now fit on a
                            single row instead of wrapping onto two. Each carries a
                            title + aria-label so the meaning is still available. */}
                        <div className="actions-cell" onClick={(e) => e.stopPropagation()}>
                          {/* Per-run actions (run, results, logs, configure) live in the
                              dropdown next to the run they act on. What stays here is what
                              acts on the study as a whole. */}
                          <button
                            className="btn btn-sm btn-primary btn-icon"
                            onClick={() => toggleExpand(id)}
                            title={isOpen ? 'Hide runs' : 'Show runs and actions'}
                            aria-label={isOpen ? 'Hide runs' : 'Show runs and actions'}
                          >{isOpen ? <ChevronDownIcon /> : <ChevronRightIcon />}</button>
                          <button className="btn btn-sm btn-secondary btn-icon" onClick={() => { setDetailStudy(id); setDuplicateOpen(true); }} title="Use as template for a new study" aria-label="Use as template for a new study"><CopyIcon /></button>
                          <button className="btn btn-sm btn-secondary btn-delete btn-icon" title="Delete study" aria-label="Delete study" onClick={() => requestDelete(id)} disabled={!canConfigure}><TrashIcon /></button>
                        </div>
                      </td>
                    </tr>
                    {isOpen && (
                      <tr className="study-row-expanded">
                        <td colSpan={4} onClick={(e) => e.stopPropagation()}>{renderRunsPanel(id)}</td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 12 }}>
          <div style={{ fontSize: 13, color: 'var(--neutral-600)' }}>
            Showing <strong>{studies.length ? (page - 1) * pageSize + 1 : 0}</strong>-
            <strong>{Math.min(page * pageSize, total)}</strong> of <strong>{total}</strong>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <label className="form-label" style={{ margin: 0 }}>Per page</label>
            <select className="form-input" value={pageSize} onChange={e => { setPage(1); setPageSize(Number(e.target.value)); }} style={{ width: 90, padding: '6px 8px', height: 34 }}>
              {[10, 20, 50, 100].map(n => <option key={n} value={n}>{n}</option>)}
            </select>
            <button className="btn btn-secondary btn-sm" onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}>← Prev</button>
            <button className="btn btn-secondary btn-sm" onClick={() => setPage(p => (p * pageSize < total ? p + 1 : p))} disabled={page * pageSize >= total}>Next →</button>
          </div>
        </div>
      </>
    );
    // Every piece of state the table reads must be listed. `newRunOpen` and `apsimByRun`
    // were not, so opening a new run row or loading the APSIM mapping did nothing visible
    // until the next status poll happened to invalidate the memo - which read as the UI
    // taking eight seconds to respond to a click.
  }, [studies, statuses, expanded, runsByStudy, configOpen, configByRun, apsimByRun, newRunOpen]);

  const handleSetupSubmission = async (payload: SetupSubmissionsPayload) => {
    if (!detailStudy) return;
    await withLoading(async () => {
      try {
        await StudiesAPI.uploadSetup(detailStudy, payload);
        show('Setup uploaded', 'success');
        await fetchSetInitialSetupData(detailStudy);
        await fetchSetInitialRunConfigData(detailStudy);
        setStep(3);
      } catch (err) {
        if (err instanceof ApiError) show(err.message, 'error');
        else show('Failed to upload setup', 'error');
      }
    }, 'Uploading setup…');
  };

  const handleRunParamsSubmission = async (payload: RunParamsSubmissionsPayload) => {
    if (!detailStudy) return;
    await withLoading(async () => {
      try {
        await StudiesAPI.uploadRunConfig(detailStudy, payload);
        show('Run config uploaded', 'success');
        await fetchSetInitialRunConfigData(detailStudy);
        setStep(4);
      } catch (err) {
        if (err instanceof ApiError) show(err.message, 'error');
        else show('Failed to upload run config', 'error');
        throw new Error('failed to upload run config');
      }
    }, 'Uploading run config…');
  };

  const handleDownloadTemplate = async () => {
    if (!detailStudy) return;
    try {
      const blob = await StudiesAPI.downloadRunConfigTemplate(detailStudy);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `run_config_${detailStudy}.yaml`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      if (err instanceof ApiError) show(err.message, 'error');
      else show('Download failed', 'error');
    }
  };

  const yearOptions = selectorContext ? Object.keys(selectorContext.timepoints).sort() : [];
  const timepointOptions = selectorContext ? (selectorContext.timepoints[selectedYear] || []) : [];

  return (
    <div className="container">
      <Header />
      <div className="page-header">
        <h1 className="title">Studies</h1>
      </div>
      <div className="actions">
        <button className="btn btn-primary" onClick={() => setCreateOpen(true)}><PlusIcon /> New study</button>
        <button className="btn btn-secondary" onClick={() => load()}>Refresh</button>
      </div>
      <div id="studiesContainer">{table}</div>

      {/* Confirm before running: a run is long and overwrites the live outputs, so the
          configuration it will use is put in front of the user first. */}
      <Modal
        open={!!runConfirm}
        onClose={() => setRunConfirm(null)}
        title={runConfirm?.force ? 'Force full rerun' : 'Start run'}
        width={640}
      >
        {runSummary === 'loading' && (
          <div className="run-config-empty"><div className="loading" /><span style={{ marginLeft: 10 }}>Loading configuration...</span></div>
        )}
        {runSummary === 'error' && <div className="run-config-empty">Could not read the run configuration.</div>}
        {runSummary && runSummary !== 'loading' && runSummary !== 'error' && (
          <>
            <table className="region-panel-table run-confirm-table">
              <tbody>
                <tr><th>Study</th><td>{runSummary.study_id}</td></tr>
                <tr><th>Years</th><td>{runSummary.years.join(', ') || '-'}</td></tr>
                <tr><th>Timepoints</th><td>{runSummary.timepoints.join(', ') || '-'}</td></tr>
                <tr><th>Simulated regions</th><td>{runSummary.n_regions}</td></tr>
                <tr><th>Aggregation levels</th><td>{runSummary.aggregation_levels.join(', ') || 'none'}</td></tr>
                <tr>
                  <th>Cropmask</th>
                  <td>
                    {Object.keys(runSummary.cropmasks).length
                      ? [...new Set(Object.values(runSummary.cropmasks))].join(', ')
                      : '-'}
                  </td>
                </tr>
                <tr><th>LAI archive</th><td>{runSummary.lai_region ?? '-'}{runSummary.lai_resolution ? ` at ${runSummary.lai_resolution} m` : ''}</td></tr>
                <tr><th>Met / precipitation</th><td>{[runSummary.met_source, runSummary.precipitation_source].filter(Boolean).join(' / ') || '-'}</td></tr>
              </tbody>
            </table>
            <p className="run-confirm-note">
              {runConfirm?.force ? (
                <>
                  Every step will be recomputed from scratch, including steps whose inputs have not
                  changed. This takes as long as a first run.
                </>
              ) : (
                <>
                  Steps whose inputs are unchanged reuse the results already on disk; a changed input
                  causes that step and everything downstream of it to be recomputed.
                </>
              )}{' '}
              Output is written to the study's live directory, overwriting the previous run's files
              there. Runs already archived are not affected.
            </p>
            <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end' }}>
              <button className="btn btn-secondary" onClick={() => setRunConfirm(null)}>Cancel</button>
              <button
                className={`btn ${runConfirm?.force ? 'btn-danger' : 'btn-success'}`}
                onClick={() => runConfirm && runStudy(runConfirm.id, runConfirm.force)}
              >{runConfirm?.force ? 'Force full rerun' : 'Start run'}</button>
            </div>
          </>
        )}
      </Modal>

      {/* Create Study */}
      <Modal open={createOpen} onClose={() => setCreateOpen(false)} title="Create New Study">
        <div className="form-group">
          <label htmlFor="studyName" className="form-label">Study Name</label>
          <input id="studyName" className="form-input" placeholder="Enter study name" value={createName} onChange={e => setCreateName(e.target.value)} />
        </div>
        <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end' }}>
          <button className="btn btn-secondary" onClick={() => setCreateOpen(false)}>Cancel</button>
          <button className="btn btn-primary" onClick={createStudy}>Create Study</button>
        </div>
      </Modal>

      {/* Detail / Setup */}
      <Modal open={detailOpen} onClose={() => setDetailOpen(false)} title={`${detailStudy ?? ''} - Configuration`} width={800}>
        <Stepper step={step} onStepChange={setStep} />
        {step === 2 && <SetupStudyForm studyId={detailStudy ?? ''} onSubmit={handleSetupSubmission} initialData={initialSetupData} />}

        {step === 3 && <RunParamsForm
          key={detailStudy ?? 'no-study'}
          onSubmit={handleRunParamsSubmission}
          onDownloadTemplate={handleDownloadTemplate}
          runConfigMessage={runConfigMessage}
          currentStudyId={detailStudy}
          initialData={initialRunConfigData}
        />}

        {step === 4 && (
          <div>
            <div className="alert alert-success">
              <strong>Setup Complete!</strong> Your study is ready to run.
            </div>
            <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'space-between' }}>
              <button className="btn btn-secondary" onClick={async () => {
                if (!detailStudy) return;
                try {
                  const blob = await StudiesAPI.downloadRunConfigTemplate(detailStudy);
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url; a.download = 'config.yaml'; a.click(); URL.revokeObjectURL(url);
                } catch { show('Download failed', 'error'); }
              }}>Download Config</button>
              <div style={{ display: 'flex', gap: '0.75rem' }}>
                <button className="btn btn-secondary" onClick={() => setStep(3)}>Update Config</button>
                <button className="btn btn-success" onClick={() => detailStudy && runStudy(detailStudy, false)}>Run Study</button>
              </div>
            </div>
          </div>
        )}
      </Modal>

      {/* Results year/timepoint selector (per-run) */}
      <Modal open={resultsSelectorOpen} onClose={() => setResultsSelectorOpen(false)} title="Select Result Timepoint">
        {selectorContext && (
          <div style={{ fontSize: 13, color: 'var(--neutral-600)', marginBottom: 12 }}>
            Run <code>{selectorContext.runId}</code> · study <strong>{selectorContext.studyId}</strong>
          </div>
        )}
        <div className="form-group">
          <label className="form-label">Year</label>
          <select value={selectedYear} className="form-input" onChange={(e) => onYearChange(e.target.value)}>
            {yearOptions.map(y => <option key={y} value={y}>{y}</option>)}
          </select>
        </div>
        <div className="form-group">
          <label className="form-label">Timepoint</label>
          <select value={selectedTimepoint} className="form-input" onChange={(e) => setSelectedTimepoint(e.target.value)}>
            {timepointOptions.map(tp => <option key={tp} value={tp}>{tp}</option>)}
          </select>
        </div>
        <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end' }}>
          <button className="btn btn-secondary" onClick={() => setResultsSelectorOpen(false)}>Cancel</button>
          <button className="btn btn-primary" onClick={openSelectedResult} disabled={!selectedYear || !selectedTimepoint}>Open</button>
        </div>
      </Modal>

      {/* Logs */}
      <Modal open={logsOpen} onClose={() => setLogsOpen(false)} title="Study Logs" width={1300}>
        <button onClick={handleCopyLogs} className="btn btn-sm btn-secondary">Copy logs</button>
        <button
          onClick={() => { if (detailStudy) downloadFullLog(detailStudy); }}
          className="btn btn-sm btn-secondary"
          style={{ marginLeft: '10px' }}
        >Download Full Log File</button>
        <div className="logs-container" style={{ marginTop: '2rem' }}>
          <pre dangerouslySetInnerHTML={{ __html: ansiToHtml(logs || 'No logs available') }} />
        </div>
      </Modal>

      {/* Duplicate Study */}
      <Modal open={duplicateOpen} onClose={() => setDuplicateOpen(false)} title="Create New Study from Existing">
        <p className="subtitle" style={{ marginTop: 4 }}>
          Setup a new study based on the configuration of an existing one. Will use the same regions, APSIM files and reference data.
        </p>
        <div className="form-group">
          <label htmlFor="studyName" className="form-label">Study Name</label>
          <input id="studyName" className="form-input" placeholder="Enter study name" value={createName} onChange={e => setCreateName(e.target.value)} />
        </div>
        <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end' }}>
          <button className="btn btn-secondary" onClick={() => setDuplicateOpen(false)}>Cancel</button>
          <button className="btn btn-primary" onClick={() => { duplicateStudy(); setDuplicateOpen(false); }}>Create Study</button>
        </div>
      </Modal>

      {/* Delete Study */}
      <Modal open={deleteOpen} onClose={() => setDeleteOpen(false)} title="Delete Study">
        <p className="subtitle" style={{ marginTop: 4 }}>
          This will permanently remove the study folder and all of its archived runs. This cannot be undone.
        </p>
        <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end', marginTop: 12 }}>
          <button className="btn btn-secondary" onClick={() => setDeleteOpen(false)}>Cancel</button>
          <button className="btn btn-danger" onClick={confirmDelete}>Confirm Delete</button>
        </div>
      </Modal>

      {/* Delete Run */}
      <Modal open={!!deleteRunCtx} onClose={() => setDeleteRunCtx(null)} title="Delete Archived Run">
        <p className="subtitle" style={{ marginTop: 4 }}>
          {deleteRunCtx && <>This will permanently remove the run <code>{deleteRunCtx.runId}</code> and all of its archived results.</>}
        </p>
        <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end', marginTop: 12 }}>
          <button className="btn btn-secondary" onClick={() => setDeleteRunCtx(null)}>Cancel</button>
          <button className="btn btn-danger" onClick={confirmDeleteRun}>Confirm Delete</button>
        </div>
      </Modal>

      <Toast />
      {isLoading && (
        <div
          role="alert"
          aria-live="polite"
          style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.35)', zIndex: 9999, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16 }}
        >
          <div style={{ background: '#fff', borderRadius: 16, padding: '18px 22px', boxShadow: '0 10px 30px rgba(0,0,0,0.2)', display: 'flex', alignItems: 'center', gap: 12, minWidth: 220, justifyContent: 'center' }}>
            <div aria-hidden style={{ width: 28, height: 28, borderRadius: '50%', border: '3px solid #E5E7EB', borderTopColor: '#3B82F6', animation: 'spin 1s linear infinite' }} />
            <span style={{ fontSize: 14, color: '#374151' }}>{loadingText || 'Working…'}</span>
          </div>
        </div>
      )}
      <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
    </div>
  );
};

export default StudiesPage;
