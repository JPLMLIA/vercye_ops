
import type { StudyStatus } from '@/types';

const StatusBadge = ({ status }: { status: StudyStatus }) => {
  const map: Record<StudyStatus, { className: string; text: string }> = {
    pending: { className: 'status-pending', text: '-' },
    queued: { className: 'status-ready', text: 'Queued' },
    running: { className: 'status-running', text: 'Running' },
    completed: { className: 'status-completed', text: 'Completed' },
    failed: { className: 'status-failed', text: 'Failed' },
    cancelled: { className: 'status-cancelled', text: 'Cancelled' },
    cancelling: { className: 'status-running', text: 'Cancelling' },
    validating: { className: 'status-running', text: 'Validating' },
    ready: { className: 'status-ready', text: 'Ready' },
  };
  const cfg = map[status] ?? map.pending;
  return <span className={`status-badge ${cfg.className}`}>{cfg.text}</span>;
}

export default StatusBadge
