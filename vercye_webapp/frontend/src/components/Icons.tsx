/**
 * Icon set, drawn to match RAAPID-ORG/stacnotator's `shared/ui/Icons.tsx`:
 * viewBox "0 0 20 20", stroke="currentColor", strokeWidth 1.5, round caps and joins,
 * no fill. The lighter 1.5 stroke on a 20-unit grid is what makes them read as quiet
 * UI furniture rather than as illustrations - a 2px stroke on a 24-unit grid, which is
 * what these were before, is noticeably heavier next to the same text.
 *
 * Paths are taken from that file where an equivalent icon exists, so the two apps'
 * chrome is literally the same shape.
 */
type IconProps = { size?: number; className?: string };

const Svg = ({ size = 16, className, children }: IconProps & { children: React.ReactNode }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width={size}
    height={size}
    viewBox="0 0 20 20"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.5}
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
    aria-hidden="true"
    focusable="false"
  >
    {children}
  </svg>
);

export const PlusIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M10 4v12M4 10h12" />
  </Svg>
);

export const GearIcon = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="10" cy="10" r="2.5" />
    <path d="M10 2.5v2M10 15.5v2M17.5 10h-2M4.5 10h-2M15.3 4.7l-1.4 1.4M6.1 13.9l-1.4 1.4M15.3 15.3l-1.4-1.4M6.1 6.1L4.7 4.7" />
  </Svg>
);

export const PlayIcon = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="10" cy="10" r="8" />
    <path d="M8 6.5l6 3.5-6 3.5V6.5z" />
  </Svg>
);

export const StopIcon = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="10" cy="10" r="8" />
    <rect x="7.5" y="7.5" width="5" height="5" rx="1" />
  </Svg>
);

export const LogsIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M5 3h7l4 4v10a1 1 0 01-1 1H5a1 1 0 01-1-1V4a1 1 0 011-1z" />
    <path d="M12 3v4h4M7 11h6M7 14h4" />
  </Svg>
);
export const ReportIcon = LogsIcon;

export const CopyIcon = (p: IconProps) => (
  <Svg {...p}>
    <rect x="7" y="7" width="9" height="9" rx="1.5" />
    <path d="M13 7V5.5A1.5 1.5 0 0011.5 4h-6A1.5 1.5 0 004 5.5v6A1.5 1.5 0 005.5 13H7" />
  </Svg>
);

export const TrashIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M4 5h12M7 5V4a1 1 0 011-1h4a1 1 0 011 1v1M8 8v6M12 8v6M5 5l1 11a1 1 0 001 1h6a1 1 0 001-1l1-11" />
  </Svg>
);

export const MapIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M7 3L2 5.5v12L7 15l6 2.5 5-2.5v-12L13 5 7 3z" />
    <path d="M7 3v12M13 5v12" />
  </Svg>
);

export const ChartIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M3 17V9M8 17V4M13 17v-5M18 17V7" />
  </Svg>
);

export const DownloadIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M10 3v9M6.5 8.5L10 12l3.5-3.5M4 15h12" />
  </Svg>
);

export const CloseIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M5 5l10 10M15 5L5 15" />
  </Svg>
);

export const BackIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M12.5 15l-5-5 5-5" />
  </Svg>
);

export const ChevronRightIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M7.5 5l5 5-5 5" />
  </Svg>
);

export const ChevronDownIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M5 7.5l5 5 5-5" />
  </Svg>
);

export const ClockIcon = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="10" cy="10" r="8" />
    <path d="M10 5v5l3 3" />
  </Svg>
);

export const RefreshIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M16.5 8A6.5 6.5 0 005 5.5M3.5 12A6.5 6.5 0 0015 14.5" />
    <path d="M16.5 4v4h-4M3.5 16v-4h4" />
  </Svg>
);
