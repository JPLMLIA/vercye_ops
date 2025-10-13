export function escapeHtml(s: string) {
  return s.replace(/[&<>"']/g, (ch) =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;', "'":'&#39;'}[ch]!)
  );
}

export function ansiToHtml(input: string) {
  const COLORS: Record<string, string> = {
    '30': '#f7fafc',
    '31': '#e53e3e',
    '32': '#38a169',
    '33': '#d69e2e',
    '34': '#63b3ed',
    '35': '#c084fc',
    '36': '#81e6d9',
    '37': '#ffffff',
    '90': '#cbd5e0',
  };
  const safe = escapeHtml(input);
  return safe.replace(/\x1b\[(\d+)m/g, (_m, code: string) => {
    if (code === '0') return '</span>';
    const color = COLORS[code];
    return color ? `<span style="color:${color}">` : '';
  });
}
