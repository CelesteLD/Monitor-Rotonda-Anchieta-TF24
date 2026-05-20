// ── Paleta de colores ─────────────────────────────────────────
export const PALETTE = {
  pink:    '#ef476f',
  yellow:  '#ffd166',
  emerald: '#06d6a0',
  ocean:   '#118ab2',
  teal:    '#073b4c',

  pinkSoft:    'rgba(239,71,111,0.10)',
  yellowSoft:  'rgba(255,209,102,0.12)',
  emeraldSoft: 'rgba(6,214,160,0.10)',
  oceanSoft:   'rgba(17,138,178,0.10)',
  tealSoft:    'rgba(7,59,76,0.10)',
}

// ── Tokens por tema ───────────────────────────────────────────
export const T = {
  light: {
    bg:           '#f0f3f9',
    bgAlt:        '#e6eaf4',
    surface:      '#ffffff',
    surfaceHover: '#f8f9fd',
    border:       '#dde3f0',
    borderMid:    '#c8d0e4',
    text:         '#0a0e1a',
    textSub:      '#39393a',
    textMuted:    '#7a8aaa',
    shadow:       '0 1px 3px rgba(10,14,26,0.06), 0 4px 16px rgba(10,14,26,0.04)',
    shadowMd:     '0 4px 24px rgba(10,14,26,0.10)',
    chartGrid:    '#eaeef8',
    chartText:    '#7a8aaa',
    dateFilter:   'none',
  },
  dark: {
    bg:           '#070a11',
    bgAlt:        '#0c1020',
    surface:      '#111520',
    surfaceHover: '#161b2c',
    border:       '#1e2438',
    borderMid:    '#2a3050',
    text:         '#e8edf8',   
    textSub:      '#c3c7d1',   
    textMuted:    '#c9c9c9',   
    shadow:       '0 1px 3px rgba(0,0,0,0.3), 0 4px 16px rgba(0,0,0,0.2)',
    shadowMd:     '0 4px 24px rgba(0,0,0,0.4)',
    chartGrid:    '#151a2a',
    chartText:    '#4a5578',
    dateFilter:   'invert(0.6)',
  },
}

// ── Estado de tráfico ─────────────────────────────────────────
export const STATE = {
  FLUIDO: {
    color:  PALETTE.emerald,
    soft:   PALETTE.emeraldSoft,
    border: PALETTE.emerald,
    label:  'Fluido',
  },
  DENSO: {
    color:  PALETTE.yellow,
    soft:   PALETTE.yellowSoft,
    border: PALETTE.yellow,
    label:  'Denso',
  },
  COLAPSO: {
    color:  PALETTE.pink,
    soft:   PALETTE.pinkSoft,
    border: PALETTE.pink,
    label:  'Colapso',
  },
}

// ── Tendencia Spark ───────────────────────────────────────────
export const TENDENCIA = {
  SUBIENDO: { color: PALETTE.pink,    label: 'Subiendo' },
  BAJANDO:  { color: PALETTE.emerald, label: 'Bajando'  },
  ESTABLE:  { color: PALETTE.ocean,   label: 'Estable'  },
}

// ── ROI chips ─────────────────────────────────────────────────
export const ROI_STYLE = {
  tramo1_rotonda: { bg: PALETTE.pinkSoft,    color: PALETTE.pink,    label: 'Tramo 1' },
  tramo2_rotonda: { bg: PALETTE.yellowSoft,  color: '#9a7000',       label: 'Tramo 2' },
  tramo3_rotonda: { bg: PALETTE.emeraldSoft, color: PALETTE.emerald, label: 'Tramo 3' },
}

// ── Score de congestión por hora (calculado sobre el dataset) ─
export const CONGESTION_SCORE_BY_HOUR = {
  8:  0.203, 9:  0.163, 10: 0.238,
  11: 0.322, 12: 0.437, 13: 0.460,
  14: 0.365, 15: 0.433, 16: 0.420,
  17: 0.219, 18: 0.098, 19: 0.056,
}
export const PEAK_HOUR_START = 11
export const PEAK_HOUR_END   = 16

// ── Helper: card base style ───────────────────────────────────
export const cardStyle = (t, extra = {}) => ({
  background:   t.surface,
  border:       `1px solid ${t.border}`,
  borderRadius: 16,
  boxShadow:    t.shadow,
  ...extra,
})
