import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { STATE, TENDENCIA, ROI_STYLE, cardStyle, PALETTE } from './tokens'

// ── StateBadge ────────────────────────────────────────────────
export function StateBadge({ estado, size = 'md' }) {
  const s    = STATE[estado] || STATE.FLUIDO
  const pad  = size === 'sm' ? '4px 12px' : '8px 20px'
  const fs   = size === 'sm' ? 12 : 15

  return (
    <div style={{
      display: 'inline-flex', alignItems: 'center', gap: 7,
      padding: pad, borderRadius: 999,
      background: s.soft, border: `1.5px solid ${s.border}`,
      color: s.color, fontWeight: 500, fontSize: fs,
      fontFamily: 'var(--font-display)',
    }}>
      <span style={{ width: 7, height: 7, borderRadius: '50%', background: s.color, flexShrink: 0 }} />
      {s.label}
    </div>
  )
}

// ── KPI card ──────────────────────────────────────────────────
export function KPI({ label, value, color, t, delay = 0 }) {
  return (
    <div className={`fade-up fade-up-${delay}`} style={{
      ...cardStyle(t), padding: '14px 16px', flex: '1 1 110px', minWidth: 0,
    }}>
      <div style={{
        color: t.textMuted, fontSize: 10, fontWeight: 500,
        letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 5,
      }}>
        {label}
      </div>
      <div style={{
        fontSize: 26, fontWeight: 500, lineHeight: 1,
        color: color || t.text, fontFamily: 'var(--font-display)',
      }}>
        {value ?? '—'}
      </div>
    </div>
  )
}

// ── ROI chips ─────────────────────────────────────────────────
export function RoiChips({ roiCounts }) {
  if (!roiCounts) return null
  return (
    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
      {Object.entries(roiCounts).map(([key, count]) => {
        const s = ROI_STYLE[key] || { bg: PALETTE.oceanSoft, color: PALETTE.ocean, label: key }
        return (
          <span key={key} style={{
            display: 'inline-flex', alignItems: 'center', gap: 4,
            fontSize: 11, padding: '3px 9px', borderRadius: 20,
            background: s.bg, color: s.color, fontWeight: 500,
          }}>
            {s.label} · {count}
          </span>
        )
      })}
    </div>
  )
}

export function Sparkline({ history, t }) {
  if (!history?.length) return null

  const max = Math.max(...history.map(r => r.vehicles || 0), 1)

  const barColor = (estado) => {
    if (estado === 'COLAPSO') return PALETTE.pink
    if (estado === 'DENSO')   return PALETTE.yellow
    return PALETTE.emerald
  }

  return (
    <div style={{
      padding: '12px 14px',
      background: t.surface,
      border: `1px solid ${t.border}`,
      borderRadius: 10,
      boxShadow: t.shadow,
    }}>
      <div style={{
        fontSize: 10, color: t.textMuted,
        textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 10,
      }}>
        Actividad última hora
      </div>
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 3, height: 36 }}>
        {history.slice(-20).map((r, i) => (
          <div key={i} style={{
            flex: 1, borderRadius: 2, opacity: 0.75,
            height: `${Math.max(10, ((r.vehicles || 0) / max) * 100)}%`,
            background: barColor(r.estado),
          }} />
        ))}
      </div>
      <div style={{ display: 'flex', gap: 12, marginTop: 8 }}>
        {[
          [PALETTE.emerald, 'Fluido'],
          [PALETTE.yellow,  'Denso'],
          [PALETTE.pink,    'Colapso'],
        ].map(([color, label]) => (
          <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <div style={{ width: 6, height: 6, borderRadius: '50%', background: color }} />
            <span style={{ fontSize: 10, color: t.textMuted }}>{label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Tooltip de gráficas ───────────────────────────────────────
export function CustomTooltip({ active, payload, label, t }) {
  if (!active || !payload?.length) return null
  const d = payload[0]
  const s = STATE[d?.payload?.estado] || {}
  return (
    <div style={{
      background: t.surface, border: `1px solid ${t.borderMid}`,
      borderRadius: 10, padding: '10px 14px', fontSize: 12,
      boxShadow: t.shadowMd, fontFamily: 'var(--font-body)',
    }}>
      <div style={{ color: t.textMuted, marginBottom: 4, fontSize: 11 }}>{label}</div>
      <div style={{ color: s.color || t.text, fontWeight: 600, fontSize: 14 }}>
        {d.value} vehículos
      </div>
      {d?.payload?.estado && (
        <div style={{ color: s.color, fontSize: 11, marginTop: 2, opacity: 0.8 }}>
          {d.payload.estado}
        </div>
      )}
    </div>
  )
}

// ── TendenciaBadge ────────────────────────────────────────────
export function TendenciaBadge({ tendencia }) {
  const cfg   = TENDENCIA[tendencia] || TENDENCIA.ESTABLE
  const icons = { SUBIENDO: TrendingUp, BAJANDO: TrendingDown, ESTABLE: Minus }
  const Icon  = icons[tendencia] || Minus
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 5,
      color: cfg.color, fontSize: 13, fontWeight: 500,
    }}>
      <Icon size={14} strokeWidth={2.5} />
      {cfg.label}
    </span>
  )
}
