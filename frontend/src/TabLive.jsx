import { Camera, Cpu, AlertTriangle } from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, ReferenceLine, CartesianGrid,
} from 'recharts'
import { STATE, ROI_STYLE, PALETTE, cardStyle, CONGESTION_SCORE_BY_HOUR } from './tokens'
import { StateBadge, KPI, RoiChips, Sparkline, CustomTooltip } from './shared'

import { useState, useEffect, useRef } from 'react'

// ── Panel derecho: estado + zonas + sistema ───────────────────
function SidePanel({ status, t }) {
  const s        = STATE[status?.estado] || STATE.FLUIDO
  const hour     = status?.hour ?? new Date().getHours()
  const score    = CONGESTION_SCORE_BY_HOUR[hour] ?? 0.15
  const isPeak   = hour >= 11 && hour <= 16

  const [countdown, setCountdown] = useState(null)
  const intervalRef = useRef(null)

  const FETCH_INTERVAL_S = 45  // igual que en el backend

  useEffect(() => {
    if (!status?.timestamp) return
    setCountdown(FETCH_INTERVAL_S)

    if (intervalRef.current) clearInterval(intervalRef.current)
    intervalRef.current = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) { clearInterval(intervalRef.current); return 0 }
        return prev - 1
      })
    }, 1000)

    return () => clearInterval(intervalRef.current)
  }, [status?.timestamp])  // se dispara solo cuando llega un frame nuevo real

  return (
    <div style={{
      ...cardStyle(t),
      border: `1px solid ${s.border}40`,
      padding: 20, display: 'flex', flexDirection: 'column', gap: 20,
    }}>

      {/* Estado actual */}
      <div style={{ textAlign: 'center' }}>
        <div style={{
          fontSize: 9, color: t.textMuted, textTransform: 'uppercase',
          letterSpacing: '0.08em', marginBottom: 10,
        }}>
          Estado actual
        </div>
        <StateBadge estado={status?.estado || 'FLUIDO'} />

        {/* Barra de confianza */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          gap: 7, marginTop: 10, fontSize: 11, color: t.textMuted,
        }}>
          <span>Confianza</span>
          <div style={{ width: 70, height: 4, background: t.border, borderRadius: 2, overflow: 'hidden' }}>
            <div style={{
              height: '100%', borderRadius: 2, background: s.color,
              width: `${Math.round((status?.confianza || 0) * 100)}%`,
              transition: 'width 0.4s ease',
            }} />
          </div>
          <span>{Math.round((status?.confianza || 0) * 100)}%</span>
        </div>

        {/* Countdown próximo análisis */}
        {countdown != null && (
          <div style={{ marginTop: 8 }}>
            <span style={{
              display: 'inline-flex', alignItems: 'center', gap: 4,
              fontSize: 10, background: `${PALETTE.ocean}18`,
              color: PALETTE.ocean, padding: '3px 9px', borderRadius: 20,
            }}>
              ⏱ Próximo análisis en {countdown}s
            </span>
          </div>
        )}
      </div>

      {/* Por zona */}
      {status?.roi_counts && (
        <div>
          <div style={{
            fontSize: 10, color: t.textMuted, textTransform: 'uppercase',
            letterSpacing: '0.06em', marginBottom: 8,
          }}>
            Vehículos por zona
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {Object.entries(status.roi_counts).map(([key, count]) => {
              const rs = ROI_STYLE[key] || { bg: PALETTE.oceanSoft, color: PALETTE.ocean, label: key }
              return (
                <div key={key} style={{
                  display: 'flex', alignItems: 'center', gap: 8,
                  padding: '7px 10px', background: t.bgAlt,
                  borderRadius: 8,
                }}>
                  <span style={{ fontSize: 12, color: t.text, flex: 1 }}>{rs.label} Rotonda</span>
                  <span style={{
                    fontSize: 11, fontWeight: 500, padding: '2px 8px',
                    borderRadius: 6, background: rs.bg, color: rs.color,
                  }}>
                    {count}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Sistema */}
      <div>
        <div style={{
          fontSize: 10, color: t.textMuted, textTransform: 'uppercase',
          letterSpacing: '0.06em', marginBottom: 8,
        }}>
          Sistema
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
          {[
            ['Clasificador', status?.metodo === 'randomforest' ? 'RandomForest'
              : status?.metodo === 'mllib' ? 'Spark MLlib' : 'Umbral provisional'],
            ['Aceleración', (
              <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 12, color: t.textMuted }}>
                <Cpu size={11} />
                {status?.cpp_threads
                  ? `C++ / OpenMP · ${status.cpp_threads} threads`
                  : status?.cuda_used ? 'CUDA' : 'CPU'
                }
              </span>
            )],
            ['Latencia C++', (
              <span style={{ fontSize: 12, fontWeight: 500, color: PALETTE.ocean }}>
                {status?.cpp_elapsed_ms != null ? `${status.cpp_elapsed_ms} ms` : '—'}
              </span>
            )],
            ['Hora punta', (
              <span style={{ fontSize: 12, fontWeight: 500, color: isPeak ? PALETTE.yellow : PALETTE.emerald }}>
                {isPeak ? `Sí · ${hour}h` : 'No'}
              </span>
            )],
            ['Score congestión', (
              <span style={{ fontSize: 12, fontWeight: 500, color: PALETTE.ocean }}>
                {(score * 100).toFixed(1)}%
              </span>
            )],
          ].map(([label, value], i, arr) => (
            <div key={label} style={{
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              padding: '9px 0',
              borderBottom: i < arr.length - 1 ? `1px solid ${t.border}` : 'none',
            }}>
              <span style={{ color: t.textMuted, fontSize: 12 }}>{label}</span>
              {typeof value === 'string'
                ? <span style={{ color: t.text, fontSize: 12, fontWeight: 500 }}>{value}</span>
                : value
              }
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Gráfico últimas 2 horas ───────────────────────────────────
function RecentChart({ history, t }) {
  const vals          = history.map(r => r.vehicles).filter(v => v != null)
  const sorted        = [...vals].sort((a, b) => a - b)
  const enough        = vals.length >= 10
  const umbralDenso   = enough ? sorted[Math.floor(sorted.length * 0.65)] : 5
  const umbralColapso = enough ? sorted[Math.floor(sorted.length * 0.85)] : 10

  return (
    <div className="fade-up fade-up-4" style={{ ...cardStyle(t), padding: '20px 24px' }}>
      <div style={{
        display: 'flex', justifyContent: 'space-between',
        alignItems: 'baseline', marginBottom: 18, flexWrap: 'wrap', gap: 8,
      }}>
        <span style={{
          fontFamily: 'var(--font-display)', fontSize: 14,
          fontWeight: 500, color: t.textSub,
        }}>
          Últimas 2 horas
        </span>
        <div style={{ display: 'flex', gap: 14 }}>
          <span style={{ color: PALETTE.yellow, fontSize: 11 }}>
            — denso {enough ? `p65 (${umbralDenso})` : `fijo (${umbralDenso})`}
          </span>
          <span style={{ color: PALETTE.pink, fontSize: 11 }}>
            — colapso {enough ? `p85 (${umbralColapso})` : `fijo (${umbralColapso})`}
          </span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={190}>
        <LineChart data={history} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
          <CartesianGrid stroke={t.chartGrid} strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="time" tick={{ fill: t.chartText, fontSize: 10 }} interval="preserveStartEnd" axisLine={false} tickLine={false} />
          <YAxis tick={{ fill: t.chartText, fontSize: 10 }} width={24} axisLine={false} tickLine={false} />
          <Tooltip content={(props) => <CustomTooltip {...props} t={t} />} />
          <ReferenceLine y={umbralDenso}   stroke={PALETTE.yellow} strokeDasharray="4 3" strokeOpacity={0.6} />
          <ReferenceLine y={umbralColapso} stroke={PALETTE.pink}   strokeDasharray="4 3" strokeOpacity={0.6} />
          <Line
            type="monotone" dataKey="vehicles"
            stroke={PALETTE.ocean} strokeWidth={2} dot={false}
            activeDot={{ r: 4, fill: PALETTE.ocean, strokeWidth: 0 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── TabLive ───────────────────────────────────────────────────
export default function TabLive({ status, history, stats, frameUrl, error, t }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

      {error && (
        <div style={{
          background: `${PALETTE.pink}14`, border: `1px solid ${PALETTE.pink}`,
          borderRadius: 12, padding: '12px 16px', color: PALETTE.pink,
          display: 'flex', alignItems: 'center', gap: 8, fontSize: 13,
        }}>
          <AlertTriangle size={15} /> {error}
        </div>
      )}

      {/* Fila principal: frame + (panel + KPIs) */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'minmax(0,1fr) minmax(260px,300px)',
        gap: 20, alignItems: 'start',
      }} className="main-grid">

        {/* Izquierda: solo el frame */}
        <div className="fade-up" style={{ ...cardStyle(t), overflow: 'hidden' }}>
          <div style={{
            padding: '10px 16px', borderBottom: `1px solid ${t.border}`,
            display: 'flex', alignItems: 'center', gap: 7,
            color: t.textMuted, fontSize: 11, fontWeight: 500,
            textTransform: 'uppercase', letterSpacing: '0.05em',
          }}>
            <Camera size={12} strokeWidth={2} />
            Imagen en directo — detección YOLO activa
          </div>
          <div style={{ position: 'relative', height: 705 }}>
            {frameUrl
              ? <img src={frameUrl} alt="Cámara rotonda"
                  style={{ width: '100%', height: '100%', display: 'block', objectFit: 'cover' }} />
              : <div style={{
                  height: '100%', display: 'flex', alignItems: 'center',
                  justifyContent: 'center', color: t.textMuted, fontSize: 13,
                  background: t.bgAlt,
                }}>
                  Cargando imagen...
                </div>
            }
            {status?.vehicle_count != null && (
              <div style={{
                position: 'absolute', top: 10, left: 10,
                background: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(6px)',
                borderRadius: 10, padding: '6px 12px',
                border: '1px solid rgba(255,255,255,0.12)',
              }}>
                <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.55)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
                  Total vehículos
                </div>
                <div style={{ fontSize: 26, fontWeight: 700, color: '#fff', lineHeight: 1.1, fontFamily: 'var(--font-display)' }}>
                  {status.vehicle_count}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Derecha: SidePanel + KPIs 2x2 */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div className="fade-up fade-up-1">
            <SidePanel status={status} t={t} />
          </div>
          {/* KPIs 2x2 */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <KPI label="Detecciones hoy" value={stats?.total}            t={t} delay={1} />
            <KPI label="Media vehículos" value={stats?.avg_vehicles != null ? parseFloat(stats.avg_vehicles).toFixed(1) : null} t={t} delay={2} />
            <KPI label="Máximo del día"  value={stats?.max_vehicles}     color={PALETTE.yellow} t={t} delay={3} />
            <KPI label="Colapsos hoy"    value={stats?.colapsos ?? 0}    color={PALETTE.pink}   t={t} delay={4} />
          </div>
        </div>
      </div>

      {/* Actividad última hora — ancho completo */}
      <Sparkline history={history} t={t} />

    </div>
  )
}