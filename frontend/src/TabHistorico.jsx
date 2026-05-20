import { useState } from 'react'
import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, ReferenceLine, Legend,
} from 'recharts'
import { PALETTE, cardStyle } from './tokens'
import { KPI, CustomTooltip } from './shared'

const API = `${import.meta.env.VITE_API_URL || ''}/api`

export default function TabHistorico({ t }) {
  const today = new Date().toISOString().split('T')[0]
  const [desde,   setDesde]   = useState(today)
  const [hasta,   setHasta]   = useState(today)
  const [data,    setData]    = useState([])
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)
  const [queried, setQueried] = useState(false)

  const fetchRange = async () => {
    if (!desde || !hasta) return
    setLoading(true); setError(null)
    try {
      const r = await fetch(`${API}/history/range?desde=${desde}&hasta=${hasta}`)
      if (!r.ok) throw new Error('Error al consultar el servidor')
      const rows = await r.json()
      setData(rows.map(row => ({
        ...row,
        time: new Date(row.timestamp).toLocaleString('es-ES', {
          month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
        }),
        vehicles: row.vehicle_count,
      })))
      setQueried(true)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const rangeStats = (() => {
    if (!data.length) return null
    const c = { FLUIDO: 0, DENSO: 0, COLAPSO: 0 }
    let sum = 0, max = 0
    data.forEach(r => {
      c[r.estado] = (c[r.estado] || 0) + 1
      sum += r.vehicles
      if (r.vehicles > max) max = r.vehicles
    })
    return { total: data.length, ...c, avg: (sum / data.length).toFixed(1), max }
  })()

  const byHour = (() => {
    if (!data.length) return []
    const map = {}
    data.forEach(r => {
      const h = new Date(r.timestamp).getHours()
      if (!map[h]) map[h] = { hour: `${String(h).padStart(2, '0')}h`, FLUIDO: 0, DENSO: 0, COLAPSO: 0, total: 0, sumV: 0 }
      map[h][r.estado]++; map[h].total++; map[h].sumV += r.vehicles
    })
    return Object.values(map)
      .sort((a, b) => parseInt(a.hour) - parseInt(b.hour))
      .map(h => ({ ...h, avg: (h.sumV / h.total).toFixed(1) }))
  })()

  const inputStyle = {
    background: t.surface, border: `1px solid ${t.borderMid}`,
    borderRadius: 10, padding: '9px 13px', color: t.text,
    fontSize: 13, outline: 'none', fontFamily: 'var(--font-body)',
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* Selector de rango */}
      <div style={{ ...cardStyle(t), padding: '20px 24px' }}>
        <div style={{
          fontFamily: 'var(--font-display)', fontSize: 14,
          fontWeight: 500, color: t.text, marginBottom: 16,
        }}>
          Consultar rango de fechas
        </div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'flex-end', flexWrap: 'wrap' }}>
          {[
            ['Desde', desde, setDesde, null,  today],
            ['Hasta', hasta, setHasta, desde, today],
          ].map(([label, val, setter, min, max]) => (
            <div key={label} style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
              <label style={{
                color: t.textMuted, fontSize: 10, fontWeight: 500,
                letterSpacing: '0.06em', textTransform: 'uppercase',
              }}>
                {label}
              </label>
              <input
                type="date" value={val} min={min} max={max}
                onChange={e => setter(e.target.value)}
                style={inputStyle}
              />
            </div>
          ))}
          <button onClick={fetchRange} disabled={loading} style={{
            background: PALETTE.ocean, border: 'none', borderRadius: 10,
            padding: '9px 22px', color: '#fff', fontWeight: 500, fontSize: 13,
            cursor: loading ? 'not-allowed' : 'pointer',
            fontFamily: 'var(--font-display)',
            opacity: loading ? 0.65 : 1,
          }}>
            {loading ? 'Consultando...' : 'Consultar'}
          </button>
        </div>
        {error && (
          <div style={{ color: PALETTE.pink, fontSize: 13, marginTop: 12 }}>
            ⚠ {error}
          </div>
        )}
      </div>

      {queried && !loading && (
        data.length === 0
          ? (
            <div style={{ ...cardStyle(t), padding: 48, textAlign: 'center', color: t.textMuted, fontSize: 14 }}>
              Sin datos para el rango seleccionado.
            </div>
          ) : (
            <>
              {/* KPIs de rango */}
              <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                <KPI label="Registros" value={rangeStats.total}          t={t} delay={1} />
                <KPI label="Media"     value={rangeStats.avg}            t={t} delay={2} />
                <KPI label="Máximo"    value={rangeStats.max}            color={PALETTE.yellow} t={t} delay={3} />
                <KPI label="Fluido"    value={rangeStats.FLUIDO}         color={PALETTE.emerald} t={t} delay={1} />
                <KPI label="Denso"     value={rangeStats.DENSO}          color={PALETTE.yellow}  t={t} delay={2} />
                <KPI label="Colapso"   value={rangeStats.COLAPSO || 0}  color={PALETTE.pink}    t={t} delay={3} />
              </div>

              {/* Evolución temporal */}
              <div style={{ ...cardStyle(t), padding: '20px 24px' }}>
                <div style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
                  marginBottom: 18, flexWrap: 'wrap', gap: 8,
                }}>
                  <span style={{ fontFamily: 'var(--font-display)', fontSize: 14, fontWeight: 500, color: t.textSub }}>
                    Evolución del número de vehículos — {desde === hasta ? desde : `${desde} → ${hasta}`}
                  </span>
                  <span style={{ fontSize: 12, color: t.textMuted }}>{data.length} registros</span>
                </div>
                <ResponsiveContainer width="100%" height={210}>
                  <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
                    <CartesianGrid stroke={t.chartGrid} strokeDasharray="3 3" vertical={false} />
                    <XAxis
                      dataKey="time"
                      tick={{ fill: t.chartText, fontSize: 9 }}
                      interval={Math.max(0, Math.floor(data.length / 14) - 1)}
                      axisLine={false} tickLine={false}
                    />
                    <YAxis
                      domain={[0, 'auto']}
                      tick={{ fill: t.chartText, fontSize: 10 }}
                      width={24} axisLine={false} tickLine={false}
                    />
                    <Tooltip content={(props) => <CustomTooltip {...props} t={t} />} />
                    <ReferenceLine y={5}  stroke={PALETTE.yellow} strokeDasharray="4 3" strokeOpacity={0.5} />
                    <ReferenceLine y={10} stroke={PALETTE.pink}   strokeDasharray="4 3" strokeOpacity={0.5} />
                    <Line
                      type="monotone" dataKey="vehicles"
                      stroke={PALETTE.ocean} strokeWidth={1.5} dot={false}
                      activeDot={{ r: 3, fill: PALETTE.ocean, strokeWidth: 0 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Distribución por hora */}
              <div style={{ ...cardStyle(t), padding: '20px 24px' }}>
                <div style={{
                  fontFamily: 'var(--font-display)', fontSize: 14,
                  fontWeight: 500, color: t.textSub, marginBottom: 18,
                }}>
                  Distribución por hora del estado del tráfico
                </div>
                <ResponsiveContainer width="100%" height={210}>
                  <BarChart data={byHour} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
                    <CartesianGrid stroke={t.chartGrid} strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="hour" tick={{ fill: t.chartText, fontSize: 10 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: t.chartText, fontSize: 10 }} width={24} axisLine={false} tickLine={false} />
                    <Tooltip
                      contentStyle={{
                        background: t.surface, border: `1px solid ${t.borderMid}`,
                        borderRadius: 10, fontSize: 12, boxShadow: t.shadowMd,
                      }}
                      labelStyle={{ color: t.text, fontWeight: 600 }}
                    />
                    <Legend wrapperStyle={{ fontSize: 12, color: t.textSub, paddingTop: 12 }} />
                    <Bar dataKey="FLUIDO"  stackId="a" fill={PALETTE.emerald} />
                    <Bar dataKey="DENSO"   stackId="a" fill={PALETTE.yellow}  />
                    <Bar dataKey="COLAPSO" stackId="a" fill={PALETTE.pink} radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          )
      )}
    </div>
  )
}
