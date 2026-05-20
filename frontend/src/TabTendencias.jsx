import { useState, useEffect, useCallback, useRef } from 'react'
import { Zap, TrendingUp, TrendingDown, Minus, AlertTriangle } from 'lucide-react'
import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, ReferenceLine,
} from 'recharts'
import { STATE, TENDENCIA, PALETTE, cardStyle } from './tokens'
import { StateBadge, TendenciaBadge } from './shared'

const API = `${import.meta.env.VITE_API_URL || ''}/api`

const TREND_ICONS = { SUBIENDO: TrendingUp, BAJANDO: TrendingDown, ESTABLE: Minus }

export default function TabTendencias({ t }) {
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(null)
  const intervalRef           = useRef(null)

  const fetchTendencias = useCallback(async () => {
    try {
      const r = await fetch(`${API}/spark/tendencias?ventanas=20`)
      if (!r.ok) throw new Error('Error al consultar tendencias')
      setData(await r.json())
      setError(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchTendencias()
    intervalRef.current = setInterval(fetchTendencias, 30000)
    return () => clearInterval(intervalRef.current)
  }, [fetchTendencias])

  if (loading) return (
    <div style={{ ...cardStyle(t), padding: 48, textAlign: 'center', color: t.textMuted, fontSize: 14 }}>
      Cargando datos de Spark Streaming...
    </div>
  )

  if (error) return (
    <div style={{ ...cardStyle(t), padding: 32, color: PALETTE.pink, fontSize: 14, display: 'flex', alignItems: 'center', gap: 8 }}>
      <AlertTriangle size={16} /> {error}
    </div>
  )

  if (!data?.ventanas?.length) return (
    <div style={{ ...cardStyle(t), padding: 48, textAlign: 'center', color: t.textMuted, fontSize: 14 }}>
      <Zap size={32} style={{ opacity: 0.3, display: 'block', margin: '0 auto 12px' }} />
      Sin datos de Spark todavía — las ventanas se generan cada 30 segundos.
    </div>
  )

  const ventanas   = data.ventanas
  const ultima     = ventanas[ventanas.length - 1]
  const tendencia  = ultima?.tendencia || 'ESTABLE'
  const tColor     = TENDENCIA[tendencia]?.color || t.textMuted
  const TIcon      = TREND_ICONS[tendencia] || Minus
  const prediccion = data.prediccion_10min
  const sPredict   = STATE[prediccion] || STATE.FLUIDO

  const chartData = ventanas.map(v => ({
    time: new Date(v.timestamp).toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit' }),
    avg:  parseFloat((v.avg_vehicles || 0).toFixed(1)),
    max:  v.max_vehicles,
    estado: v.estado_ventana,
  }))

  const dist = ventanas.reduce((acc, v) => {
    acc[v.estado_ventana] = (acc[v.estado_ventana] || 0) + 1
    return acc
  }, {})
  const pct = (key) => Math.round((dist[key] || 0) / ventanas.length * 100)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* Cabecera — 3 tarjetas */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 16 }} className="spark-grid">

        {/* Estado sostenido */}
        <div style={{ ...cardStyle(t), padding: 22 }}>
          <div style={{ fontSize: 10, color: t.textMuted, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 12 }}>
            Estado sostenido (5 min)
          </div>
          <StateBadge estado={ultima?.estado_ventana || 'FLUIDO'} />
          <div style={{ display: 'flex', gap: 16, marginTop: 14 }}>
            {[
              ['Media',    ultima?.avg_vehicles?.toFixed(1), t.text],
              ['Máximo',   ultima?.max_vehicles,             PALETTE.pink],
              ['Muestras', ultima?.n_samples,                t.textSub],
            ].map(([lbl, val, col]) => (
              <div key={lbl}>
                <div style={{ fontSize: 10, color: t.textMuted, marginBottom: 3 }}>{lbl}</div>
                <div style={{ fontSize: 20, fontWeight: 500, color: col, fontFamily: 'var(--font-display)' }}>
                  {val ?? '—'}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Tendencia */}
        <div style={{ ...cardStyle(t), padding: 22, border: `1px solid ${tColor}30` }}>
          <div style={{ fontSize: 10, color: t.textMuted, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 12 }}>
            Tendencia actual
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
            <div style={{
              width: 40, height: 40, borderRadius: 10,
              background: `${tColor}18`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <TIcon size={20} color={tColor} strokeWidth={2.5} />
            </div>
            <div>
              <div style={{ fontWeight: 500, fontSize: 20, color: tColor, fontFamily: 'var(--font-display)' }}>
                {TENDENCIA[tendencia]?.label || tendencia}
              </div>
              <div style={{ color: t.textMuted, fontSize: 11, marginTop: 2 }}>
                últimas {ventanas.length} ventanas
              </div>
            </div>
          </div>
          {/* Barra de distribución */}
          <div style={{ display: 'flex', height: 5, borderRadius: 3, overflow: 'hidden', gap: 1 }}>
            {pct('FLUIDO')  > 0 && <div style={{ flex: pct('FLUIDO'),  background: PALETTE.emerald }} />}
            {pct('DENSO')   > 0 && <div style={{ flex: pct('DENSO'),   background: PALETTE.yellow  }} />}
            {pct('COLAPSO') > 0 && <div style={{ flex: pct('COLAPSO'), background: PALETTE.pink    }} />}
          </div>
          <div style={{ display: 'flex', gap: 10, marginTop: 6 }}>
            {[['FLUIDO', PALETTE.emerald], ['DENSO', PALETTE.yellow], ['COLAPSO', PALETTE.pink]].map(([k, c]) =>
              pct(k) > 0 && (
                <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <div style={{ width: 6, height: 6, borderRadius: '50%', background: c }} />
                  <span style={{ color: t.textMuted, fontSize: 10 }}>{k.charAt(0) + k.slice(1).toLowerCase()} {pct(k)}%</span>
                </div>
              )
            )}
          </div>
        </div>

        {/* Predicción */}
        <div style={{ ...cardStyle(t), padding: 22, border: `1px solid ${sPredict.color}40` }}>
          <div style={{ fontSize: 10, color: t.textMuted, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 12 }}>
            Predicción próximos 10 min
          </div>
          <StateBadge estado={prediccion || 'FLUIDO'} />
          <div style={{ color: t.textMuted, fontSize: 11, lineHeight: 1.6, marginTop: 12 }}>
            {tendencia === 'SUBIENDO' && 'El tráfico está aumentando. Se espera mayor densidad.'}
            {tendencia === 'BAJANDO'  && 'El tráfico está disminuyendo. La situación mejora.'}
            {tendencia === 'ESTABLE'  && 'El tráfico se mantiene estable. Sin cambios previstos.'}
          </div>
          <div style={{ color: t.textMuted, fontSize: 10, display: 'flex', alignItems: 'center', gap: 4, marginTop: 'auto', paddingTop: 12 }}>
            <Zap size={10} color={PALETTE.ocean} /> Calculado por Spark Streaming
          </div>
        </div>
      </div>

      {/* Gráfico de área */}
      <div style={{ ...cardStyle(t), padding: '20px 24px' }}>
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
          marginBottom: 18, flexWrap: 'wrap', gap: 8,
        }}>
          <span style={{ fontFamily: 'var(--font-display)', fontSize: 14, fontWeight: 500, color: t.textSub }}>
            Evolución de ventanas (media de vehículos por ventana 5 min)
          </span>
          <span style={{ fontSize: 11, color: t.textMuted }}>
            {ventanas.length} ventanas · actualiza cada 30s
          </span>
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
            <defs>
              <linearGradient id="gradAvg" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor={PALETTE.ocean} stopOpacity={0.25} />
                <stop offset="95%" stopColor={PALETTE.ocean} stopOpacity={0}    />
              </linearGradient>
            </defs>
            <CartesianGrid stroke={t.chartGrid} strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="time" tick={{ fill: t.chartText, fontSize: 10 }} interval="preserveStartEnd" axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: t.chartText, fontSize: 10 }} width={24} axisLine={false} tickLine={false} />
            <Tooltip
              contentStyle={{ background: t.surface, border: `1px solid ${t.borderMid}`, borderRadius: 10, fontSize: 12, boxShadow: t.shadowMd }}
              labelStyle={{ color: t.text, fontWeight: 600 }}
              formatter={(val, name) => [val, name === 'avg' ? 'Media' : 'Máximo']}
            />
            <ReferenceLine y={5}  stroke={PALETTE.yellow} strokeDasharray="4 3" strokeOpacity={0.5} />
            <ReferenceLine y={10} stroke={PALETTE.pink}   strokeDasharray="4 3" strokeOpacity={0.5} />
            <Area type="monotone" dataKey="avg" stroke={PALETTE.ocean} strokeWidth={2} fill="url(#gradAvg)" dot={false} activeDot={{ r: 4, fill: PALETTE.ocean, strokeWidth: 0 }} />
            <Line type="monotone" dataKey="max" stroke={PALETTE.yellow} strokeWidth={1.5} strokeDasharray="3 3" dot={false} />
          </AreaChart>
        </ResponsiveContainer>
        <div style={{ display: 'flex', gap: 16, marginTop: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 20, height: 2, background: PALETTE.ocean,  borderRadius: 1 }} />
            <span style={{ color: t.textMuted, fontSize: 11 }}>Media ventana</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 20, height: 2, background: PALETTE.yellow, borderRadius: 1, opacity: 0.7 }} />
            <span style={{ color: t.textMuted, fontSize: 11 }}>Máximo ventana</span>
          </div>
        </div>
      </div>

      {/* Tabla de últimas ventanas */}
      <div style={{ ...cardStyle(t), padding: '20px 24px' }}>
        <div style={{ fontFamily: 'var(--font-display)', fontSize: 14, fontWeight: 500, color: t.textSub, marginBottom: 16 }}>
          Últimas ventanas
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr>
                {['Hora', 'Estado', 'Media', 'Máx', 'Tendencia', 'Muestras'].map(h => (
                  <th key={h} style={{
                    color: t.textMuted, fontWeight: 500, fontSize: 10,
                    letterSpacing: '0.06em', textTransform: 'uppercase',
                    padding: '0 12px 10px 0', textAlign: 'left',
                    borderBottom: `1px solid ${t.border}`,
                  }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...ventanas].reverse().slice(0, 10).map((v, i) => {
                const sV = STATE[v.estado_ventana] || STATE.FLUIDO
                return (
                  <tr key={i} style={{ borderBottom: `1px solid ${t.border}` }}>
                    <td style={{ padding: '10px 12px 10px 0', color: t.textSub }}>
                      {new Date(v.timestamp).toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit' })}
                    </td>
                    <td style={{ padding: '10px 12px 10px 0' }}>
                      <span style={{ color: sV.color, fontWeight: 500 }}>{v.estado_ventana}</span>
                    </td>
                    <td style={{ padding: '10px 12px 10px 0', color: t.text, fontWeight: 500, fontFamily: 'var(--font-display)' }}>
                      {parseFloat(v.avg_vehicles).toFixed(1)}
                    </td>
                    <td style={{ padding: '10px 12px 10px 0', color: PALETTE.yellow, fontWeight: 500, fontFamily: 'var(--font-display)' }}>
                      {v.max_vehicles}
                    </td>
                    <td style={{ padding: '10px 12px 10px 0' }}>
                      <TendenciaBadge tendencia={v.tendencia} />
                    </td>
                    <td style={{ padding: '10px 0', color: t.textMuted }}>
                      {v.n_samples}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
