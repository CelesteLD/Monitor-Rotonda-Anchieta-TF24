import { useState, useEffect, useCallback, useRef } from 'react'
import { Clock, Sun, Moon, Radio, TrendingUp, BarChart2 } from 'lucide-react'
import { T, PALETTE } from './tokens'
import TabLive       from './TabLive'
import TabTendencias from './TabTendencias'
import TabHistorico  from './TabHistorico'

const API = `${import.meta.env.VITE_API_URL || ''}/api`

const TABS = [
  { id: 'live',       label: 'En directo', Icon: Radio      },
  { id: 'tendencias', label: 'Tendencias', Icon: TrendingUp  },
  { id: 'historico',  label: 'Histórico',  Icon: BarChart2   },
]

export default function App() {
  const [theme,        setTheme]        = useState('light')
  const [tab,          setTab]          = useState('live')
  const [status,       setStatus]       = useState(null)
  const [history,      setHistory]      = useState([])
  const [stats,        setStats]        = useState(null)
  const [frameUrl,     setFrameUrl]     = useState(null)
  const [lastUpdate,   setLastUpdate]   = useState(null)
  const [error,        setError]        = useState(null)
  const [sseConnected, setSseConnected] = useState(false)
  const eventSourceRef                  = useRef(null)

  const t = T[theme]

  const fetchAll = useCallback(async () => {
    try {
      const [s, h, st] = await Promise.all([
        fetch(`${API}/status`).then(r => r.json()),
        fetch(`${API}/history?minutes=120`).then(r => r.json()),
        fetch(`${API}/stats`).then(r => r.json()),
      ])
      setStatus(s)
      setHistory(h.map(row => ({
        ...row,
        time: new Date(row.timestamp).toLocaleTimeString('es-ES', {
          hour: '2-digit', minute: '2-digit',
        }),
        vehicles: row.vehicle_count,
      })))
      setStats(st)
      setFrameUrl(`${API}/frame?t=${Date.now()}`)
      setLastUpdate(new Date())
      setError(null)
    } catch {
      setError('No se puede conectar con el backend')
    }
  }, [])

  useEffect(() => {
    fetchAll()
    function connect() {
      if (eventSourceRef.current) eventSourceRef.current.close()
      const es = new EventSource(`${API}/events`)
      eventSourceRef.current = es
      es.addEventListener('connected',  () => { setSseConnected(true); setError(null) })
      es.addEventListener('detection',  fetchAll)
      es.onerror = () => { setSseConnected(false); es.close(); setTimeout(connect, 5000) }
    }
    connect()
    return () => eventSourceRef.current?.close()
  }, [fetchAll])

  return (
    <>
      <style>{`
        body {
          background: ${t.bg};
          color: ${t.text};
          transition: background 0.2s, color 0.2s;
        }
        .main-grid {
          grid-template-columns: minmax(0,1fr) minmax(260px,300px) !important;
        }
        .spark-grid {
          grid-template-columns: repeat(3,1fr) !important;
        }
        @media (max-width: 900px) {
          .spark-grid { grid-template-columns: 1fr !important; }
        }
        @media (max-width: 700px) {
          .main-grid { grid-template-columns: 1fr !important; }
        }
        @media (max-width: 520px) {
          .header-inner { flex-direction: column; gap: 12px; align-items: flex-start !important; }
          .tab-bar      { width: 100% !important; }
          .tab-bar button { flex: 1; justify-content: center; }
        }
      `}</style>

      <div style={{ minHeight: '100vh', background: t.bg, transition: 'background 0.2s' }}>
        <div style={{ maxWidth: 1450, margin: '0 auto', padding: 'clamp(16px, 3vw, 28px)' }}>

          {/* Header */}
          <div className="header-inner" style={{
            display: 'flex', justifyContent: 'space-between',
            alignItems: 'center', marginBottom: 28, gap: 16,
          }}>
            <div>
              <h1 style={{
                fontFamily: 'var(--font-display)',
                fontSize: 'clamp(17px, 2.5vw, 22px)', fontWeight: 500,
                color: t.text, margin: 0, letterSpacing: '-0.03em', lineHeight: 1.1,
              }}>
                Monitor Rotonda Anchieta
                <span style={{ color: PALETTE.ocean, marginLeft: 6 }}>TF-24</span>
              </h1>
              <div style={{ color: t.textMuted, fontSize: 11, marginTop: 4 }}>
                TV 5.17 · CIC Tenerife · Cabildo Insular
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 }}>
              {/* Indicador conexión */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: t.textMuted, fontSize: 11 }}>
                <span className={sseConnected ? 'pulse-connected' : ''} style={{
                  width: 7, height: 7, borderRadius: '50%', display: 'inline-block',
                  background: sseConnected ? PALETTE.emerald : '#94a3b8',
                }} />
                <Clock size={12} />
                <span>{lastUpdate?.toLocaleTimeString('es-ES') || 'Cargando...'}</span>
              </div>

              {/* Toggle tema */}
              <button onClick={() => setTheme(th => th === 'light' ? 'dark' : 'light')} style={{
                background: t.surface, border: `1px solid ${t.border}`,
                borderRadius: 10, padding: '7px 13px', cursor: 'pointer',
                display: 'flex', alignItems: 'center', gap: 6,
                color: t.textSub, fontSize: 12, fontWeight: 500,
                fontFamily: 'var(--font-display)', boxShadow: t.shadow,
              }}>
                {theme === 'light' ? <Moon size={14} /> : <Sun size={14} />}
                {theme === 'light' ? 'Oscuro' : 'Claro'}
              </button>
            </div>
          </div>

          {/* Pestañas */}
          <div className="tab-bar" style={{
            display: 'inline-flex', gap: 3, marginBottom: 24,
            background: t.surface, border: `1px solid ${t.border}`,
            borderRadius: 12, padding: 4, boxShadow: t.shadow,
          }}>
            {TABS.map(({ id, label, Icon }) => (
              <button key={id} onClick={() => setTab(id)} style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '8px 16px', borderRadius: 9, cursor: 'pointer',
                fontSize: 13, fontWeight: 500, border: 'none',
                fontFamily: 'var(--font-display)',
                transition: 'all 0.15s',
                background: tab === id ? PALETTE.ocean : 'transparent',
                color:      tab === id ? '#fff'         : t.textMuted,
                boxShadow:  tab === id ? `0 2px 8px ${PALETTE.ocean}44` : 'none',
              }}>
                <Icon size={14} strokeWidth={2} />
                {label}
              </button>
            ))}
          </div>

          {/* Contenido */}
          {tab === 'live'       && (
            <TabLive
              status={status} history={history}
              stats={stats} frameUrl={frameUrl}
              error={error} t={t}
            />
          )}
          {tab === 'tendencias' && <TabTendencias t={t} />}
          {tab === 'historico'  && <TabHistorico  t={t} />}

        </div>
      </div>
    </>
  )
}
