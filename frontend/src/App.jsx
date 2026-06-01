import React, { useState } from "react";

/** Künye 6.3: Vite proxy /api → FastAPI kökü */
const API = import.meta.env.VITE_API_BASE || "/api";

async function apiGet(path) {
  const r = await fetch(`${API}${path}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function apiPost(path, opts = {}) {
  const r = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [predict, setPredict] = useState(null);
  const [act, setAct] = useState(null);
  const [algo, setAlgo] = useState("ppo");
  const [err, setErr] = useState("");

  const loadHealth = async () => {
    setErr("");
    try {
      setHealth(await apiGet("/health"));
    } catch (e) {
      setErr(String(e.message));
    }
  };

  const runPredict = async () => {
    setErr("");
    try {
      setPredict(await apiPost("/predict"));
    } catch (e) {
      setErr(String(e.message));
    }
  };

  const runAct = async () => {
    setErr("");
    try {
      setAct(await apiPost(`/act?algo=${encodeURIComponent(algo)}`));
    } catch (e) {
      setErr(String(e.message));
    }
  };

  return (
    <div>
      <h1>Akıllı Park — React (PDF 6.3)</h1>
      <p className="card">
        FastAPI arka uç: <code>uvicorn api.main:app --app-dir .</code>
        <br />
        Geliştirme: <code>npm run dev</code> (proxy <code>/api</code> → 8000)
      </p>
      {err && <p style={{ color: "#f88" }}>{err}</p>}

      <div className="card">
        <h2>GET /health</h2>
        <button type="button" onClick={loadHealth}>
          Yükle
        </button>
        {health && <pre>{JSON.stringify(health, null, 2)}</pre>}
      </div>

      <div className="card">
        <h2>POST /predict</h2>
        <button type="button" onClick={runPredict}>
          Tahmin
        </button>
        {predict && <pre>{JSON.stringify(predict, null, 2)}</pre>}
      </div>

      <div className="card">
        <h2>POST /act</h2>
        <label>
          Algoritma{" "}
          <select value={algo} onChange={(e) => setAlgo(e.target.value)}>
            <option value="ppo">PPO</option>
            <option value="dqn">DQN</option>
          </select>
        </label>
        <button type="button" onClick={runAct}>
          Bölüm oynat
        </button>
        {act && <pre>{JSON.stringify(act, null, 2)}</pre>}
      </div>
    </div>
  );
}
