import React, { useEffect, useRef, useState, useCallback } from "react";
import { io, Socket } from "socket.io-client";
import { Stage, Layer, Line, Circle, Rect } from "react-konva";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://localhost:5000";

interface RRTNode {
  x: number;
  y: number;
  parent: number | null;
}

interface RRTSnapshot {
  iteration: number;
  nodes: [number, number, number | null][];
  goal_reached: boolean;
  path: [number, number][] | null;
}

interface MapfSnapshot {
  iteration: number;
  paths: [number, number][][]; // per agent list of coords
  conflict: any; // simplified
}

interface MapfDone {
  run_id: string;
  paths: [number, number][][];
}

export default function App() {
  const socketRef = useRef<Socket | null>(null);
  const [rrtSnap, setRrtSnap] = useState<RRTSnapshot | null>(null);
  const [mapfSnap, setMapfSnap] = useState<MapfSnapshot | null>(null);
  const [mode, setMode] = useState<"rrt" | "mapf">("rrt");
  // MAPF animation state
  const [mapfPaths, setMapfPaths] = useState<[number, number][][]>([]);
  const [timeStep, setTimeStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1); // steps per second

  // RRT* parameters
  const [maxIter, setMaxIter] = useState<number>(5000);

  // Default configurations (used for initial render too)
  const RRT_CONFIG = {
    start: [20, 20] as [number, number],
    goal: [780, 580] as [number, number],
    bounds: [ [0,0] as [number,number], [800,600] as [number,number] ] as [[number,number],[number,number]],
    obstacles: [
      { x: 300, y: 0, w: 50, h: 400 },
      { x: 500, y: 200, w: 60, h: 400 },
      { x: 100, y: 300, w: 150, h: 40 },
      { x: 650, y: 0, w: 40, h: 250 },
    ]
  };

  const MAPF_CONFIG = {
    cellSize: 20,
    cols: 30,
    rows: 20,
    obstacles: [
      { x: 10*20, y: 0, w: 2*20, h: 15*20 },
      { x: 18*20, y: 5*20, w: 2*20, h: 15*20 },
      { x: 5*20, y: 9*20, w: 20*20, h: 2*20 },
    ],
    agents: [
      { start: [0,0] as [number,number], goal: [29,19] as [number,number] },
      { start: [29,0] as [number,number], goal: [0,19] as [number,number] },
      { start: [0,19] as [number,number], goal: [29,0] as [number,number] },
    ]
  };

  // connect socket once
  useEffect(() => {
    const socket = io(BACKEND_URL);
    socketRef.current = socket;
    socket.on("rrt_snapshot", (snap) => {
      console.log("rrt snapshot", snap.iteration);
      setRrtSnap(snap as RRTSnapshot);
    });
    socket.on("mapf_snapshot", (snap) => {
      setMapfSnap(snap as MapfSnapshot);
    });
    socket.on("mapf_done", (data: MapfDone) => {
      setMapfPaths(data.paths);
      setTimeStep(0);
      setPlaying(true);
    });
    return () => {
      socket.disconnect();
    };
  }, []);

  const handleRunRRT = async () => {
    setMode("rrt");
    setRrtSnap(null);
    await safeFetch(`${BACKEND_URL}/api/run/rrtstar`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        start: RRT_CONFIG.start,
        goal: RRT_CONFIG.goal,
        bounds: RRT_CONFIG.bounds,
        obstacles: RRT_CONFIG.obstacles,
        step_size: 15,
        max_iter: maxIter,
        snapshot_interval: 10
      }),
    });
  };

  const handleRunMapf = async () => {
    setMode("mapf");
    setMapfSnap(null);
    await safeFetch(`${BACKEND_URL}/api/run/mapf`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        bounds: [[0,0],[MAPF_CONFIG.cols*MAPF_CONFIG.cellSize, MAPF_CONFIG.rows*MAPF_CONFIG.cellSize]],
        cell_size: MAPF_CONFIG.cellSize,
        obstacles: MAPF_CONFIG.obstacles,
        agents: MAPF_CONFIG.agents,
      }),
    });
  };

  // animation effect for MAPF
  useEffect(() => {
    if (!playing) return;
    const handle = setInterval(() => {
      setTimeStep((t) => t + 1);
    }, 1000 / speed);
    return () => clearInterval(handle);
  }, [playing, speed]);

  const togglePlay = () => setPlaying((p) => !p);
  const resetPlay = () => {
    setTimeStep(0);
    setPlaying(false);
  };

  // helper to handle fetch errors
  const safeFetch = (input: RequestInfo, init?: RequestInit) => fetch(input, init).then(r=>{
    if(!r.ok){console.error("Request failed", r.status);}
    return r;
  }).catch(err=>console.error("Network error", err));

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      <header style={{ padding: "8px", background: "#eee", display: "flex", alignItems: "center", gap: 8 }}>
        <button onClick={handleRunRRT}>Run RRT*</button>
        <label style={{marginLeft:12}}>Iterations:
          <input type="number" min={100} max={100000} step={100} value={maxIter}
            onChange={e=>setMaxIter(Number(e.target.value))}
            style={{width:80,marginLeft:4}}/>
        </label>
        <button onClick={handleRunMapf}>Run MAPF (CBS)</button>
        {mode === "mapf" && mapfPaths.length > 0 && (
          <>
            <button onClick={togglePlay}>{playing ? "Pause" : "Play"}</button>
            <button onClick={resetPlay}>Reset</button>
            <label style={{ marginLeft: 8 }}>
              Speed: {speed}x
              <input
                type="range"
                min={1}
                max={10}
                value={speed}
                onChange={(e) => setSpeed(Number(e.target.value))}
                style={{ marginLeft: 4 }}
              />
            </label>
          </>
        )}
      </header>
      <div style={{ flex: 1 }}>
        {mode === "rrt" ? (
          <RRTCanvas snapshot={rrtSnap} config={RRT_CONFIG} />
        ) : (
          <MapfCanvas snapshot={mapfSnap} paths={mapfPaths} time={timeStep} config={MAPF_CONFIG} />
        )}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------
// RRT Canvas
// ------------------------------------------------------------------

function RRTCanvas({ snapshot, config }: { snapshot: RRTSnapshot | null; config: {start:[number,number];goal:[number,number];bounds:[[number,number],[number,number]];obstacles:{x:number;y:number;w:number;h:number}[]} }) {
  const width = config.bounds[1][0];
  const height = config.bounds[1][1];

  const nodes = snapshot?.nodes ?? [];
  const path = snapshot?.path ?? null;

  // ---- Sampling to reduce clutter ----
  const MAX_EDGES = 6000;
  const step = Math.max(1, Math.floor(nodes.length / MAX_EDGES));
  const lines: number[] = [];
  const sampledNodes: [number, number, number | null][] = [];
  // helper geometry
  const pointInsideRect = (x:number,y:number,r:any)=> x>=r.x && x<=r.x+r.w && y>=r.y && y<=r.y+r.h;
  const EPS=1e-6;
  const segIntersect = (p:any,q:any,r:any,s:any)=>{
    const orient=(a:any,b:any,c:any)=>{
       const val=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
       if(Math.abs(val)<EPS) return 0;
       return val>0?1:2; // 1 counterclockwise, 2 clockwise
    };
    const onSeg=(a:any,b:any,c:any)=>Math.min(a[0],b[0])-EPS<=c[0]&&c[0]<=Math.max(a[0],b[0])+EPS&&Math.min(a[1],b[1])-EPS<=c[1]&&c[1]<=Math.max(a[1],b[1])+EPS;
    const o1=orient(p,q,r), o2=orient(p,q,s), o3=orient(r,s,p), o4=orient(r,s,q);
    if(o1!=o2 && o3!=o4) return true;
    if(o1===0 && onSeg(p,q,r)) return true;
    if(o2===0 && onSeg(p,q,s)) return true;
    if(o3===0 && onSeg(r,s,p)) return true;
    if(o4===0 && onSeg(r,s,q)) return true;
    return false;
  };
  const edgeFree=(x1:number,y1:number,x2:number,y2:number)=>{
    for(const ob of config.obstacles){
      // if both endpoints inside rect -> collision
      if(pointInsideRect(x1,y1,ob) || pointInsideRect(x2,y2,ob)) return false;
      // check intersection with rect edges
      const corners=[[ob.x,ob.y],[ob.x+ob.w,ob.y],[ob.x+ob.w,ob.y+ob.h],[ob.x,ob.y+ob.h]];
      for(let i=0;i<4;i++){
         const a=corners[i], b=corners[(i+1)%4];
         if(segIntersect([x1,y1],[x2,y2],a,b)) return false;
      }
    }
    return true;
  };
  nodes.forEach((n, idx) => {
    if (idx % step !== 0) return;
    // skip nodes inside any obstacle to avoid drawing invalid nodes/edges
    if(config.obstacles.some(ob=>pointInsideRect(n[0],n[1],ob))) return;
    sampledNodes.push(n);
    const parent = n[2];
    if (parent !== null && parent >= 0 && parent % step === 0) {
      const parentNode = nodes[parent];
      if(edgeFree(n[0],n[1],parentNode[0],parentNode[1])){
        lines.push(n[0], n[1], parentNode[0], parentNode[1]);
      }
    }
  });

  return (
    <Stage width={width} height={height} style={{ background: "#fafafa" }}>
      <Layer>
        {/* Tree edges */}
        {lines.length>0 && <Line key="edges" points={lines} stroke="#666" strokeWidth={1} opacity={0.6} />}
        {/* Tree nodes */}
        {sampledNodes.map((n,i)=>(<Circle key={i} x={n[0]} y={n[1]} radius={2} fill="#888" />))}
        {/* Obstacles (drawn last so they cover edges) */}
        {config.obstacles.map((o,i)=>(<Rect key={i} x={o.x} y={o.y} width={o.w} height={o.h} fill="#666" opacity={0.8} />))}
        {/* Path */}
        {path && (
          <Line
            points={path.flat()}
            stroke="red"
            strokeWidth={3}
            lineCap="round"
          />
        )}
        {/* Start / Goal */}
        <Circle x={config.start[0]} y={config.start[1]} radius={6} fill="green" />
        <Circle x={config.goal[0]} y={config.goal[1]} radius={6} fill="blue" />
      </Layer>
    </Stage>
  );
}

// ------------------------------------------------------------------
// MAPF Canvas
// ------------------------------------------------------------------

function MapfCanvas({ snapshot, paths, time, config }: { snapshot: MapfSnapshot | null; paths: [number, number][][]; time: number; config: {cellSize:number;cols:number;rows:number;obstacles:{x:number;y:number;w:number;h:number}[];agents:{start:[number,number];goal:[number,number]}[]} }) {
  const cell = config.cellSize;
  const cols = config.cols;
  const rows = config.rows;

  const agentsPos =
    paths.length > 0
      ? paths.map((p) => p[Math.min(time, p.length - 1)])
      : config.agents.map((a) => a.start);

  const colors = ["red", "blue", "green", "orange", "purple", "pink", "cyan", "lime"];

  return (
    <Stage width={cols * cell} height={rows * cell} style={{ background: "#fff" }}>
      <Layer>
        {/* grid */}
        {Array.from({ length: cols + 1 }, (_, i) => (
          <Line key={`v${i}`} points={[i * cell, 0, i * cell, rows * cell]} stroke="#ddd" />
        ))}
        {Array.from({ length: rows + 1 }, (_, i) => (
          <Line key={`h${i}`} points={[0, i * cell, cols * cell, i * cell]} stroke="#ddd" />
        ))}
        {/* obstacles */}
        {config.obstacles.map((o,i)=>(<Rect key={i} x={o.x} y={o.y} width={o.w} height={o.h} fill="#888" />))}
        {/* paths */}
        {(paths.length > 0 ? paths : []).map((path, idx) => (
          <Line
            key={"p" + idx}
            points={path.flat().map((v, i) => (i % 2 === 0 ? v * cell + cell / 2 : v * cell + cell / 2))}
            stroke={colors[idx % colors.length]}
            strokeWidth={2}
          />
        ))}
        {/* agents */}
        {agentsPos.map((p, idx) => (
          <Circle
            key={"a" + idx}
            x={p[0] * cell + cell / 2}
            y={p[1] * cell + cell / 2}
            radius={cell * 0.3}
            fill={colors[idx % colors.length]}
          />
        ))}
        {/* goals */}
        {config.agents.map((a,idx)=>(<Circle key={"g"+idx} x={a.goal[0]*cell+cell/2} y={a.goal[1]*cell+cell/2} radius={cell*0.3} stroke={colors[idx%colors.length]} strokeWidth={2} fill="transparent" />))}
      </Layer>
    </Stage>
  );
} 