import { useState, useEffect, useRef, useCallback } from "react";
import LiveFeed from "@/components/timeout/LiveFeed";
import MomentumTimeline from "@/components/timeout/MomentumCenter";
import StatisticalModel from "@/components/timeout/StatisticalModel";

// ============================================================
// Types
// ============================================================

export interface Play {
  idx: number;
  period: number;
  clock: string;
  clockSeconds: number;
  team: string;
  action: string;
  description: string;
  homeScore: number;
  awayScore: number;
  scoreDiff: number;
  sacRun: number;
  sasRun: number;
  momentum: number;
  m1Timeout: boolean;
  m1AvgMomentum: number;
  m1Confidence: number;
  m2Timeout: boolean;
  m2ProbBeneficial: number;
  m2Confidence: number;
  finalTimeout: boolean;
  finalConfidence: number;
  agreement: "AGREE" | "DISAGREE";
  isMomentumShift: boolean;
  isSignificant: boolean;
  oppFgPct: number;
  ownFgPct: number;
  ownTurnovers: number;
  oppTurnovers: number;
  isCoachTimeout?: boolean;
}

export interface GameInfo {
  homeTeam: string;
  awayTeam: string;
  homeTeamFull: string;
  awayTeamFull: string;
  gameId: string;
  date: string;
  finalHomeScore: number;
  finalAwayScore: number;
}

export interface LiveDemoData {
  gameInfo: GameInfo;
  plays: Play[];
}

export interface MomentumPoint {
  label: string;
  momentum: number;
  period: number;
}

export default function Index() {
  const [data, setData] = useState<LiveDemoData | null>(null);
  const [currentPlayIdx, setCurrentPlayIdx] = useState(0);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Load JSON
  useEffect(() => {
    fetch("/live_demo.json")
      .then((r) => r.json())
      .then((d: LiveDemoData) => setData(d))
      .catch((e) => console.error("Failed to load demo data:", e));
  }, []);

  // Filter to significant plays (including all timeouts from both teams)
  const filteredPlays = data
    ? data.plays.filter((p) => p.isSignificant)
    : [];

  // Auto-advance with random 10-20s delay
  const scheduleNext = useCallback(() => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    const delay = Math.floor(Math.random() * 11 + 10) * 1000;
    timeoutRef.current = setTimeout(() => {
      setCurrentPlayIdx((prev) => {
        if (prev >= filteredPlays.length - 1) return prev;
        return prev + 1;
      });
    }, delay);
  }, [filteredPlays.length]);

  useEffect(() => {
    if (filteredPlays.length > 0) {
      scheduleNext();
    }
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, [currentPlayIdx, scheduleNext, filteredPlays.length]);

  if (!data) {
    return (
      <div className="h-screen bg-black text-gray-300 flex items-center justify-center font-mono">
        <div className="text-center">
          <div className="text-2xl animate-pulse mb-4">Loading game data...</div>
          <div className="text-sm text-gray-500">Fetching SAS @ SAC play-by-play</div>
        </div>
      </div>
    );
  }

  const currentPlay = filteredPlays[currentPlayIdx] || null;
  const visiblePlays = filteredPlays.slice(0, currentPlayIdx + 1);

  // Build momentum points with timeout recalibration
  let momentumOffset = 0;
  const recalibrated: { label: string; momentum: number; period: number }[] = [];
  for (let i = 0; i < visiblePlays.length; i++) {
    const p = visiblePlays[i];
    const isTimeout = p.action === "TIMEOUT";

    if (isTimeout) {
      momentumOffset = p.momentum;
      recalibrated.push({ label: `Q${p.period} ${p.clock}`, momentum: 0, period: p.period });
    } else {
      recalibrated.push({
        label: `Q${p.period} ${p.clock}`,
        momentum: p.momentum - momentumOffset,
        period: p.period,
      });
    }
  }

  const momentumPoints: MomentumPoint[] = [{ label: "Start", momentum: 0, period: 1 }];
  const sampleStep = Math.max(1, Math.floor(recalibrated.length / 60));
  for (let i = 0; i < recalibrated.length; i++) {
    if (i % sampleStep === 0) {
      momentumPoints.push(recalibrated[i]);
    }
  }
  if (recalibrated.length > 0) {
    momentumPoints.push(recalibrated[recalibrated.length - 1]);
  }

  // Ensemble Logic: Model B is only considered if opponent streak > 4
  const modelBActive = (currentPlay?.sasRun ?? 0) > 4;
  const isCallTimeout = modelBActive
    ? (currentPlay?.finalTimeout ?? false)
    : (currentPlay?.m1Timeout ?? false);

  const verdictColor = isCallTimeout ? "text-purple-400" : "text-gray-500";
  const verdictBg = isCallTimeout
    ? "bg-purple-500/10 border-purple-500/40"
    : "bg-gray-800/30 border-gray-600";

  return (
    <div className="h-screen bg-black text-gray-300 flex flex-col overflow-hidden">
      {/* Thin top banner */}
      <div className="border-b border-gray-700 bg-gray-950/80 px-4 py-1.5 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-500 font-mono">
            {data.gameInfo.awayTeamFull} @ {data.gameInfo.homeTeamFull}
          </span>
        </div>
        <div className="text-xs text-gray-500 font-mono">
          Q{currentPlay?.period ?? 1} | {currentPlay?.clock ?? "12:00"} | {data.gameInfo.homeTeam} {currentPlay?.homeScore ?? 0} — {data.gameInfo.awayTeam} {currentPlay?.awayScore ?? 0}
        </div>
      </div>

      {/* 2x2 Grid */}
      <div className="flex-1 grid grid-cols-2 grid-rows-2 gap-3 p-3 overflow-hidden">

        {/* TOP LEFT — Live Feed */}
        <div className="min-h-0 overflow-hidden">
          <LiveFeed
            currentPlay={currentPlay}
            recentPlays={visiblePlays.slice(-20).reverse()}
            gameInfo={data.gameInfo}
          />
        </div>

        {/* TOP RIGHT — Dual Model Verdict */}
        <div className="terminal-panel p-4 bg-gray-950 overflow-hidden flex flex-col min-h-0">
          <div className="mb-2 pb-2 border-b border-gray-600 shrink-0">
            <div className="terminal-glow text-sm font-bold">
              &gt; Dual Model Verdict
            </div>
          </div>

          <div className="flex-1 flex flex-col justify-center space-y-3 overflow-hidden">
            {/* BIG VERDICT */}
            {currentPlay && (
              <div className={`text-center p-4 border transition-all duration-500 ${verdictBg}`}>
                <div className="text-xs text-gray-500 mb-1 font-mono">ENSEMBLE RECOMMENDATION</div>
                <div className={`text-3xl font-bold ${verdictColor}`}>
                  {currentPlay.finalTimeout ? "CALL TIMEOUT" : "NO TIMEOUT"}
                </div>
              </div>
            )}

            {/* Model cards */}
            {currentPlay && (
              <div className="grid grid-cols-2 gap-2">
                <div className="p-2 bg-gray-900/50 border border-gray-700">
                  <div className="text-[10px] text-gray-500 font-mono mb-1">MODEL A — MORALE</div>
                  <div className={`text-sm font-bold ${currentPlay.m1Timeout ? "text-purple-400" : "text-gray-500"}`}>
                    {currentPlay.m1Timeout ? "TIMEOUT" : "NO"}
                  </div>
                </div>
                <div className="p-2 bg-gray-900/50 border border-gray-700">
                  <div className="text-[10px] text-gray-500 font-mono mb-1">MODEL B — XGBOOST</div>
                  <div className={`text-sm font-bold ${currentPlay.m2Timeout ? "text-purple-400" : "text-gray-500"}`}>
                    {currentPlay.m2Timeout ? "TIMEOUT" : "NO"}
                  </div>
                </div>
              </div>
            )}

            {/* Ensemble Status Indicator */}
            {currentPlay && (
              <div className="pt-2 border-t border-gray-700/50 flex items-center justify-between text-[10px] font-mono">
                <span className="text-gray-500 uppercase tracking-widest">Ensemble Logic Status:</span>
                <span className={modelBActive ? "text-purple-400" : "text-yellow-600"}>
                  {modelBActive ? "MODEL B ACTIVE (>4 STREAK)" : "MODEL B BYPASSED (STREAK ≤ 4)"}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* BOTTOM LEFT — Morale Analysis (Model A) */}
        <div className="min-h-0 overflow-hidden">
          <MomentumTimeline
            momentumPoints={momentumPoints}
            gameInfo={data.gameInfo}
          />
        </div>

        {/* BOTTOM RIGHT — Statistical Model Bar Chart */}
        <div className="min-h-0 overflow-hidden">
          <StatisticalModel
            allPlays={filteredPlays}
            currentPlay={currentPlay}
            gameInfo={data.gameInfo}
          />
        </div>
      </div>
    </div>
  );
}
