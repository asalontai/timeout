import { useState, useEffect, useRef } from "react";
import type { Play, GameInfo } from "@/pages/Index";

interface TimeoutAlgorithmProps {
  currentPlay: Play | null;
  gameInfo: GameInfo;
}

interface LogEntry {
  id: string;
  level: "SYS" | "ALERT" | "CRITICAL" | "INFO" | "MODEL";
  message: string;
}

function buildLogsForPlay(p: Play, gi: GameInfo): LogEntry[] {
  const logs: LogEntry[] = [];
  const prefix = `Q${p.period}`;

  logs.push({ id: `${p.idx}-1`, level: "SYS", message: `[${prefix} ${p.clock}] Analyzing play #${p.idx}...` });
  logs.push({ id: `${p.idx}-2`, level: "SYS", message: `Score: ${gi.homeTeam} ${p.homeScore} - ${gi.awayTeam} ${p.awayScore} (diff: ${p.scoreDiff > 0 ? "+" : ""}${p.scoreDiff})` });

  if (p.sasRun >= 5) {
    logs.push({ id: `${p.idx}-3`, level: "ALERT", message: `${gi.awayTeam} ${p.sasRun}-0 RUN DETECTED` });
  }
  if (p.sacRun >= 5) {
    logs.push({ id: `${p.idx}-3b`, level: "INFO", message: `${gi.homeTeam} ${p.sacRun}-0 RUN` });
  }

  logs.push({ id: `${p.idx}-4`, level: "MODEL", message: "--- MORALE MODEL ---" });
  logs.push({
    id: `${p.idx}-5`,
    level: p.m1Timeout ? "ALERT" : "INFO",
    message: `Momentum: ${p.m1AvgMomentum > 0 ? "+" : ""}${p.m1AvgMomentum.toFixed(2)} -> ${p.m1Timeout ? "TIMEOUT" : "NO"} (${(p.m1Confidence * 100).toFixed(0)}%)`,
  });

  logs.push({ id: `${p.idx}-6`, level: "MODEL", message: "--- XGBOOST MODEL ---" });
  logs.push({
    id: `${p.idx}-7`,
    level: p.m2Timeout ? "ALERT" : "INFO",
    message: `P(beneficial): ${(p.m2ProbBeneficial * 100).toFixed(0)}% -> ${p.m2Timeout ? "TIMEOUT" : "NO"} (${(p.m2Confidence * 100).toFixed(0)}%)`,
  });

  logs.push({ id: `${p.idx}-8`, level: "MODEL", message: "--- ENSEMBLE ---" });
  logs.push({
    id: `${p.idx}-9`,
    level: p.finalTimeout ? "CRITICAL" : "INFO",
    message: `VERDICT: ${p.finalTimeout ? "CALL TIMEOUT" : "NO TIMEOUT"} (${(p.finalConfidence * 100).toFixed(0)}%) [${p.agreement}]`,
  });

  return logs;
}

export default function TimeoutAlgorithm({ currentPlay, gameInfo }: TimeoutAlgorithmProps) {
  const [logHistory, setLogHistory] = useState<LogEntry[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const lastPlayIdx = useRef<number>(-1);

  useEffect(() => {
    if (!currentPlay || currentPlay.idx === lastPlayIdx.current) return;
    lastPlayIdx.current = currentPlay.idx;

    const newLogs = buildLogsForPlay(currentPlay, gameInfo);
    setLogHistory((prev) => [...newLogs, ...prev].slice(0, 100)); // keep last 100 entries

    // Auto-scroll to top
    if (scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
  }, [currentPlay?.idx]);

  const isCritical = currentPlay?.finalTimeout ?? false;

  return (
    <div className="terminal-panel p-4 h-full flex flex-col overflow-hidden bg-gray-950">
      {/* Header */}
      <div className="mb-3 pb-2 border-b border-gray-600">
        <div className="terminal-glow text-sm font-bold">
          &gt; TIMEOUT_ALGORITHM_OUTPUT
        </div>
        <div className="text-xs text-gray-600 mt-1">
          Real-time analysis — {gameInfo.awayTeam} @ {gameInfo.homeTeam}
        </div>
      </div>

      {/* Scrolling logs */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-0.5 font-mono text-xs mb-3">
        {logHistory.map((log) => (
          <div
            key={log.id}
            className={`py-0.5 ${log.level === "CRITICAL"
                ? "text-green-400 font-bold"
                : log.level === "ALERT"
                  ? "text-orange-400"
                  : log.level === "MODEL"
                    ? "text-blue-300 opacity-70"
                    : log.level === "INFO"
                      ? "text-gray-400"
                      : "text-gray-500"
              }`}
          >
            {log.level === "MODEL" ? (
              <span>{log.message}</span>
            ) : (
              <span>
                [{log.level}] {">"} {log.message}
              </span>
            )}
          </div>
        ))}
      </div>

      {/* Verdict bar */}
      <div className="border-t border-gray-600 pt-3 space-y-2">
        {isCritical ? (
          <div className="bg-green-900/20 border border-green-700/50 p-2 animate-pulse-glow">
            <div className="text-green-400 font-bold text-xs">
              RECOMMENDATION: CALL TIMEOUT NOW
            </div>
          </div>
        ) : (
          <div className="bg-gray-800/40 border border-gray-600/50 p-2">
            <div className="text-gray-500 text-xs">
              Status: Monitoring — no timeout needed
            </div>
          </div>
        )}

        {/* Quick stats */}
        {currentPlay && (
          <div className="space-y-1 text-xs pt-2 border-t border-gray-600/30">
            <div className="text-gray-500">
              Momentum:{" "}
              <span className={currentPlay.momentum > 0 ? "text-purple-400" : "text-gray-300"}>
                {currentPlay.momentum > 0 ? "+" : ""}{currentPlay.momentum.toFixed(2)}
              </span>
            </div>
            <div className="text-gray-500">
              Opp FG%:{" "}
              <span className="terminal-glow">{(currentPlay.oppFgPct * 100).toFixed(0)}%</span>
              {" | "}Own FG%:{" "}
              <span className="terminal-glow">{(currentPlay.ownFgPct * 100).toFixed(0)}%</span>
            </div>
            <div className="text-gray-500">
              Turnovers — Own: {currentPlay.ownTurnovers} | Opp: {currentPlay.oppTurnovers}
            </div>
          </div>
        )}

        {/* Prompt */}
        <div className="bg-gray-800/40 border border-gray-600/50 p-2">
          <div className="text-gray-500 text-xs">
            ADMIN@TIMEOUT_SYS:~# <span className="animate-blink">|</span>
          </div>
        </div>
      </div>
    </div>
  );
}
