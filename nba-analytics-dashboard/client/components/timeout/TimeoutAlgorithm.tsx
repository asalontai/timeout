import { useState, useEffect } from "react";

interface TimeoutAlgorithmProps {
  gameState: {
    homeTeam: string;
    awayTeam: string;
    momentum: number;
    timeRemaining: string;
  };
}

interface LogEntry {
  id: number;
  timestamp: string;
  level: "SYS" | "ALERT" | "CRITICAL" | "INFO";
  message: string;
}

export default function TimeoutAlgorithm({ gameState }: TimeoutAlgorithmProps) {
  const [logs, setLogs] = useState<LogEntry[]>([
    {
      id: 1,
      timestamp: "10:24:32",
      level: "SYS",
      message: "System initialized and running",
    },
    {
      id: 2,
      timestamp: "10:24:25",
      level: "SYS",
      message: "Analyzing possession metrics...",
    },
    {
      id: 3,
      timestamp: "10:24:18",
      level: "SYS",
      message: "Fatigue index for MIA lineup at 78%",
    },
    {
      id: 4,
      timestamp: "10:24:10",
      level: "INFO",
      message: "Waiting for next dead ball event",
    },
  ]);

  const [criticalAlert, setCriticalAlert] = useState(true);

  useEffect(() => {
    // Simulate system logs
    const interval = setInterval(() => {
      const newLog: LogEntry = {
        id: Math.random(),
        timestamp: new Date().toLocaleTimeString(),
        level: Math.random() > 0.7 ? "ALERT" : "SYS",
        message: [
          "Analyzing possession metrics...",
          "Calculating player fatigue levels",
          "Monitoring game momentum shift",
          "Tracking timeout utilization",
          "Processing real-time analytics",
        ][Math.floor(Math.random() * 5)],
      };
      setLogs((prev) => [newLog, ...prev].slice(0, 15));
    }, 6000);

    return () => clearInterval(interval);
  }, []);

  const isCritical = gameState.momentum < -10 || gameState.momentum > 20;

  return (
    <div className="terminal-panel p-4 h-full flex flex-col overflow-hidden bg-gray-950">
      {/* Header */}
      <div className="mb-4 pb-2 border-b border-gray-600">
        <div className="terminal-glow text-sm font-bold">
          &gt; TIMEOUT_ALGORITHM_OUTPUT
        </div>
      </div>

      {/* Scrolling logs */}
      <div className="flex-1 overflow-y-auto space-y-1 font-mono text-xs mb-4">
        {logs.map((log) => (
          <div
            key={log.id}
            className={`py-0.5 ${
              log.level === "CRITICAL"
                ? "text-red-400"
                : log.level === "ALERT"
                  ? "text-orange-400"
                  : "text-gray-500"
            }`}
          >
            [{log.timestamp}] {log.level === "SYS" ? "[SYS]" : `[${log.level}]`} {">"} {log.message}
          </div>
        ))}
      </div>

      {/* Command line prompt with critical alert */}
      <div className="border-t border-gray-600 pt-3 space-y-2">
        {isCritical && (
          <div className="bg-red-900/20 border border-red-700/50 p-2 rounded-none animate-pulse-glow">
            <div className="terminal-alert font-bold text-xs">
              ADMIN@TIMEOUT_SYS:~# CRITICAL_ALERT
            </div>
            <div className="terminal-alert text-xs ml-2 mt-1">
              OPPONENT RUN DETECTED (12-2)
            </div>
            <div className="terminal-alert text-xs ml-2 mt-1 font-bold">
              RECOMMENDATION: {'>>>>'} INITIATE TIMEOUT SEQUENCE NOW {'<<<<'}
            </div>
          </div>
        )}

        {/* Status prompt */}
        <div className="bg-gray-800/40 border border-gray-600/50 p-2 rounded-none">
          <div className="text-gray-500 text-xs">
            ADMIN@TIMEOUT_SYS:~#
            <span className="animate-blink">▌</span>
          </div>
        </div>

        {/* Quick stats */}
        <div className="space-y-1 text-xs pt-2 border-t border-gray-600/30">
          <div className="text-gray-500">
            Momentum: <span className="terminal-glow">{gameState.momentum > 0 ? "+" : ""}{gameState.momentum}%</span>
          </div>
          <div className="text-gray-500">
            Time Remaining: <span className="terminal-glow">{gameState.timeRemaining}</span>
          </div>
          <div className="text-gray-500">
            Status: <span className={isCritical ? "terminal-alert animate-blink" : "terminal-glow"}>{isCritical ? "CRITICAL" : "MONITORING"}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
