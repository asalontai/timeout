import { useEffect, useRef } from "react";
import type { Play, GameInfo } from "@/pages/Index";

interface LiveFeedProps {
  currentPlay: Play | null;
  recentPlays: Play[]; // newest first
  gameInfo: GameInfo;
}

function actionColor(action: string): string {
  if (action.includes("MADE")) return "text-green-400";
  if (action === "TURNOVER") return "text-red-400";
  if (action === "STEAL") return "text-yellow-400";
  if (action === "FOUL") return "text-orange-400";
  if (action === "MISS" || action === "FT MISS") return "text-gray-500";
  if (action === "REBOUND") return "text-blue-300";
  if (action === "BLOCK") return "text-purple-400";
  if (action === "TIMEOUT") return "text-cyan-400";
  if (action === "FT MADE") return "text-green-300";
  return "text-gray-400";
}

function teamColor(team: string, gi: GameInfo): string {
  if (team === gi.homeTeam) return "text-purple-400";
  if (team === gi.awayTeam) return "text-gray-300";
  return "text-gray-500";
}

// Strip stat annotations like "(6 PTS)", "(Off:1 Def:0)", etc. from descriptions
function cleanDescription(desc: string): string {
  return desc.replace(/\s*\([^)]*\d[^)]*\)\s*/g, " ").trim();
}

export default function LiveFeed({ currentPlay, recentPlays, gameInfo }: LiveFeedProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
  }, [currentPlay?.idx]);

  return (
    <div className="terminal-panel p-3 h-full flex flex-col overflow-hidden bg-gray-950">
      {/* Header */}
      <div className="mb-2 pb-1.5 border-b border-gray-600 shrink-0">
        <div className="terminal-glow text-sm font-bold">&gt; Live Feed</div>
      </div>

      {/* Scoreboard */}
      <div className="mb-2 p-2 bg-gray-900/60 border border-gray-600 shrink-0">
        <div className="flex items-center justify-between text-center">
          <div className="flex-1">
            <div className="text-[10px] text-purple-400 font-bold">{gameInfo.homeTeam}</div>
            <div className="text-xl font-bold text-purple-400">{currentPlay?.homeScore ?? 0}</div>
          </div>
          <div className="px-3">
            <div className="text-[10px] text-gray-600">Q{currentPlay?.period ?? 1}</div>
            <div className="text-sm font-bold text-gray-500">{currentPlay?.clock ?? "12:00"}</div>
          </div>
          <div className="flex-1">
            <div className="text-[10px] text-gray-300 font-bold">{gameInfo.awayTeam}</div>
            <div className="text-xl font-bold text-gray-300">{currentPlay?.awayScore ?? 0}</div>
          </div>
        </div>
        {(currentPlay?.sasRun ?? 0) >= 5 && (
          <div className="mt-1 pt-1 border-t border-gray-700 text-center">
            <span className="text-[10px] text-red-400 animate-pulse font-bold">
              {gameInfo.awayTeam} {currentPlay!.sasRun}-0 RUN
            </span>
          </div>
        )}
        {(currentPlay?.sacRun ?? 0) >= 5 && (
          <div className="mt-1 pt-1 border-t border-gray-700 text-center">
            <span className="text-[10px] text-green-400 font-bold">
              {gameInfo.homeTeam} {currentPlay!.sacRun}-0 RUN
            </span>
          </div>
        )}
      </div>

      {/* Scrollable play log */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-1 text-xs font-mono min-h-0">
        {/* Current play — highlighted */}
        {currentPlay && (
          <div className="mb-1 p-2 bg-gray-900/60 border border-gray-600 animate-pulse-glow">
            <div className="terminal-glow font-bold text-xs">
              [{currentPlay.clock}] Q{currentPlay.period} |{" "}
              <span className={teamColor(currentPlay.team, gameInfo)}>{currentPlay.team}</span>
            </div>
            <div className="ml-3 text-gray-300 text-xs mt-0.5">
              <span className={actionColor(currentPlay.action)}>{currentPlay.action}</span>{" "}
              {cleanDescription(currentPlay.description)}
            </div>
            {currentPlay.isMomentumShift && (
              <div className="ml-3 mt-0.5 text-orange-400 text-[9px] font-bold animate-pulse">
                MOMENTUM SHIFT
              </div>
            )}
          </div>
        )}

        {/* Previous plays */}
        {recentPlays.slice(1).map((play) => (
          <div
            key={play.idx}
            className="py-0.5 text-[11px] border-l-2 border-gray-700/40 pl-2 text-gray-500 opacity-50"
          >
            [{play.clock}] Q{play.period} |{" "}
            <span className={teamColor(play.team, gameInfo) + " opacity-60"}>{play.team}</span>{" "}
            <span className={actionColor(play.action) + " opacity-60"}>{play.action}</span>{" "}
            {cleanDescription(play.description)}
          </div>
        ))}
      </div>
    </div>
  );
}
