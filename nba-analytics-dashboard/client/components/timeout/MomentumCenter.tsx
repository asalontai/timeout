import { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

interface MomentumCenterProps {
  gameState: {
    homeTeam: string;
    awayTeam: string;
    homeScore: number;
    awayScore: number;
    momentum: number;
  };
}

const SCORE_DATA = [
  { time: "14:20", away: 0, home: 0 },
  { time: "12:45", away: 5, home: 8 },
  { time: "10:30", away: 12, home: 18 },
  { time: "08:15", away: 20, home: 25 },
  { time: "06:00", away: 28, home: 35 },
  { time: "04:00", away: 40, home: 45 },
  { time: "02:14", away: 98, home: 102 },
];

export default function MomentumCenter({ gameState }: MomentumCenterProps) {
  const [animatedMomentum, setAnimatedMomentum] = useState(gameState.momentum);

  useEffect(() => {
    setAnimatedMomentum(gameState.momentum);
  }, [gameState.momentum]);

  const momentumPercent = ((animatedMomentum + 100) / 200) * 100;
  const momentumDirection =
    gameState.momentum > 0
      ? `${gameState.homeTeam} +${gameState.momentum}%`
      : gameState.momentum < 0
        ? `${gameState.awayTeam} +${Math.abs(gameState.momentum)}%`
        : "NEUTRAL";

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Top - Momentum Gauge */}
      <div className="terminal-panel p-4 flex-1 flex flex-col bg-gray-950">
        <div className="mb-4 pb-2 border-b border-gray-600">
          <div className="terminal-glow text-sm font-bold">
            &gt; GAME_MOMENTUM_GAUGE
          </div>
        </div>

        <div className="flex-1 flex flex-col justify-center space-y-6">
          {/* Advantage text */}
          <div className="text-center">
            <div className="terminal-accent font-bold text-lg animate-pulse-glow">
              [ ADVANTAGE: {momentumDirection} ]
            </div>
          </div>

          {/* Momentum bar */}
          <div className="space-y-3">
            {/* Labels */}
            <div className="flex justify-between text-xs font-bold">
              <span className="terminal-accent">{gameState.awayTeam} (AWAY)</span>
              <span className="text-gray-500">MOMENTUM</span>
              <span className="terminal-glow">{gameState.homeTeam} (HOME)</span>
            </div>

            {/* Bar visualization */}
            <div className="relative h-12 bg-gray-900/50 border border-gray-600 rounded-none overflow-hidden">
              {/* Grid lines */}
              <div className="absolute inset-0 flex opacity-20">
                {[0, 1, 2, 3, 4].map((i) => (
                  <div
                    key={i}
                    className="flex-1 border-r border-gray-600/30"
                  ></div>
                ))}
              </div>

              {/* Momentum indicator */}
              <div
                className="absolute top-0 h-full w-1 bg-gray-400 animate-pulse-glow"
                style={{ left: `${momentumPercent}%` }}
              ></div>

              {/* Center marker */}
              <div className="absolute top-0 left-1/2 h-full w-0.5 bg-gray-600/50 -ml-0.25"></div>

              {/* Pulsating glow effect */}
              <div
                className="absolute top-1/2 h-4 w-4 bg-gray-400/30 rounded-full transform -translate-y-1/2 -translate-x-1/2 animate-pulse-glow"
                style={{ left: `${momentumPercent}%` }}
              ></div>
            </div>

            {/* Percentage display */}
            <div className="text-center font-bold">
              <span className="terminal-glow text-sm">
                {Math.abs(gameState.momentum)}% ADVANTAGE{" "}
                {gameState.momentum > 0 ? gameState.homeTeam : gameState.awayTeam}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom - Score Progression Chart */}
      <div className="terminal-panel p-4 flex-1 flex flex-col bg-gray-950">
        <div className="mb-3 pb-2 border-b border-gray-600">
          <div className="terminal-glow text-sm font-bold">
            &gt; SCORE_PROGRESSION_CHART [LAST_5_MINS]
          </div>
        </div>

        <div className="flex-1 flex items-center justify-center">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={SCORE_DATA} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="#4a4a4a"
                opacity={0.3}
                vertical={false}
              />
              <XAxis
                dataKey="time"
                stroke="#666666"
                style={{ fontSize: "10px" }}
                tick={{ fill: "#666666" }}
              />
              <YAxis
                stroke="#666666"
                style={{ fontSize: "10px" }}
                tick={{ fill: "#666666" }}
                width={30}
              />
              <Line
                type="monotone"
                dataKey="home"
                stroke="#a89984"
                strokeWidth={2}
                dot={{ fill: "#a89984", r: 3 }}
                isAnimationActive={false}
                name={gameState.homeTeam}
              />
              <Line
                type="monotone"
                dataKey="away"
                stroke="#83a598"
                strokeWidth={2}
                dot={{ fill: "#83a598", r: 3 }}
                isAnimationActive={false}
                name={gameState.awayTeam}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Legend */}
        <div className="flex justify-center gap-6 mt-2 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-gray-400 rounded-none"></div>
            <span className="terminal-glow">{gameState.homeTeam}</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-300 rounded-none"></div>
            <span className="terminal-accent">{gameState.awayTeam}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
