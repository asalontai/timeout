import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  ResponsiveContainer,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";
import type { GameInfo, MomentumPoint } from "@/pages/Index";

interface MomentumTimelineProps {
  momentumPoints: MomentumPoint[];
  gameInfo: GameInfo;
}

export default function MomentumTimeline({ momentumPoints, gameInfo }: MomentumTimelineProps) {
  const chartData = momentumPoints.slice(-50);
  const maxAbs = Math.max(...momentumPoints.map((p) => Math.abs(p.momentum)), 0.5);

  return (
    <div className="terminal-panel p-3 h-full flex flex-col bg-gray-950 overflow-hidden">
      <div className="mb-2 pb-1.5 border-b border-gray-600 shrink-0">
        <div className="terminal-glow text-sm font-bold">&gt; Morale Analysis (Model A)</div>
        <div className="text-[9px] text-gray-600 mt-0.5">
          Above 0 = {gameInfo.homeTeam} | Below 0 = {gameInfo.awayTeam}
        </div>
      </div>

      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 5, right: 5, left: -25, bottom: 5 }}>
            <defs>
              <linearGradient id="momGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#a78bfa" stopOpacity={0.5} />
                <stop offset="45%" stopColor="#a78bfa" stopOpacity={0.05} />
                <stop offset="50%" stopColor="transparent" stopOpacity={0} />
                <stop offset="55%" stopColor="#9ca3af" stopOpacity={0.05} />
                <stop offset="100%" stopColor="#9ca3af" stopOpacity={0.35} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#4a4a4a" opacity={0.2} vertical={false} />
            <XAxis
              dataKey="label"
              stroke="#555"
              tick={{ fontSize: 0 }}
              tickLine={{ stroke: "#555" }}
              axisLine={{ stroke: "#555" }}
            />
            <YAxis
              stroke="#555"
              style={{ fontSize: "9px" }}
              tick={{ fill: "#555" }}
              width={25}
              domain={[-maxAbs * 1.2, maxAbs * 1.2]}
            />
            <Tooltip
              contentStyle={{ backgroundColor: "#111", border: "1px solid #333", fontSize: "10px", fontFamily: "monospace" }}
              formatter={(value: number) => [
                `${value > 0 ? "+" : ""}${value.toFixed(2)}`,
                value >= 0 ? gameInfo.homeTeam : gameInfo.awayTeam,
              ]}
            />
            <ReferenceLine y={0} stroke="#666" strokeDasharray="3 3" strokeWidth={1} />
            <Area
              type="monotone"
              dataKey="momentum"
              stroke="#a78bfa"
              strokeWidth={1.5}
              fill="url(#momGrad)"
              isAnimationActive={false}
              dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="flex justify-center gap-4 mt-1 text-[10px] shrink-0">
        <span className="text-purple-400">{gameInfo.homeTeam} (+)</span>
        <span className="text-gray-400">{gameInfo.awayTeam} (-)</span>
      </div>
    </div>
  );
}
