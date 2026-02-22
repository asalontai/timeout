import { useMemo } from "react";
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    ResponsiveContainer,
    CartesianGrid,
    Tooltip,
    Cell,
} from "recharts";
import type { Play, GameInfo } from "@/pages/Index";

interface StatisticalModelProps {
    allPlays: Play[];
    currentPlay: Play | null;
    gameInfo: GameInfo;
}

interface StreakBucket {
    streak: string;
    winPct: number;
    count: number;
    active: boolean;
}

const STREAK_LABELS = ["0-1", "2-3", "4-5", "6-7", "8-10", "11+"];

function getStreakBucket(run: number): string {
    if (run <= 1) return "0-1";
    if (run <= 3) return "2-3";
    if (run <= 5) return "4-5";
    if (run <= 7) return "6-7";
    if (run <= 10) return "8-10";
    return "11+";
}

export default function StatisticalModel({ allPlays, currentPlay, gameInfo }: StatisticalModelProps) {
    const baseData = useMemo(() => {
        const map = new Map<string, { sum: number; count: number }>();
        for (const l of STREAK_LABELS) {
            map.set(l, { sum: 0, count: 0 });
        }

        for (const p of allPlays) {
            const run = p.sasRun;
            const prob = p.m2ProbBeneficial;
            const bucket = getStreakBucket(run);
            const entry = map.get(bucket)!;
            entry.sum += prob;
            entry.count += 1;
        }

        const result: Omit<StreakBucket, "active">[] = [];
        for (const label of STREAK_LABELS) {
            const entry = map.get(label)!;
            result.push({
                streak: label,
                winPct: entry.count > 0 ? Math.round((entry.sum / entry.count) * 100) : 0,
                count: entry.count,
            });
        }
        return result;
    }, [allPlays]);

    const activeBucket = currentPlay ? getStreakBucket(currentPlay.sasRun) : null;

    const chartData: StreakBucket[] = baseData.map((d) => ({
        ...d,
        active: d.streak === activeBucket,
    }));

    return (
        <div className="terminal-panel p-3 h-full flex flex-col bg-gray-950 overflow-hidden">
            <div className="mb-2 pb-1.5 border-b border-gray-600 shrink-0">
                <div className="terminal-glow text-sm font-bold">&gt; Statistical Model (Model B)</div>
                <div className="text-[9px] text-gray-600 mt-0.5">
                    XGBoost — P(timeout beneficial) vs opponent point streak
                </div>
            </div>

            <div className="flex-1 min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#4a4a4a" opacity={0.2} vertical={false} />
                        <XAxis
                            dataKey="streak"
                            stroke="#555"
                            style={{ fontSize: "10px" }}
                            tick={{ fill: "#999" }}
                            label={{ value: "Opponent Point Streak", position: "insideBottom", offset: -2, fill: "#666", fontSize: 9 }}
                        />
                        <YAxis
                            stroke="#555"
                            style={{ fontSize: "9px" }}
                            tick={{ fill: "#999" }}
                            width={40}
                            domain={[0, 100]}
                            tickFormatter={(v: number) => `${v}%`}
                            label={{ value: "Win %", angle: -90, position: "insideLeft", offset: 20, fill: "#666", fontSize: 9 }}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: "#111",
                                border: "1px solid #333",
                                fontSize: "11px",
                                fontFamily: "monospace",
                            }}
                            formatter={(value: number) => [`${value}%`, "P(beneficial)"]}
                            labelFormatter={(label: string) => `Streak: ${label} pts`}
                        />
                        <Bar dataKey="winPct" radius={[3, 3, 0, 0]} isAnimationActive={false}>
                            {chartData.map((entry, index) => (
                                <Cell
                                    key={`cell-${index}`}
                                    fill={entry.active ? "#a78bfa" : "#6b7280"}
                                    fillOpacity={entry.active ? 1 : 0.35}
                                    stroke={entry.active ? "#a78bfa" : "transparent"}
                                    strokeWidth={entry.active ? 2 : 0}
                                />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>

            {/* Legend */}
            <div className="flex justify-center gap-4 mt-1 text-[10px] shrink-0">
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 bg-purple-400 rounded-sm" />
                    <span className="text-purple-400">Current streak</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 bg-gray-500 opacity-40 rounded-sm" />
                    <span className="text-gray-500">Database avg</span>
                </div>
                {currentPlay && (
                    <span className="text-gray-500">
                        {gameInfo.awayTeam} run: <span className="text-white font-bold">{currentPlay.sasRun}</span> pts
                    </span>
                )}
            </div>
        </div>
    );
}
