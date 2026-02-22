import { useState, useEffect } from "react";
import LiveFeed from "@/components/timeout/LiveFeed";
import MomentumCenter from "@/components/timeout/MomentumCenter";
import TimeoutAlgorithm from "@/components/timeout/TimeoutAlgorithm";

interface GameState {
  homeTeam: string;
  awayTeam: string;
  homeScore: number;
  awayScore: number;
  quarter: number;
  timeRemaining: string;
  momentum: number; // -100 to 100, negative = away team, positive = home team
}

export default function Index() {
  const [gameState] = useState<GameState>({
    homeTeam: "BOS",
    awayTeam: "MIA",
    homeScore: 102,
    awayScore: 98,
    quarter: 4,
    timeRemaining: "02:14",
    momentum: 15, // BOS has +15% advantage
  });

  const [terminalReady, setTerminalReady] = useState(false);

  useEffect(() => {
    // Simulate terminal boot sequence
    setTerminalReady(true);
  }, []);

  return (
    <div className="min-h-screen bg-black text-gray-300 overflow-hidden">
      {/* Three-column layout */}
      <div className="flex h-screen gap-4 p-4">
        {/* Left Sidebar - Live Feed */}
        <div className="w-1/4 flex flex-col">
          <LiveFeed gameState={gameState} />
        </div>

        {/* Center - Momentum & Score */}
        <div className="w-1/2 flex flex-col">
          <MomentumCenter gameState={gameState} />
        </div>

        {/* Right Sidebar - Timeout Algorithm */}
        <div className="w-1/4 flex flex-col">
          <TimeoutAlgorithm gameState={gameState} />
        </div>
      </div>
    </div>
  );
}
