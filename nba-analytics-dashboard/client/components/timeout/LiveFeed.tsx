import { useState, useEffect } from "react";

interface GameEvent {
  id: number;
  time: string;
  quarter: number;
  team: string;
  action: string;
  description: string;
}

interface LiveFeedProps {
  gameState: {
    homeTeam: string;
    awayTeam: string;
    quarter: number;
    timeRemaining: string;
  };
}

const MOCK_EVENTS: GameEvent[] = [
  {
    id: 1,
    time: "10:24:32",
    quarter: 4,
    team: "MIA",
    action: "TURNOVER",
    description: "BAD PASS BY T. HERRO MOMENTUM SHIFT DETECTED",
  },
  {
    id: 2,
    time: "10:24:15",
    quarter: 4,
    team: "BOS",
    action: "BASKET",
    description: "J. TATUM MAKES 2PT SHOT",
  },
  {
    id: 3,
    time: "10:23:50",
    quarter: 4,
    team: "MIA",
    action: "REBOUND",
    description: "B. ADEBAYO DEFENSIVE REBOUND",
  },
  {
    id: 4,
    time: "10:23:20",
    quarter: 4,
    team: "BOS",
    action: "FOUL",
    description: "PERSONAL FOUL ON D. WHITE",
  },
  {
    id: 5,
    time: "10:22:45",
    quarter: 4,
    team: "MIA",
    action: "BASKET",
    description: "B. ADEBAYO MAKES 2PT SHOT",
  },
];

export default function LiveFeed({ gameState }: LiveFeedProps) {
  const [events, setEvents] = useState<GameEvent[]>(MOCK_EVENTS);
  const [displayCount, setDisplayCount] = useState(5);

  useEffect(() => {
    // Simulate new events coming in
    const interval = setInterval(() => {
      const newEvent: GameEvent = {
        id: Math.random(),
        time: new Date().toLocaleTimeString(),
        quarter: gameState.quarter,
        team: Math.random() > 0.5 ? gameState.homeTeam : gameState.awayTeam,
        action: ["BASKET", "FOUL", "REBOUND", "TURNOVER"][
          Math.floor(Math.random() * 4)
        ],
        description: "SIMULATED EVENT",
      };
      setEvents((prev) => [newEvent, ...prev].slice(0, 20));
    }, 8000);

    return () => clearInterval(interval);
  }, [gameState.quarter, gameState.homeTeam, gameState.awayTeam]);

  return (
    <div className="terminal-panel p-4 h-full flex flex-col overflow-hidden bg-gray-950">
      {/* Header */}
      <div className="mb-4 pb-2 border-b border-gray-600">
        <div className="terminal-glow text-sm font-bold">
          &gt; LIVE_FEED
        </div>
      </div>

      {/* Events scroll area */}
      <div className="flex-1 overflow-y-auto space-y-2 text-xs font-mono">
        {/* Current event - highlighted */}
        {events.length > 0 && (
          <div className="mb-4 p-3 bg-gray-900/60 border border-gray-600 rounded-none animate-pulse-glow">
            <div className="terminal-glow font-bold mb-1">
              [{events[0].time}] &gt; <span className="text-blue-300">Q{events[0].quarter}</span> |{" "}
              <span className="text-blue-300">{gameState.timeRemaining}</span> | {events[0].team}
            </div>
            <div className="terminal-glow ml-4 text-gray-300">
              {events[0].action} {'>>>>'} {events[0].description}
            </div>
            <div className="text-gray-500 mt-1 ml-4">
              <span className="animate-blink">▌</span>
            </div>
          </div>
        )}

        {/* Previous events - dimmed */}
        {events.slice(1, displayCount).map((event) => (
          <div
            key={event.id}
            className="text-gray-500 text-opacity-60 py-1 border-l-2 border-gray-600/50 pl-2"
          >
            <div className="text-xs">
              [{event.time}] &gt; Q{event.quarter} | {event.team}
            </div>
            <div className="text-xs ml-2">{event.action}: {event.description}</div>
          </div>
        ))}

        {/* Scrollbar indicator */}
        <div className="text-center text-gray-600 text-xs mt-4 py-2 border-t border-gray-700/50">
          ... scrolling ...
        </div>
      </div>
    </div>
  );
}
