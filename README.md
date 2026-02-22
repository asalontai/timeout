# Timeout! - NBA Timeout Optimization Intelligence

**Timeout!** is an AI-powered analytics platform built to solve one of the most debated questions in basketball: *Does calling a timeout actually help?*

Most coaches call timeouts based on "feel" when an opponent starts a run. We decided to see what the data actually says. By analyzing thousands of timeouts from real NBA play-by-play data, we built a system that predicts the statistical effectiveness of a timeout before it's even called.

---

## How it Works: The Dual-Model Approach

Decisions in sports are based on both psychology and statistics. We use a hybrid model system to quantify the psychological and statistical aspects, deducing an overall optimal time to use a time out.

### 1. The Morale Model
This model calculates an average momentum shift for every play. It isn't looking at the score, but rather certain player patterns. For example:
- **3-Point Streaks**: The psychological "dagger" effect of back-to-back threes.
- **Fouls followed by a miss**: Teams may feel discouraged after two bad possessions
- **Verdict**: Patterns are quantified based on how often they appear in winning and losing games averaged out and normalized. The result is an association between player patterns and momentum shift.

### 2. The Statistics Model (XGBoost)
 We trained an **XGBoost** model on timeout data for all NBA teams over 6 seasons, labeling a timeout as "beneficial" if the opponent's scoring momentum stopped or decreased, or the current team's performance improved after the timeout. This predicts the probability that a timeout in this exact situation will lead to a positive outcome.
 - **Verdict**: The model understands based on the point difference, streak, etc. when the best time to take a time out is purely on statistics.

### The Ensemble
Both models' conclusion is factored into a weighing algorithm that weighs one model's decision over the other based on different scenarios. As a result, a final conclusion is reached and a time out may be suggested.

---

## Getting Started

### Backend: Run the Analysis
If you want to see the models in action on a specific game (like our case study of the 2017 NBA Finals), run the Python analysis.

```bash
# 1. Install dependencies
pip install pandas numpy xgboost joblib

# 2. Run the dual-model analysis for Game 1 of the 17 Finals
python3 analyze_game1_finals.py
```

### Frontend: Interactive Dashboard
The premium dashboard is built with **React (Next.js/Vite)** and **Recharts**. It provides real-time visualization of the AI's morale analysis and ensemble recommendations.

```bash
# 1. Navigate to the dashboard directory
cd nba-analytics-dashboard

# 2. Install dependencies
pnpm install

# 3. Start the local development server
pnpm run dev
```

The dashboard will be available at `http://localhost:8080`.

---

## Key Findings
- **The "Run" Myth**: Timeouts are most effective when an opponent is on a **7-9 point run**. Waiting until a 12-0 run is often too late—the momentum has already shifted.
- **Don't Freeze Your Own Heat**: Calling a timeout during your *own* scoring run (to "rest" players) is statistically the worst mistake a coach can make, decreasing win probability significantly.
- **Timing Matters**: Timeouts in the 3rd quarter have a higher "success rate" for stopping runs than those in the 1st quarter.

---

## Tech Stack
- **Data**: NBA API (PlayByPlayV3)
- **ML**: Python, XGBoost, Scikit-learn
- **Frontend**: Vanilla HTML5, CSS3 (Glassmorphism), JavaScript (D3-inspired visualization)
- **Data Processing**: Pandas / NumPy

Built for the **Hacklytics Hackathon 2026**.
