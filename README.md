# Timeout!: NBA Timeout Optimization Intelligence

**Timeout!** is an AI-powered analytics platform built to solve one of the most debated questions in basketball: *Does calling a timeout actually help?*

Most coaches call timeouts based on "feel" when an opponent starts a run. We decided to see what the data actually says. By analyzing thousands of timeouts from real NBA play-by-play data, we built a system that predicts the statistical effectiveness of a timeout before it's even called.

---

## How it Works: The Dual-Model Approach

Basketball is a mix of statistics and psychology. To capture both, we use a hybrid ensemble of two distinct models:

### 1. The Morale Model
This model calculates an average momentum shift for every play. It isn't just looking at the score; it tracks:
- **Scoring Run Intensity**: How fast are points being scored?
- **3-Point Streaks**: The psychological "dagger" effect of back-to-back threes.
- **Margin Swings**: Rapid changes in the lead.
- **Verdict**: If MSI drops below a certain threshold, the model signals that GSW is losing the "vibes" battle and needs a timeout.

### 2. The Statistics Model (XGBoost)
This is a cold, hard macnhine learning classifier. We trained an **XGBoost** model on historical timeout data, labeling a timeout as "beneficial" if the team's performance (score differential or opponent run suppression) improved in the following 15 plays.
- **Features**: Period, clock time, score diff, opponent run size, own run size, recent FG%, and turnovers.
- **Verdict**: Predicts the probability (0-100%) that a timeout in this exact situation will lead to a positive outcome.

### The Ensemble
The final "Verdict" is a weighted average: **70% Stats + 30% Morale**. We trust the data most, but we let the "momentum feel" break the tie in close calls.

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
We built a premium, glassmorphic dashboard to visualize the findings. You can use the **Live Simulator** to input any game state and see if the AI recommends a timeout.

1. Navigate to the `nba_statistics_model` folder.
2. Open `index.html` in any modern browser.

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

Built for the **Sports Analytics Hackathon 2026**. Stop guessing. Call the timeout when it matters.
