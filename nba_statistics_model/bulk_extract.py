"""
bulk_extract.py — Pull timeout data from many teams across many seasons.

This creates a massive dataset for finding genuine, cross-team correlations.
"""

import pandas as pd
import time
import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_timeouts import extract_season_timeouts, TEAM_IDS

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'timeout_data_bulk.csv')

# Teams with historically great timeout-calling coaches + variety
# Pop (SAS), Kerr (GSW), Spo (MIA), Stevens (BOS), D'Antoni (HOU), 
# Lue (CLE), Carlisle (DAL), Budenholzer (MIL)
TEAMS = ['GSW', 'CLE', 'SAS', 'BOS', 'HOU', 'MIA', 'MIL', 'DAL']

# Multiple seasons for trend analysis
SEASONS = ['2016-17', '2017-18', '2018-19', '2019-20', '2021-22', '2022-23']

def main():
    all_data = []
    
    # Resume from existing file if it exists
    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE)
        all_data.append(existing)
        # Figure out what we've already processed
        if 'team' in existing.columns and 'season' in existing.columns:
            done = set(zip(existing['team'], existing['season']))
        else:
            done = set()
        print(f"Resuming from {len(existing)} existing records...")
    else:
        done = set()
    
    total_combos = len(TEAMS) * len(SEASONS)
    completed = len(done)
    
    for team in TEAMS:
        for season in SEASONS:
            if (team, season) in done:
                print(f"  ✓ {team} {season} already done, skipping")
                continue
            
            completed += 1
            print(f"\n{'='*60}")
            print(f"  [{completed}/{total_combos}] Extracting {team} — {season}")
            print(f"{'='*60}")
            
            team_id = TEAM_IDS[team]
            
            try:
                df = extract_season_timeouts(team_id, season, max_games=82)
                
                if len(df) > 0:
                    df['team'] = team
                    df['season'] = season
                    all_data.append(df)
                    
                    # Save after each team-season (in case of crash)
                    combined = pd.concat(all_data, ignore_index=True)
                    combined.to_csv(OUTPUT_FILE, index=False)
                    print(f"  ✓ {team} {season}: {len(df)} timeouts (total: {len(combined)})")
                else:
                    print(f"  ⚠ {team} {season}: no timeouts found")
                    
            except Exception as e:
                print(f"  ✗ {team} {season} failed: {e}")
            
            # Brief pause between team-seasons
            time.sleep(2)
    
    # Final save
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined.to_csv(OUTPUT_FILE, index=False)
        
        print(f"\n{'='*60}")
        print(f"✅ BULK EXTRACTION COMPLETE")
        print(f"   Total timeouts: {len(combined)}")
        print(f"   Teams: {combined['team'].nunique()}")
        print(f"   Seasons: {combined['season'].nunique()}")
        print(f"   Beneficial rate: {combined['beneficial'].mean()*100:.1f}%")
        print(f"   Saved to: {OUTPUT_FILE}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
