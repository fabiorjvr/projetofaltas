CREATE TABLE IF NOT EXISTS player_season_fouls (
  id BIGSERIAL PRIMARY KEY,
  player_name TEXT NOT NULL,
  team TEXT NOT NULL,
  league TEXT NOT NULL,
  season TEXT NOT NULL,
  position TEXT NULL,
  appearances INT NULL,
  fouls INT NOT NULL DEFAULT 0,
  fouls_drawn INT NOT NULL DEFAULT 0,
  yellow_cards INT NOT NULL DEFAULT 0,
  red_cards INT NOT NULL DEFAULT 0,
  minutes INT NOT NULL DEFAULT 0,
  source TEXT NOT NULL DEFAULT 'fbref',
  scraped_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT uq_player_team_season_league UNIQUE (player_name, team, season, league)
);

CREATE INDEX IF NOT EXISTS idx_psf_team_season ON player_season_fouls(team, season);
CREATE INDEX IF NOT EXISTS idx_psf_league_season ON player_season_fouls(league, season);