from sqlalchemy import (
    Column, Integer, String, Date, DateTime, 
    ForeignKey, Text, PrimaryKeyConstraint
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

class Stadium(Base):
    __tablename__ = 'stadiums'
    
    name = Column('NAME', String(100), nullable=True)
    city = Column('CITY', String(50), nullable=True)
    country = Column('COUNTRY', String(50), nullable=True)
    capacity = Column('CAPACITY', Integer, nullable=True)
    
    __table_args__ = (
        PrimaryKeyConstraint('NAME', 'COUNTRY', name='stadiums_pk'),
    )

    teams = relationship("Team", back_populates="stadium")
    matches = relationship("Match", back_populates="stadium_ref")
    
    def __repr__(self):
        return f"<Stadium(name='{self.name}', country='{self.country}', capacity={self.capacity})>"

class Team(Base):
    __tablename__ = 'teams'
    
    team_name = Column('TEAM_NAME', String(100), primary_key=True, nullable=True)
    country = Column('COUNTRY', String(50), nullable=True)
    home_stadium = Column('HOME_STADIUM', String(100), nullable=True)  # без FK на составной ключ ПОТОМУ ЧТО ДАННЫЕ ГОВНО
    
    stadium = relationship("Stadium", back_populates="teams", primaryjoin="Stadium.name==Team.home_stadium")
    players = relationship("Player", back_populates="team")
    managers = relationship("Manager", back_populates="team")
    home_matches = relationship("Match", foreign_keys="Match.home_team", back_populates="home_team_ref")
    away_matches = relationship("Match", foreign_keys="Match.away_team", back_populates="away_team_ref")
    
    def __repr__(self):
        return f"<Team(name='{self.team_name}', country='{self.country}')>"

class Player(Base):
    __tablename__ = 'players'
    
    player_id = Column('PLAYER_ID', String, primary_key=True, nullable=True)
    first_name = Column('FIRST_NAME', String(50), nullable=True)
    last_name = Column('LAST_NAME', String(50), nullable=True)
    nationality = Column('NATIONALITY', String(50), nullable=True)
    dob = Column('DOB', Date, nullable=True)
    team = Column('TEAM', String(100), ForeignKey('teams.TEAM_NAME'), nullable=True)
    jersey_number = Column('JERSEY_NUMBER', Integer, nullable=True)
    position = Column('POSITION', String(30), nullable=True)
    height = Column('HEIGHT', Integer, nullable=True)
    weight = Column('WEIGHT', Integer, nullable=True)
    foot = Column('FOOT', String(1), nullable=True)  
    
    team_ref = relationship("Team", back_populates="players")
    goals = relationship("Goal", back_populates="player")
    
    @property
    def full_name(self):
        return f"{self.first_name or ''} {self.last_name or ''}".strip()
    
    def __repr__(self):
        return f"<Player(id={self.player_id}, name='{self.full_name}')>"

class Manager(Base):
    __tablename__ = 'managers'

    manager_id = Column('MANAGER_ID', Integer, primary_key=True, autoincrement=True, nullable=True)
    first_name = Column('FIRST_NAME', String(50), nullable=True)
    last_name = Column('LAST_NAME', String(50), nullable=True)
    nationality = Column('NATIONALITY', String(50), nullable=True)
    dob = Column('DOB', Date, nullable=True)
    team = Column('TEAM', String(100), ForeignKey('teams.TEAM_NAME'), nullable=True)
    
    team_ref = relationship("Team", back_populates="managers")
    
    @property
    def full_name(self):
        return f"{self.first_name or ''} {self.last_name or ''}".strip()
    
    def __repr__(self):
        return f"<Manager(name='{self.full_name}', team='{self.team}')>"

class Match(Base):
    __tablename__ = 'matches'
    
    match_id = Column('MATCH_ID', String, primary_key=True, nullable=True)
    season = Column('SEASON', String(20), nullable=True)
    date_time = Column('DATE_TIME', DateTime, nullable=True)
    home_team = Column('HOME_TEAM', String(100), ForeignKey('teams.TEAM_NAME'), nullable=True)
    away_team = Column('AWAY_TEAM', String(100), ForeignKey('teams.TEAM_NAME'), nullable=True)
    stadium = Column('STADIUM', String(100), nullable=True)  # без FK
    home_team_score = Column('HOME_TEAM_SCORE', Integer, nullable=True)
    away_team_score = Column('AWAY_TEAM_SCORE', Integer, nullable=True)
    penalty_shoot_out = Column('PENALTY_SHOOT_OUT', Integer, nullable=True)
    attendance = Column('ATTENDANCE', Integer, nullable=True)
    
    home_team_ref = relationship("Team", foreign_keys=[home_team], back_populates="home_matches")
    away_team_ref = relationship("Team", foreign_keys=[away_team], back_populates="away_matches")
    stadium_ref = relationship("Stadium", primaryjoin="Stadium.name==Match.stadium", back_populates="matches")
    goals = relationship("Goal", back_populates="match", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Match(id={self.match_id}, {self.home_team} {self.home_team_score}-{self.away_team_score} {self.away_team})>"

class Goal(Base):
    __tablename__ = 'goals'
    
    goal_id = Column('GOAL_ID', String, primary_key=True, nullable=True)
    match_id = Column('MATCH_ID', String, ForeignKey('matches.MATCH_ID', ondelete='CASCADE'), nullable=True)
    pid = Column('PID', String, ForeignKey('players.PLAYER_ID', ondelete='SET NULL'), nullable=True)
    duration = Column('DURATION', Integer, nullable=True)
    assist = Column('ASSIST', String, ForeignKey('players.PLAYER_ID', ondelete='SET NULL'), nullable=True)
    goal_desc = Column('GOAL_DESC', Text, nullable=True)
    
    match = relationship("Match", back_populates="goals")
    player = relationship("Player", back_populates="goals")
    
    def __repr__(self):
        return f"<Goal(match={self.match_id}, minute={self.duration}')>"
