from dataclasses import dataclass

@dataclass
class DBConfig:
    host: str = "localhost"
    port: int = 5433
    database: str = "football_db"
    user: str = "admin"
    password: str = "admin"
    
    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def alembic_url(self) -> str:
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


db_config = DBConfig()