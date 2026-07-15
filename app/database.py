import os
import time
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./world.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class BlockModel(Base):
    __tablename__ = "blocks"
    x = Column(Integer, primary_key=True)
    y = Column(Integer, primary_key=True)
    z = Column(Integer, primary_key=True)
    type = Column(String(50), nullable=False)

class AgentModel(Base):
    __tablename__ = "agents"
    agent_id = Column(String(100), primary_key=True)
    name = Column(String(100), nullable=False)
    x = Column(Integer, nullable=False, default=0)
    y = Column(Integer, nullable=False, default=1)
    z = Column(Integer, nullable=False, default=0)
    color = Column(String(20), nullable=False, default="#f5d000")

class AgentMemoryModel(Base):
    __tablename__ = "agent_memory"
    agent_id = Column(String(100), primary_key=True)
    key = Column(String(255), primary_key=True)
    value = Column(Text, nullable=False)

def init_db():
    Base.metadata.create_all(bind=engine)

def seed_if_empty():
    db = SessionLocal()
    try:
        if db.query(BlockModel).first() is None:
            size = 24
            blocks_to_insert = []
            for x in range(-size, size + 1):
                for z in range(-size, size + 1):
                    blocks_to_insert.append(BlockModel(x=x, y=0, z=z, type="grass"))
                    blocks_to_insert.append(BlockModel(x=x, y=-1, z=z, type="dirt"))
            db.bulk_save_objects(blocks_to_insert)
            db.commit()
    finally:
        db.close()
