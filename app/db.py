import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker 
from datetime import datetime
from sqlalchemy.dialects.postgresql import insert
from app.config import DATABASE_URL
import json

engine = sa.create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

metadata = sa.MetaData()


psych_profiles = sa.Table(
    "psych_profiles", metadata,
    sa.Column("user_id", sa.Integer, primary_key=True),
    sa.Column("profile", sa.JSON),   # JSONB in Postgres
    sa.Column("updated_at", sa.DateTime, default=datetime.utcnow)
)

conversations = sa.Table(
    "conversations", metadata,
    sa.Column("user_id", sa.Integer),
    sa.Column("session_id", sa.String),
    sa.Column("messages", sa.JSON),        # JSONB
    sa.Column("habit_goal", sa.String),
    sa.Column("micro_habit", sa.String),
    sa.Column("barriers", sa.ARRAY(sa.Text))  # TEXT[]
)

# --- Dependency helper ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- CRUD functions ---
def save_psych_profile(user_id: int, profile: dict):
    """Insert or update a psych profile for a user."""
    with SessionLocal() as db:
        stmt = insert(psych_profiles).values(
            user_id=user_id,
            profile=profile
        ).on_conflict_do_update(
            index_elements=["user_id"],
            set_=dict(profile=profile, updated_at=datetime.utcnow())
        )
        db.execute(stmt)
        db.commit()

def save_conversation(user_id: int, session_id: str, messages: list,
                      goal: str = None, micro: str = None, barriers: list = None):
    """Insert a conversation record."""
    with SessionLocal() as db:
        stmt = conversations.insert().values(
            user_id=user_id,
            session_id=session_id,
            messages=messages,   # dict/list → JSONB
            habit_goal=goal,
            micro_habit=micro,
            barriers=barriers    # list of strings → TEXT[]
        )
        db.execute(stmt)
        db.commit()

def get_psych_profile(user_id: int):
    """Fetch a psych profile for a user."""
    with SessionLocal() as db:
        stmt = sa.select(psych_profiles.c.profile).where(psych_profiles.c.user_id == user_id)
        result = db.execute(stmt).fetchone()
        return result[0] if result else None


