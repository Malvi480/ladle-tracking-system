"""
database.py
-----------
All MySQL interaction for the Ladle Tracking System.

Credentials are read from environment variables (or a .env file via
python-dotenv).  Never hardcode credentials in source files.

Schema (expected in the target database)
-----------------------------------------
CREATE TABLE live_status (
    ladle_id            INT          NOT NULL,
    board_number        INT          NOT NULL,
    camera_number       INT          NOT NULL,
    detection_timestamp DATETIME     NOT NULL,
    PRIMARY KEY (ladle_id)
);
"""

import os
import pymysql
from dotenv import load_dotenv

load_dotenv()   # reads .env if present; no-op if missing

def _get_config() -> dict:
    """Build connection config from environment variables."""
    cfg = {
        "host":     os.environ["DB_HOST"],
        "port":     int(os.environ.get("DB_PORT", 3306)),
        "user":     os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
        "db":       os.environ["DB_NAME"],
        "connect_timeout": 5,
    }
    ssl_ca = os.environ.get("DB_SSL_CA")
    if ssl_ca:
        cfg["ssl"] = {"ca": ssl_ca}
    return cfg


def test_connection() -> bool:
    """
    Verify the database is reachable.
    Returns True on success, False on any error.
    """
    try:
        print("Testing MySQL connection …")
        conn = pymysql.connect(**_get_config())
        with conn.cursor() as cur:
            cur.execute("SELECT DATABASE();")
            db_name = cur.fetchone()[0]
        conn.close()
        print(f"✅  Connected to database: {db_name}")
        return True
    except KeyError as e:
        print(f"❌  Missing environment variable: {e}")
        print("    Copy .env.example → .env and fill in credentials.")
        return False
    except Exception as e:
        print(f"❌  Database connection failed: {e}")
        return False


def upsert_live_status(board_no: int, cam_id: int, lad_id: int) -> None:
    """
    Insert or update a row in live_status.

    Uses ON DUPLICATE KEY UPDATE so the table always holds the *latest*
    known position of each ladle without accumulating stale rows.

    Args:
        board_no: ID of this machine / board.
        cam_id:   Camera number that spotted the ladle.
        lad_id:   Decoded ladle number (from colour strips).
    """
    sql = """
        INSERT INTO live_status (ladle_id, board_number, camera_number, detection_timestamp)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        ON DUPLICATE KEY UPDATE
            board_number        = VALUES(board_number),
            camera_number       = VALUES(camera_number),
            detection_timestamp = CURRENT_TIMESTAMP
    """
    conn = None
    try:
        conn = pymysql.connect(**_get_config())
        with conn.cursor() as cur:
            cur.execute(sql, (lad_id, board_no, cam_id))
        conn.commit()
        print(f"✅  DB write → Board {board_no} | Camera {cam_id} | Ladle {lad_id}")
    except Exception as e:
        print(f"❌  DB write failed: {e}")
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
