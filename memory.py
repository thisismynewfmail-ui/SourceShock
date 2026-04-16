"""
AutoType v1.02 — Long-Term Memory Module

Architecture (following wawawario2/long_term_memory):
  - sentence-transformers/all-MiniLM-L6-v2 for embeddings (384-dim, normalized)
  - SQLite for text storage (user_content, assistant_content, metadata)
  - numpy .npy file for embedding vectors (loaded into RAM at startup)
  - Cosine similarity = dot product (embeddings are L2-normalized)
  - Linear search across all stored embeddings per retrieval

Key fixes over previous version:
  - All vector/ID operations protected by _vec_lock (thread-safe)
  - evaluate_and_store runs SYNCHRONOUSLY (no race conditions)
  - Vector rebuild from DB if .npy gets out of sync
  - Think tags triple-stripped at every entry point
"""

import os, json, sqlite3, time, re, threading
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "memory")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "ltm.db")
VEC_PATH = os.path.join(DATA_DIR, "embeddings.npy")
CONFIG_PATH = os.path.join(DATA_DIR, "ltm_config.json")
MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_model_lock = threading.Lock()
_vec_lock = threading.Lock()
_vectors = None
_vector_ids = []


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(MODEL_NAME)
                print(f"[LTM] Model loaded: {MODEL_NAME}")
    return _model


def _encode(texts):
    return _get_model().encode(
        texts, normalize_embeddings=True,
        show_progress_bar=False, convert_to_numpy=True
    )


def _get_db():
    conn = sqlite3.connect(DB_PATH)
    existing = set()
    try:
        existing = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
    except:
        pass

    # If old schema has 'role' column (legacy), migrate to new schema
    if "role" in existing:
        print("[LTM] Detected legacy schema with 'role' column — migrating...")
        try:
            conn.execute("ALTER TABLE memories RENAME TO memories_old")
            conn.execute("""CREATE TABLE memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_content TEXT NOT NULL DEFAULT '',
                assistant_content TEXT NOT NULL DEFAULT '',
                combined_text TEXT NOT NULL DEFAULT '',
                interest_score REAL NOT NULL DEFAULT 0.5,
                timestamp REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT '',
                char_len INTEGER DEFAULT 0
            )""")
            # Try to migrate old data
            old_cols = {r[1] for r in conn.execute("PRAGMA table_info(memories_old)").fetchall()}
            if "user_content" in old_cols and "assistant_content" in old_cols:
                conn.execute("""INSERT INTO memories
                    (user_content, assistant_content, combined_text, interest_score, timestamp, created_at, char_len)
                    SELECT user_content, assistant_content, combined_text,
                           COALESCE(interest_score, 0.5), COALESCE(timestamp, 0),
                           COALESCE(created_at, ''), COALESCE(char_len, 0)
                    FROM memories_old""")
            elif "content" in old_cols:
                # Very old schema with just role+content
                conn.execute("""INSERT INTO memories (combined_text, char_len)
                    SELECT content, LENGTH(content) FROM memories_old""")
            conn.execute("DROP TABLE memories_old")
            conn.commit()
            print("[LTM] Migration complete")
            # Delete old vectors since schema changed
            if os.path.exists(VEC_PATH):
                os.remove(VEC_PATH)
                print("[LTM] Cleared old vectors (will rebuild)")
        except Exception as e:
            print(f"[LTM] Migration error: {e} — creating fresh table")
            try: conn.execute("DROP TABLE IF EXISTS memories_old")
            except: pass

    # Create table if it doesn't exist (fresh install or post-migration)
    conn.execute("""CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_content TEXT NOT NULL DEFAULT '',
        assistant_content TEXT NOT NULL DEFAULT '',
        combined_text TEXT NOT NULL DEFAULT '',
        interest_score REAL NOT NULL DEFAULT 0.5,
        timestamp REAL NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT '',
        char_len INTEGER DEFAULT 0
    )""")

    # Add any missing columns for partially-migrated schemas
    existing = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
    for col, td in {"user_content":"TEXT NOT NULL DEFAULT ''",
        "assistant_content":"TEXT NOT NULL DEFAULT ''",
        "combined_text":"TEXT NOT NULL DEFAULT ''",
        "interest_score":"REAL NOT NULL DEFAULT 0.5",
        "timestamp":"REAL NOT NULL DEFAULT 0",
        "created_at":"TEXT NOT NULL DEFAULT ''",
        "char_len":"INTEGER DEFAULT 0"}.items():
        if col not in existing:
            try: conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {td}")
            except: pass
    conn.commit()
    return conn


def _load_vectors():
    global _vectors, _vector_ids
    with _vec_lock:
        conn = _get_db()
        db_ids = [r[0] for r in conn.execute("SELECT id FROM memories ORDER BY id").fetchall()]
        conn.close()

        disk_vecs = None
        if os.path.exists(VEC_PATH):
            try: disk_vecs = np.load(VEC_PATH)
            except: disk_vecs = None

        if disk_vecs is not None and len(disk_vecs) == len(db_ids):
            _vectors = disk_vecs
            _vector_ids = db_ids
            print(f"[LTM] Loaded {len(_vectors)} embeddings")
        elif len(db_ids) == 0:
            _vectors = None
            _vector_ids = []
            print("[LTM] No memories")
        else:
            print(f"[LTM] Mismatch (vecs={len(disk_vecs) if disk_vecs is not None else 0}, db={len(db_ids)}). Rebuilding...")
            _rebuild_locked()


def _rebuild_locked():
    global _vectors, _vector_ids
    conn = _get_db()
    rows = conn.execute("SELECT id, combined_text FROM memories ORDER BY id").fetchall()
    conn.close()
    if not rows:
        _vectors = None; _vector_ids = []; return
    texts = [r[1] for r in rows]
    ids = [r[0] for r in rows]
    try:
        _vectors = _encode(texts)
        _vector_ids = ids
        np.save(VEC_PATH, _vectors)
        print(f"[LTM] Rebuilt {len(_vectors)} embeddings")
    except Exception as e:
        print(f"[LTM] Rebuild failed: {e}")
        _vectors = None; _vector_ids = []


def _save_vecs_locked():
    if _vectors is not None and len(_vectors) > 0:
        np.save(VEC_PATH, _vectors)


def strip_think(text):
    if not text: return ""
    text = re.sub(r'<think>[\s\S]*?</think>\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>[\s\S]*$', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>\s*</think>\s*', '', text)
    return text.strip()


def _time_ago(ts):
    d = time.time() - ts
    if d < 60: return "now"
    if d < 3600: return f"{int(d/60)}m"
    if d < 86400: return f"{int(d/3600)}h"
    return f"{int(d/86400)}d"

def _fmt_ts(ts):
    try: return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except: return "?"


MEMORY_HEADER = (
    "[SYSTEM MEMORY CONTEXT]\n"
    "Below are memories from previous conversations. Do not directly\n"
    "reference or quote these memories. Incorporate relevant information\n"
    "naturally as if you simply remember it.\n\n"
)
MEMORY_FOOTER = "\n[END MEMORY CONTEXT]"


class LongTermMemory:
    def __init__(self):
        self.config = {
            "enable_saving": False,
            "enable_injection": False,
            "dumb_mode": False,
            "interest_threshold": 0.30,
            "load_similarity_threshold": 0.35,
            "max_memories_to_fetch": 3,
            "memory_length_cutoff": 600,
            "min_pair_length": 40,
        }
        self._last_status = ""
        self._last_interest = None
        self._last_action = ""
        self._load_config()
        _load_vectors()

    def _load_config(self):
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH) as f: saved = json.load(f)
                if "enabled" in saved and "enable_saving" not in saved:
                    saved["enable_saving"] = saved["enabled"]
                    saved["enable_injection"] = saved["enabled"]
                self.config.update(saved)
            except: pass

    def save_config(self):
        with open(CONFIG_PATH, "w") as f: json.dump(self.config, f, indent=2)

    @property
    def last_status(self): return self._last_status
    @property
    def last_interest(self): return self._last_interest
    @property
    def last_action(self): return self._last_action

    def get_stats(self):
        conn = _get_db()
        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        total_chars = conn.execute("SELECT COALESCE(SUM(char_len),0) FROM memories").fetchone()[0]
        avg_int = conn.execute("SELECT COALESCE(AVG(interest_score),0) FROM memories").fetchone()[0]
        conn.close()
        with _vec_lock:
            vc = len(_vectors) if _vectors is not None else 0
        return {"total_memories": total, "total_chars": total_chars,
                "avg_interest": round(avg_int, 3) if avg_int else 0,
                "vector_count": vc, "model": MODEL_NAME}

    # ── STORE ─────────────────────────────────────────────────────
    def _store_pair(self, uc, ac, interest):
        """Store pair in DB + append embedding. Thread-safe."""
        global _vectors, _vector_ids
        combined = f"User: {uc}\nAssistant: {ac}"
        try:
            emb = _encode([combined])[0]
            now = time.time()
            created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn = _get_db()
            cur = conn.execute(
                "INSERT INTO memories (user_content,assistant_content,combined_text,"
                "interest_score,timestamp,created_at,char_len) VALUES(?,?,?,?,?,?,?)",
                (uc, ac, combined, interest, now, created, len(uc)+len(ac)))
            new_id = cur.lastrowid
            conn.commit(); conn.close()
            with _vec_lock:
                e2 = emb.reshape(1, -1)
                if _vectors is None or len(_vectors) == 0:
                    _vectors = e2
                else:
                    _vectors = np.vstack([_vectors, e2])
                _vector_ids.append(new_id)
                _save_vecs_locked()
            return True
        except Exception as e:
            print(f"[LTM] Store error: {e}")
            return False

    def evaluate_and_store(self, user_msg, asst_msg):
        """
        Evaluate pair and store if appropriate. SYNCHRONOUS.
        Caller can read last_action/last_interest immediately after.
        """
        if not self.config["enable_saving"]:
            self._last_action = "save_disabled"
            self._last_interest = None
            return

        uc = strip_think(user_msg or "").strip()
        ac = strip_think(asst_msg or "").strip()
        if not uc or not ac:
            self._last_action = "empty"; self._last_interest = None; return
        if len(uc) + len(ac) < self.config["min_pair_length"]:
            self._last_status = "Too short"
            self._last_interest = 0; self._last_action = "too_short"; return

        cut = self.config["memory_length_cutoff"]
        if len(uc) > cut: uc = uc[:cut] + "..."
        if len(ac) > cut: ac = ac[:cut] + "..."
        combined = f"User: {uc}\nAssistant: {ac}"

        # DUMB MODE: save everything
        if self.config.get("dumb_mode", False):
            self._last_interest = 1.0
            if self._store_pair(uc, ac, 1.0):
                self._last_status = "Dumb: saved"
                self._last_action = "saved_dumb"
            else:
                self._last_action = "error"
            return

        # SMART MODE: score interest
        with _vec_lock:
            if _vectors is None or len(_vectors) == 0:
                interest, max_sim = 1.0, 0.0
            else:
                try:
                    emb = _encode([combined])[0]
                    sims = np.dot(_vectors, emb)
                    max_sim = float(np.max(sims))
                    interest = round(1.0 - max_sim, 4)
                except Exception as e:
                    print(f"[LTM] Score error: {e}")
                    interest, max_sim = 1.0, 0.0

        self._last_interest = interest
        thresh = self.config["interest_threshold"]
        if thresh > 0 and interest < thresh:
            self._last_status = f"Skip: int={interest:.3f}<{thresh:.2f}"
            self._last_action = "skipped"
            return

        if self._store_pair(uc, ac, interest):
            self._last_status = f"Saved: int={interest:.3f}"
            self._last_action = "saved"
        else:
            self._last_action = "error"

    # ── RETRIEVE ──────────────────────────────────────────────────
    def retrieve_memories(self, query_text):
        """Retrieve relevant memories. Returns (fmt_list, raw_list)."""
        if not self.config["enable_injection"]:
            self._last_action = "inject_disabled"; return [], []

        with _vec_lock:
            if _vectors is None or len(_vectors) == 0:
                self._last_action = "no_memories"; return [], []
            vc = _vectors.copy()
            ic = list(_vector_ids)

        query_text = strip_think(query_text)
        if not query_text or len(query_text) < 5:
            self._last_action = "query_short"; return [], []

        try:
            emb = _encode([query_text])[0]
            sims = np.dot(vc, emb)
            thresh = self.config["load_similarity_threshold"]
            mf = self.config["max_memories_to_fetch"]
            top = np.argsort(sims)[::-1]

            conn = _get_db(); fmt, raw = [], []
            for idx in top:
                if len(fmt) >= mf: break
                sim = float(sims[idx])
                if sim < thresh: break
                if idx >= len(ic): continue
                mid = ic[idx]
                row = conn.execute(
                    "SELECT user_content,assistant_content,timestamp,interest_score "
                    "FROM memories WHERE id=?", (mid,)).fetchone()
                if not row: continue
                uc, ac, ts, isc = row
                fmt.append(
                    f"--- Memory ({_time_ago(ts)} ago, {_fmt_ts(ts)}) ---\n"
                    f"  User: \"{uc}\"\n  Assistant: \"{ac}\"")
                raw.append({"user":uc,"assistant":ac,"time_ago":_time_ago(ts),
                    "timestamp":_fmt_ts(ts),"similarity":round(sim,3),
                    "interest":round(isc,3),"id":mid})
            conn.close()
            self._last_action = f"injected:{len(fmt)}" if fmt else "no_match"
            return fmt, raw
        except Exception as e:
            print(f"[LTM] Retrieve error: {e}")
            self._last_action = "error"; return [], []

    def build_injection(self, query_text):
        fmt, raw = self.retrieve_memories(query_text)
        if not fmt: return None, []
        return MEMORY_HEADER + "\n\n".join(fmt) + MEMORY_FOOTER, raw

    # ── MANAGEMENT ────────────────────────────────────────────────
    def delete_memory(self, mid):
        global _vectors, _vector_ids
        conn = _get_db()
        conn.execute("DELETE FROM memories WHERE id=?", (mid,))
        conn.commit(); conn.close()
        with _vec_lock:
            if mid in _vector_ids:
                idx = _vector_ids.index(mid)
                _vector_ids.pop(idx)
                if _vectors is not None and idx < len(_vectors):
                    _vectors = np.delete(_vectors, idx, axis=0)
                _save_vecs_locked()

    def reset_all(self):
        global _vectors, _vector_ids
        conn = _get_db(); conn.execute("DELETE FROM memories"); conn.commit(); conn.close()
        with _vec_lock:
            _vectors = None; _vector_ids = []
            if os.path.exists(VEC_PATH): os.remove(VEC_PATH)
        self._last_status = "Cleared."

    def rebuild_vectors(self):
        with _vec_lock: _rebuild_locked()
        self._last_status = "Rebuilt."

    def list_memories(self, limit=100):
        conn = _get_db()
        rows = conn.execute(
            "SELECT id,user_content,assistant_content,created_at,interest_score,char_len "
            "FROM memories ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        conn.close()
        return [{"id":r[0],"user":r[1],"assistant":r[2],"created_at":r[3],
                 "interest":round(r[4],3),"char_len":r[5]} for r in rows]

    def export_text(self):
        conn = _get_db()
        rows = conn.execute(
            "SELECT user_content,assistant_content,created_at,interest_score "
            "FROM memories ORDER BY id").fetchall()
        conn.close()
        return "\n".join([f"[{ts}] int={sc:.2f}\n  User: {u}\n  Asst: {a}\n"
                          for u,a,ts,sc in rows])
