import sqlite3

# DB 연결 (전역으로 하나만)
conn = sqlite3.connect("chat.db", check_same_thread=False)
cursor = conn.cursor()


# ✅ 테이블 초기화 (서버 시작 시 실행용)
def init_db():
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER,
        role TEXT,
        content TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()


# ✅ 새 채팅 생성
def create_chat():
    cursor.execute("INSERT INTO chats DEFAULT VALUES")
    conn.commit()
    return cursor.lastrowid


# ✅ 메시지 저장
def save_message(chat_id, role, content):
    cursor.execute(
        "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
        (chat_id, role, content)
    )
    conn.commit()


# ✅ 메시지 불러오기 (대화 이어짐 핵심🔥)
def get_messages(chat_id):
    cursor.execute(
        "SELECT role, content FROM messages WHERE chat_id=? ORDER BY id",
        (chat_id,)
    )
    rows = cursor.fetchall()

    return [{"role": r, "content": c} for r, c in rows]

def get_last_chat():
    cursor.execute("SELECT id FROM chats ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    return row[0] if row else None