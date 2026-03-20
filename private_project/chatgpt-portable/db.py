import sqlite3

# DB 연결 (전역으로 하나만)
conn = sqlite3.connect("chat.db", check_same_thread=False)
cursor = conn.cursor()


# ✅ 테이블 초기화 (서버 시작 시 실행용)
def init_db():
    # 테이블 생성
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

    # 🔥 컬럼 존재 여부 확인
    cursor.execute("PRAGMA table_info(chats)")
    columns = [col[1] for col in cursor.fetchall()]

    if "title" not in columns:
        cursor.execute("ALTER TABLE chats ADD COLUMN title TEXT")

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

def get_all_chats():
    cursor.execute("""
        SELECT id, created_at, title
        FROM chats
        ORDER BY id DESC
    """)
    return cursor.fetchall()

def get_chat_title(chat_id):
    cursor.execute("""
        SELECT content FROM messages
        WHERE chat_id=? AND role='user'
        ORDER BY id ASC LIMIT 1
    """, (chat_id,))
    
    row = cursor.fetchone()
    return row[0][:20] if row else "새 채팅"



# 채팅 삭제
def delete_chat(chat_id):
    cursor.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
    cursor.execute("DELETE FROM chats WHERE id=?", (chat_id,))
    conn.commit()


# 제목 업데이트
def update_chat_title(chat_id, title):
    cursor.execute("UPDATE chats SET title=? WHERE id=?", (title, chat_id))
    conn.commit()


# 첫 메시지 기반 제목 생성
def generate_title(chat_id):
    cursor.execute("""
        SELECT content FROM messages
        WHERE chat_id=? AND role='user'
        ORDER BY id ASC LIMIT 1
    """, (chat_id,))
    
    row = cursor.fetchone()
    return row[0][:20] if row else "새 채팅"