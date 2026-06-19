import sqlite3
import os
import json

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chatbot.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    # Ensure database directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # PDFs Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pdfs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        filepath TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'parsing', -- 'parsing', 'ready', 'error'
        error_message TEXT,
        page_count INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Schema Migration for existing databases
    try:
        cursor.execute("ALTER TABLE pdfs ADD COLUMN page_count INTEGER DEFAULT 1")
    except sqlite3.OperationalError:
        pass # Column already exists

    
    # PDF Tables (extracted by Docling)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pdf_tables (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pdf_id INTEGER,
        page_number INTEGER,
        markdown TEXT,
        FOREIGN KEY (pdf_id) REFERENCES pdfs (id) ON DELETE CASCADE
    )
    """)
    
    # Chats Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pdf_id INTEGER,
        title TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (pdf_id) REFERENCES pdfs (id) ON DELETE CASCADE
    )
    """)
    
    # Messages Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER,
        role TEXT NOT NULL, -- 'user', 'assistant'
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (chat_id) REFERENCES chats (id) ON DELETE CASCADE
    )
    """)
    
    # FTS5 Parent Sections table
    cursor.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_parent_sections USING fts5(
        pdf_id UNINDEXED,
        section_id UNINDEXED,
        heading,
        summary,
        page_numbers UNINDEXED,
        tokenize="unicode61"
    )
    """)
    
    # FTS5 Child Chunks table
    cursor.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_child_chunks USING fts5(
        pdf_id UNINDEXED,
        chunk_id UNINDEXED,
        parent_id UNINDEXED,
        content,
        page_number UNINDEXED,
        tokenize="unicode61"
    )
    """)
    
    conn.commit()
    conn.close()

# FTS5 Helper Functions
def add_fts_parent_sections(rows):
    """
    rows: List of tuples (pdf_id, section_id, heading, summary, page_numbers_str)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT INTO fts_parent_sections (pdf_id, section_id, heading, summary, page_numbers) VALUES (?, ?, ?, ?, ?)",
        rows
    )
    conn.commit()
    conn.close()

def add_fts_child_chunks(rows):
    """
    rows: List of tuples (pdf_id, chunk_id, parent_id, content, page_number)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT INTO fts_child_chunks (pdf_id, chunk_id, parent_id, content, page_number) VALUES (?, ?, ?, ?, ?)",
        rows
    )
    conn.commit()
    conn.close()

def query_fts_parent_sections(pdf_id, query_text, limit=15):
    """
    Queries FTS5 for parent section summaries matching query_text.
    """
    import re
    words = re.findall(r'\w+', query_text)
    if not words:
        return []
    match_query = " OR ".join(words)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT section_id, heading, summary, page_numbers, bm25(fts_parent_sections) as fts_score
        FROM fts_parent_sections
        WHERE pdf_id = ? AND fts_parent_sections MATCH ?
        ORDER BY fts_score ASC LIMIT ?
        """,
        (pdf_id, match_query, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        r = dict(row)
        try:
            r["page_numbers"] = json.loads(r["page_numbers"])
        except Exception:
            try:
                r["page_numbers"] = [int(x.strip()) for x in r["page_numbers"].split(",") if x.strip()]
            except Exception:
                r["page_numbers"] = []
        r["score"] = -float(r["fts_score"])
        results.append(r)
    return results

def query_fts_child_chunks(pdf_id, parent_ids, query_text, limit=30):
    """
    Queries FTS5 for child chunks matching query_text, restricted to parent_ids.
    """
    import re
    words = re.findall(r'\w+', query_text)
    if not words or not parent_ids:
        return []
    match_query = " OR ".join(words)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    placeholders = ",".join(["?"] * len(parent_ids))
    params = [pdf_id, match_query] + list(parent_ids) + [limit]
    
    cursor.execute(
        f"""
        SELECT chunk_id, parent_id, content, page_number, bm25(fts_child_chunks) as fts_score
        FROM fts_child_chunks
        WHERE pdf_id = ? AND fts_child_chunks MATCH ? AND parent_id IN ({placeholders})
        ORDER BY fts_score ASC LIMIT ?
        """,
        params
    )
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        r = dict(row)
        r["score"] = -float(r["fts_score"])
        results.append(r)
    return results

def delete_fts_data(pdf_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM fts_parent_sections WHERE pdf_id = ?", (pdf_id,))
    cursor.execute("DELETE FROM fts_child_chunks WHERE pdf_id = ?", (pdf_id,))
    conn.commit()
    conn.close()

# PDF Helper Functions
def add_pdf(filename, filepath):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO pdfs (filename, filepath, status) VALUES (?, ?, ?)",
        (filename, filepath, "parsing")
    )
    pdf_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return pdf_id

def update_pdf_status(pdf_id, status, error_message=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE pdfs SET status = ?, error_message = ? WHERE id = ?",
        (status, error_message, pdf_id)
    )
    conn.commit()
    conn.close()

def update_pdf_page_count(pdf_id, page_count):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE pdfs SET page_count = ? WHERE id = ?",
        (page_count, pdf_id)
    )
    conn.commit()
    conn.close()

def get_pdfs():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM pdfs ORDER BY created_at DESC")
    pdfs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return pdfs

def get_pdf(pdf_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM pdfs WHERE id = ?", (pdf_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

# Table Extraction Helpers
def add_pdf_table(pdf_id, page_number, markdown):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO pdf_tables (pdf_id, page_number, markdown) VALUES (?, ?, ?)",
        (pdf_id, page_number, markdown)
    )
    conn.commit()
    conn.close()

def get_pdf_tables(pdf_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM pdf_tables WHERE pdf_id = ? ORDER BY page_number ASC", (pdf_id,))
    tables = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return tables

# Chat Helper Functions
def create_chat(pdf_id, title):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chats (pdf_id, title) VALUES (?, ?)",
        (pdf_id, title)
    )
    chat_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return chat_id

def get_chats():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chats ORDER BY created_at DESC")
    chats = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return chats

def get_chat(chat_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chats WHERE id = ?", (chat_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

# Message Helper Functions
def add_message(chat_id, role, content):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
        (chat_id, role, content)
    )
    message_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return message_id

def get_chat_messages(chat_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at ASC", (chat_id,))
    messages = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return messages
