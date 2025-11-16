import sqlite3
from queue import Queue

# NOTE: Connetion pooling thoughts for async

# let pool: list[connections] = connections;
# fn api_hit(args) {
#     conn = pool.pop();
#     result = call_work_with_connection(conn);
#     return pool.push(conn); // it's a mutex would need to check locks }
# }

class SQLitePool:
    def __init__(self, db_path: str, max_size: int = 1):
        self._pool = Queue(maxsize=max_size)
        for _ in range(max_size):
            conn = sqlite3.connect(db_path, check_same_thread=False)
            self._pool.put(conn)

    def acquire(self):
        return self._pool.get()

    def release(self, conn):
        self._pool.put(conn)

    def close_all(self):
        while not self._pool.empty():
            conn = self._pool.get()
            conn.close()

# example
# pool = SQLitePool("source.db", max_size=2)

# def run_query(query: str):
#     conn = pool.acquire()
#     try:
#         cursor = conn.execute(query)
#         result = cursor.fetchall()
#     finally:
#         pool.release(conn)
#     return result
