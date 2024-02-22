import redis

conn = redis.Redis(host="localhost", port=6379)

if not conn.ping():
    raise Exception("Redis unavailable")
