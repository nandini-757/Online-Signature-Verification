from pymongo import MongoClient

MONGO_URI = "your repo"

client = MongoClient(MONGO_URI)

db = client["osv_db"]
users = db["users"]
