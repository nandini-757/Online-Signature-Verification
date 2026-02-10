from pymongo import MongoClient



client = MongoClient(MONGO_URI)

db = client["osv_db"]
users = db["users"]
