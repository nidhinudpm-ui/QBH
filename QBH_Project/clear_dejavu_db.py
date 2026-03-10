import mysql.connector
from config import DEJAVU_DB_CONFIG

def clear_db():
    print("Clearing Dejavu database tables...")
    db_conf = DEJAVU_DB_CONFIG["database"]
    
    conn = mysql.connector.connect(
        host=db_conf["host"],
        user=db_conf["user"],
        password=db_conf["password"],
        database=db_conf["database"]
    )
    cursor = conn.cursor()
    
    # Disable foreign key checks if any (Dejavu doesn't use them usually but good practice)
    cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
    
    tables = ["fingerprints", "songs"]
    for table in tables:
        print(f"Dropping table: {table}")
        cursor.execute(f"DROP TABLE IF EXISTS `{table}`;")
    
    cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
    conn.commit()
    cursor.close()
    conn.close()
    print("Database cleared.")

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    clear_db()
