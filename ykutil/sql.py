import sqlite3


def merge_databases(target_db: str, source_dbs: list[str]):
    """
    Merge multiple SQLite databases into one.

    Parameters:
    target_db (str): Path to the target SQLite database.
    source_dbs (list): List of paths to source SQLite databases.
    """
    # Create a new database or connect to an existing one
    target_conn = sqlite3.connect(target_db)
    target_cursor = target_conn.cursor()

    for source_db in source_dbs:
        print(f"Merging {source_db} into {target_db}...")

        # Connect to the source database
        source_conn = sqlite3.connect(source_db)
        source_cursor = source_conn.cursor()

        # Get all table names from the source database
        source_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in source_cursor.fetchall()]

        for table in tables:
            print(f"  Copying table {table}...")

            # Fetch column names
            source_cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in source_cursor.fetchall()]
            column_list = ", ".join(columns)

            # Copy data from source to target
            source_cursor.execute(f"SELECT {column_list} FROM {table}")
            rows = source_cursor.fetchall()

            # Ensure the target table exists
            target_cursor.execute(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}';"
            )
            if not target_cursor.fetchone():
                print(f"  Creating table {table} in target database...")
                source_cursor.execute(
                    f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'"
                )
                create_table_sql = source_cursor.fetchone()[0]
                target_cursor.execute(create_table_sql)

            # Insert data into the target database
            placeholders = ", ".join(["?"] * len(columns))
            target_cursor.executemany(
                f"INSERT INTO {table} ({column_list}) VALUES ({placeholders})", rows
            )

        # Commit and close the source connection
        target_conn.commit()
        source_conn.close()

    # Close the target connection
    target_conn.close()
    print("Merge completed successfully.")
