import psycopg2

# Connect to your postgres DB
conn = psycopg2.connect("dbname=postgres user=postgres host=pgvector password=postgres")

# Open a cursor to perform database operations
cur = conn.cursor()

# Execute a query
cur.execute("CREATE EXTENSION vector;")
cur.execute("CREATE TABLE document (id serial primary key, text text not null, source text not null, embedding vector(768));")


# Commit the transaction
conn.commit()

# Close the cursor and connection to clean up
cur.close()
conn.close()