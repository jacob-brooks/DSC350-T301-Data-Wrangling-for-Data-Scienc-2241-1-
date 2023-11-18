# %%
import sqlite3

with sqlite3.connect(":memory:") as con:
    c = con.cursor()
    c.execute('''CREATE TABLE sensors (date text, city text, code text, sensor_id real, temperature real)''')

    for table in c.execute("SELECT name FROM sqlite_master WHERE type = 'table'"):
        print("Table", table[0])

    c.execute("INSERT INTO sensors VALUES ('2016-11-05','Utrecht','Red',42,15.14)")
    c.execute("SELECT * FROM sensors")
    print(c.fetchone())
    con.execute("DROP TABLE sensors")

    print("# of tables", c.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'").fetchone()[0])

    c.close()

# %%
import statsmodels.api as sm
from pandas.io.sql import read_sql
import sqlite3

with sqlite3.connect(":memory:") as con:
    c = con.cursor()

    data_loader = sm.datasets.sunspots.load_pandas()
    df = data_loader.data
    rows = [tuple(x) for x in df.values]

    con.execute("CREATE TABLE sunspots(year, sunactivity)")
    con.executemany("INSERT INTO sunspots(year, sunactivity) VALUES (?, ?)", rows)
    c.execute("SELECT COUNT(*) FROM sunspots")
    print(c.fetchone())
    print("Deleted", con.execute("DELETE FROM sunspots where sunactivity > 20").rowcount, "rows")

    print(read_sql("SELECT * FROM sunspots where year < 1732", con))
    con.execute("DROP TABLE sunspots")

    c.close()

# %%
pip install sqlalchemy

# %%
import os
import time

# Delete the existing database file if it exists
db_file = 'demo.db'
if os.path.exists(db_file):
    # Close the database connection
    session.close()
    engine.dispose()

    # Wait for a longer duration to allow processes to release the file
    time.sleep(10)  # You can increase the sleep duration

    try:
        os.remove(db_file)
        print(f"Deleted {db_file}")
    except PermissionError:
        print(f"Could not delete {db_file}. Please try again.")


# %%
from sqlalchemy import Column, ForeignKey, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy import UniqueConstraint

Base = declarative_base()

class Station(Base):
    __tablename__ = 'station'
    id = Column(Integer, primary_key=True)
    name = Column(String(14), nullable=False)  # Remove unique=True

    def __repr__(self):
        return "Id=%d name=%s" % (self.id, self.name)

class Sensor(Base):
    __tablename__ = 'sensor'
    id = Column(Integer, primary_key=True)
    last = Column(Integer)
    multiplier = Column(Float)
    station_id = Column(Integer, ForeignKey('station.id'))
    station = relationship(Station)

    def __repr__(self):
        return "Id=%d last=%d multiplier=%.1f station_id=%d" % (self.id, self.last, self.multiplier, self.station_id)

if __name__ == "__main__":
    print("This script is used by code further down in this notebook.")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from pandas.io.sql import read_sql

def populate(engine):
    Base.metadata.bind = engine

    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    de_bilt = Station(name='De Bilt')
    session.add(de_bilt)
    session.add(Station(name='Utrecht'))
    session.commit()
    print("Station", de_bilt)

    temp_sensor = Sensor(last=20, multiplier=.1, station=de_bilt)
    session.add(temp_sensor)
    session.commit()
    print("Sensor", temp_sensor)

if __name__ == "__main__":
    print("This script is used by code further down in this notebook")

engine = create_engine('sqlite:///demo.db')
Base.metadata.create_all(engine)
populate(engine)
Base.metadata.bind = engine
DBSession = sessionmaker()
DBSession.bind = engine
session = DBSession()

station = session.query(Station).first()

print("Query 1", session.query(Station).all())
print("Query 2", session.query(Sensor).all())
print("Query 3", session.query(Sensor).filter(Sensor.station == station).one())
print(read_sql("SELECT * FROM station", engine.raw_connection()))

try:
    os.remove('demo.db')
    print("Deleted demo.db")
except OSError:
    pass


# %%
pip install dataset

# %%
import dataset
from pandas.io.sql import read_sql
from pandas.io.sql import to_sql
import statsmodels.api as sm

db = dataset.connect('sqlite:///:memory:')
table = db["books"]
table.insert(dict(title="NumPy Beginner's Guide", author='Ivan Idris'))
table.insert(dict(title="NumPy Cookbook", author='Ivan Idris'))
table.insert(dict(title="Learning NumPy", author='Ivan Idris'))
print(read_sql('SELECT * FROM books', db.executable.raw_connection()) )

data_loader = sm.datasets.sunspots.load_pandas()
df = data_loader.data
#write_frame(df, "sunspots", db.executable.raw_connection()) 
df.to_sql("sunspots", db.executable.raw_connection()) 

table = db['sunspots']

for row in table.find(_limit=5):
   print(row)

print("Tables", db.tables)

# %%
pip install pymongo

# %%
from pymongo import MongoClient
import statsmodels.api as sm
import json
import pandas as pd

client = MongoClient()
db = client.test_database

data_loader = sm.datasets.sunspots.load_pandas()
df = data_loader.data
rows = json.loads(df.T.to_json()).values()
db.sunspots.insert_many(rows)

cursor = db['sunspots'].find({})
df =  pd.DataFrame(list(cursor))
print(df)

db.drop_collection('sunspots')

# %%
pip install redis

# %%
import redis
import statsmodels.api as sm
import pandas as pd

r = redis.StrictRedis()
data_loader = sm.datasets.sunspots.load_pandas()
df = data_loader.data
data = df.T.to_json()
r.set('sunspots', data)
blob = r.get('sunspots')
print(pd.read_json(blob))

# %%
pip install memcache

# %%
import memcache
import statsmodels.api as sm
import pandas as pd

client = memcache.Client([('127.0.0.1', 11211)])
data_loader = sm.datasets.sunspots.load_pandas()
df = data_loader.data
data = df.T.to_json()
client.set('sunspots', data, time=600)
print("Stored data to memcached, auto-expire after 600 seconds")
blob = client.get('sunspots')
print(pd.read_json(blob))

# %%
pip install cassandra

# %%
from cassandra import ConsistencyLevel
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement
import statsmodels.api as sm

cluster = Cluster()
session = cluster.connect()
session.execute("CREATE KEYSPACE IF NOT EXISTS mykeyspace WITH REPLICATION = { 'class' : 'SimpleStrategy', 'replication_factor' : 1 };")
session.set_keyspace('mykeyspace')
session.execute("CREATE TABLE IF NOT EXISTS sunspots (year decimal PRIMARY KEY, sunactivity decimal);")

query = SimpleStatement(
    "INSERT INTO sunspots (year, sunactivity) VALUES (%s, %s)",
    consistency_level=ConsistencyLevel.QUORUM)

data_loader = sm.datasets.sunspots.load_pandas()
df = data_loader.data
rows = [tuple(x) for x in df.values]
for row in rows:
    session.execute(query, row)

rows=session.execute("SELECT COUNT(*) FROM sunspots")
for row in rows:
    print(row)

session.execute('DROP KEYSPACE mykeyspace') 
cluster.shutdown()


