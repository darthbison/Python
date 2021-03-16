from neo4jrestclient.client import GraphDatabase
 
db = GraphDatabase("http://localhost:7474", username="neo4j", password="neo4j")
 
# Create some nodes with labels
user = db.labels.create("User")
u1 = db.nodes.create(name="Marco")
user.add(u1)
u2 = db.nodes.create(name="Daniela")
user.add(u2)
 
beer = db.labels.create("Beer")
b1 = db.nodes.create(name="Punk IPA")
b2 = db.nodes.create(name="Hoegaarden Rosee")
b3 = db.nodes.create(name="Stella")
# You can associate a label with many nodes in one go
beer.add(b1, b2, b3)

# User-likes->Beer relationships
u1.relationships.create("likes", b1)
u1.relationships.create("likes", b2)
u2.relationships.create("likes", b1)
u1.relationships.create("hates", b3)
u2.relationships.create("likes", b3)
# Bi-directional relationship?
u1.relationships.create("friends", u2)