from neo4jrestclient.client import GraphDatabase
 
db = GraphDatabase("http://localhost:7474", username="neo4j", password="neo4j")
 
# Create some nodes with labels
user = db.labels.create("User")
u1 = db.nodes.create(name="Marcus")
user.add(u1)
u2 = db.nodes.create(name="Gwen")
user.add(u2)

#people = db.labels.get("User")
#people.add(alice, bob)

#u3 = db.nodes.get(100)
u4 = db.nodes.get(name="Daniela")
#u3 = person.get(name='Marco') 
#u4 = person.get(name='Daniela')



# Bi-directional relationship?
#u1.relationships.create("friends", u2)
#u2.relationships.create("friends", u3)
#u1.relationships.create("friends", u3)
u2.relationships.create("friends", u4)
u1.relationships.create("friends", u4)
