import xml.etree.ElementTree as exml
import xml.dom.minidom

def prettify(elem):
    myxml = xml.dom.minidom.parseString(elem)
    xml_pretty_str = myxml.toprettyxml()
    return xml_pretty_str 

filename = "test.xml"
root = exml.Element("Users")
userelement = exml.Element("user")
root.append(userelement)


uid = exml.SubElement(userelement, "uid")
uid.text = "1"

FirstName = exml.SubElement(userelement, "FirstName")
FirstName.text = "testuser"

LastName = exml.SubElement(userelement, "LastName")
LastName.text = "testuser"

Email = exml.SubElement(userelement, "Email")
Email.text = "testuser@hotmail.com"

state = exml.SubElement(userelement, "state")
state.text = "xyz"

location = exml.SubElement(userelement, "location")
location.text = "abc"

tree = exml.ElementTree(root)
xmlstr = exml.tostring(root, encoding='utf8', method='xml')
prettyXML = prettify(xmlstr)

f = open(filename, 'w')
f.write(prettyXML)
f.close()
