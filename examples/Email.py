import smtplib

server = smtplib.SMTP('smtp.gmail.com', 587)
server.ehlo()
server.starttls()
server.ehlo()

#Next, log in to the server
server.login("", "")

#Send the mail
SUBJECT = "This is a test"
TEXT = "Test Email\nTesting Hello" 
message = 'Subject: {}\n\n{}'.format(SUBJECT, TEXT)

server.sendmail("sender@gmail.com", "recipient@hotmail.com", message)
server.quit()
