# Version: v1.4.0

import smtplib
from email.mime.text import MIMEText

def send_signal_email(subject, body, to_email="lagodigarda@gmail.com", from_email="gardasee@gmail.com"):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    try:
        # Sostituisci 'your_password_here' con la tua app password Gmail
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        # server.login(from_email, "your_password_here")  # Sblocca e inserisci la password
        # server.send_message(msg)
        server.quit()
        print(f"Email ready to be sent: {subject}")
    except Exception as e:
        print(f"Email sending failed: {e}")
