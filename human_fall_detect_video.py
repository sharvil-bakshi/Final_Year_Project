import cv2
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
import argparse

fitToEllipse = False

def send_email(image_path):
    sender_email = "spbarve4321@gmail.com"
    receiver_email = "sharvilbakshi23@gmail.com"
    password = "wlzj rscl szbw rwtj" # you can generate your own password by visit this link https://myaccount.google.com/apppasswords

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Fall Detected"

    body = "A fall has been detected. Please check the attached image for details."
    msg.attach(MIMEText(body, 'plain'))

    attachment = open(image_path, 'rb')
    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= fall_image.jpg")
    msg.attach(part)

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        exit(0)

def main():
    fall_detected = False
    cap = cv2.VideoCapture('video_fall.mp4')
    time.sleep(2)

    fgbg = cv2.createBackgroundSubtractorMOG2()
    j = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgmask = fgbg.apply(gray)

            contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                areas = [cv2.contourArea(contour) for contour in contours]
                max_area = max(areas, default=0)
                max_area_index = areas.index(max_area)
                cnt = contours[max_area_index]

                M = cv2.moments(cnt)

                x, y, w, h = cv2.boundingRect(cnt)

                cv2.drawContours(fgmask, [cnt], 0, (255, 255, 255), 3, maxLevel=0)

                if h < w:
                    j += 1

                if j > 18:
                    print("FALL")
                    cv2.putText(fgmask, 'FALL', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    # Save the frame with the fall for email attachment
                    image_path = "fall_image.jpg"
                    cv2.imwrite(image_path, frame)

                    # Send email alert
                    if not fall_detected:
                        fall_detected = True
                        send_email(image_path)
                        print('Email Sent')

                if h > w:
                    j = 0
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.imshow('video', frame)

                if cv2.waitKey(33) == 27:
                    break
        except Exception as e:
            print(f"Error: {e}")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fall Detection from Video")
    args = parser.parse_args()
    main()
