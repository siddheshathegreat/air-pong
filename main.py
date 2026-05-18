import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Reduce resolution (faster)
cap.set(3, 640)
cap.set(4, 480)

# Ball (slow start)
ball_x, ball_y = 300, 200
ball_dx, ball_dy = 2, 2
radius = 10

# Paddle
paddle_x = 250
paddle_width = 100
score = 0

prev_paddle_x = paddle_x

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # ---------------- BIG ROI (FULL WIDTH) ----------------
    roi = frame[50:350, 0:w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 40, 60])
    upper = np.array([20, 150, 255])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)

        if cv2.contourArea(cnt) > 6000:
            x, y, cw, ch = cv2.boundingRect(cnt)

            # Smooth paddle movement (full width mapping)
            target_x = x
            paddle_x = int(0.7 * prev_paddle_x + 0.3 * target_x)
            prev_paddle_x = paddle_x

            cv2.rectangle(roi, (x, y), (x+cw, y+ch), (0,255,0), 2)

    # ---------------- BALL MOVEMENT ----------------
    ball_x += int(ball_dx)
    ball_y += int(ball_dy)

    # Wall collision
    if ball_x <= 0 or ball_x >= w:
        ball_dx *= -1
    if ball_y <= 0:
        ball_dy *= -1

    # Paddle collision
    if (h - 40 <= ball_y <= h - 30) and (paddle_x <= ball_x <= paddle_x + paddle_width):
        ball_dy *= -1
        score += 1

        # Increase difficulty gradually
        ball_dx *= 1.05
        ball_dy *= 1.05

    # Game over
    if ball_y > h:
        cv2.putText(frame, "GAME OVER", (180, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
        cv2.imshow("Air Pong", frame)
        cv2.waitKey(2000)
        break

    # ---------------- DRAW ----------------
    cv2.circle(frame, (ball_x, ball_y), radius, (0,255,255), -1)

    cv2.rectangle(frame,
                  (paddle_x, h - 30),
                  (paddle_x + paddle_width, h - 20),
                  (255,0,0), -1)

    # Score
    cv2.putText(frame, f"Score: {score}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # BIG ROI BOX
    cv2.rectangle(frame, (0,50), (w,350), (255,0,0), 2)

    cv2.imshow("Air Pong Game", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(10) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()