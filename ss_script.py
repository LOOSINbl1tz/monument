import pyautogui
import time
import uuid

print("Initiating....")
print('Get ready in 5s')
time.sleep(5)
i = 1
while True:
    time.sleep(1)
    print("Screenshot ",i)
    myScreenshot = pyautogui.screenshot()
    file_name = str(uuid.uuid4())+".png"
    myScreenshot.save('cropped/'+file_name)
    if i == 10:
        break;
    i+=1
