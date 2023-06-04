import os
import glob, random
import matplotlib.pyplot as plt
import cv2
import numpy as np
import csv
import keras

# Intialize the file new position
for i in range(1):
    File_New_Position = i

# Loading the models
# Model of the packet face
model1 = keras.models.load_model("TLVGG16FaceMask.h5")

# Model of the packet back
model2 = keras.models.load_model("TLVGG16BackMask.h5")


while True:

    # Reading the part ID sent by the PLC controller 
    emittedPart = open(
        "E:\Local Disk D_3192023833\Private\Programming\AMIT\AI and ML Diploma\Python\Projects\project Elements\process values.csv",
        "r",
    )
    
    Part = emittedPart.readlines()
    File_Position = emittedPart.tell()

    # Determine to avoid duplicate check the same part
    if File_Position >= File_New_Position:
        if Part[-1][0:2] != Part[-2][0:2]:
            print(Part[-1][0:2])

            # Loading the photo based on the emitted part
            file_path_type = []
            face_path = "E:\\Local Disk D_3192023833\\Private\\Programming\\AMIT\\AI and ML Diploma\\Python\\Projects\\Good_Bad_Syringes_Separate_Folders\\Camera2\\Testing\\face\\*.jpg"
            back_path = "E:\\Local Disk D_3192023833\\Private\\Programming\\AMIT\\AI and ML Diploma\\Python\\Projects\\Good_Bad_Syringes_Separate_Folders\\Camera2\\Testing\\back\\*.jpg"
            
            # Get photo path
            def getPath():
                if Part[-1][0:2] == "32":
                    file_path_type.append(face_path)
                elif Part[-1][0:2] == "16":
                    file_path_type.append(back_path)
                return file_path_type

            path = getPath()
            print(path)

            # Random loading the photos from specific directory based on the emitted part
            images = glob.glob(random.choice(path))
            random_image = random.choice(images)
            img = cv2.imread(random_image)
            img = cv2.resize(img, (224, 224))
            plt.imshow(img)
            img.shape

            # Camera captured Photo preprocessing before processing by the model
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            lower_blue = np.array([70, 40, 50])
            upper_blue = np.array([170, 150, 255])

            # Create a mask. Threshold the HSV image to get only blue colors
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # convert to RGB to get the correct shape
            final_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            final_img = np.expand_dims(final_img, axis=0)

            # Prediction 
            Prediction = list()

            if Part[-1][0:2] == "32":
                Prediction = model1.predict(final_img)

            elif Part[-1][0:2] == "16":
                Prediction = model2.predict(final_img)

            print(Prediction)  # array Prediction 0=False 1=True

            # Prediction Result
            if Prediction[0][0] > Prediction[0][1]:
                print("Rejected")
            else:
                print("Accepted")

            # writing the inspection result to control the robot action by the PLC controller
            Vision_Insp_Result = open(
                "E:\Local Disk D_3192023833\Private\Programming\AMIT\AI and ML Diploma\Python\Projects\project Elements\Vision_Inspection_Result.csv",
                "a",
                newline="",
            )

            if Prediction[0][0] > Prediction[0][1]:
                Action = "Reject"
            else:
                Action = "Accept"

            Result = (Prediction[0][0], Prediction[0][1], Action)
            writer = csv.writer(Vision_Insp_Result)

            writer.writerow(Result)
            Vision_Insp_Result.close()
            print(
                "Different Orientation Emitted Sheet Compared To The Previous One, Result = Processed"
            )

            # Image show for the current product inspection with the status
            img = cv2.resize(img, (250, 400))
            final_img = cv2.resize(final_img[0], (250, 400))
            cv2.imshow(f"Status : {Action}", img)
            cv2.waitKey(2000)
            cv2.imshow(f"Status : {Action}", final_img)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

        # Duplicate part detection    
        else:
            print("Duplicated Part")

            # Write duplicate part to evaluate the machine synchronization
            Duplicated_Emitted_Part = open(
                "E:\Local Disk D_3192023833\Private\Programming\AMIT\AI and ML Diploma\Python\Projects\project Elements\Duplicated_Emitted_Parts.csv",
                "a",
                newline="",
            )
            Duplication = (Part[-1][:2], "Duplicated")
            writer = csv.writer(Duplicated_Emitted_Part)

            writer.writerow(Duplication)
            Duplicated_Emitted_Part.close()

        # Set the new file position
        def getPosition():
            File_New_Position = File_Position + 4
            return File_New_Position

        File_New_Position = getPosition()
