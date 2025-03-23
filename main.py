import os
import cv2
import numpy as np
import speech_recognition as sr
import mediapipe as mp
import tensorflow.lite as tflite

def sign():
    model_path = "model.tflite"

    if os.path.exists(model_path):
        print("File exists!")
    else:
        print("File is missing!")

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'}

    with open("predictions.txt", "a") as file:  
        while True:
            data_aux = []
            x_ = []
            y_ = []

            ret, frame = cap.read()
            if not ret:
                break

            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                for hand_landmarks in results.multi_hand_landmarks:
                    temp_data = []
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        temp_data.append(x - min(x_))
                        temp_data.append(y - min(y_))

                    if len(temp_data) == 42:
                        data_aux.extend(temp_data)

                while len(data_aux) < 84:
                    data_aux.append(0)

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                input_data = np.array([data_aux], dtype=np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])

                predicted_character = labels_dict[int(np.argmax(output_data))]

                file.write(predicted_character + "\n")
                file.flush() 

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                            cv2.LINE_AA)

            cv2.imshow('frame', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: 
                break

    cap.release()
    cv2.destroyAllWindows()
    
def voice_rec():
    WHITE_IMAGE_SIZE = (100, 100)  
    MAX_IMAGES_PER_ROW = 12  

    def char_to_filename(char):
        if 'A' <= char <= 'Z':  
            return f"{ord(char) - ord('A')}.jpeg"
        elif '0' <= char <= '9':  
            return f"{ord(char) - ord('0') + 26}.jpeg"
        elif char == " ":  
            return "WHITE_SPACE"
        else:
            return None  

    def recognize_speech():
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Speak now...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text.upper()  
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            return None
        except sr.RequestError:
            print("Could not request results, check your internet connection.")
            return None

    def resize_images(image_list, target_height):
        resized_images = []
        for img in image_list:
            aspect_ratio = img.shape[1] / img.shape[0]  
            new_width = int(target_height * aspect_ratio)
            resized_img = cv2.resize(img, (new_width, target_height))
            resized_images.append(resized_img)
        return resized_images

    def pad_images_to_same_width(images):
        max_width = max(img.shape[1] for img in images)
        padded_images = []
        for img in images:
            h, w, c = img.shape
            padding = max_width - w
            padded_img = cv2.copyMakeBorder(img, 0, 0, 0, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            padded_images.append(padded_img)
        return padded_images

    def merge_images(image_list):
        if not image_list:
            print("No images to merge.")
            return

        target_height = min(img.shape[0] for img in image_list)  
        resized_images = resize_images(image_list, target_height)

        rows = [resized_images[i:i + MAX_IMAGES_PER_ROW] for i in range(0, len(resized_images), MAX_IMAGES_PER_ROW)]

        final_image = None
        for row in rows:
            row = pad_images_to_same_width(row)  
            row_image = np.hstack(row)  

            if final_image is None:
                final_image = row_image
            else:
                max_row_width = max(final_image.shape[1], row_image.shape[1])
                
                final_image = cv2.copyMakeBorder(final_image, 0, 0, 0, max_row_width - final_image.shape[1], cv2.BORDER_CONSTANT, value=[255, 255, 255])
                row_image = cv2.copyMakeBorder(row_image, 0, 0, 0, max_row_width - row_image.shape[1], cv2.BORDER_CONSTANT, value=[255, 255, 255])

                final_image = np.vstack([final_image, row_image])  

        cv2.imshow("Full Word", final_image)
        cv2.waitKey(5000)  #5 sec to display
        cv2.destroyAllWindows()

    def display_images_from_voice(folder_path="data"):
        input_string = recognize_speech()
        if not input_string:
            return

        images = []
        for char in input_string:
            filename = char_to_filename(char)
            if filename == "WHITE_SPACE":
                white_img = np.ones((WHITE_IMAGE_SIZE[1], WHITE_IMAGE_SIZE[0], 3), dtype=np.uint8) * 255  
                images.append(white_img)
            elif filename:
                image_path = os.path.join(folder_path, filename)
                if os.path.exists(image_path):
                    img = cv2.imread(image_path)
                    if img is not None:
                        images.append(img)
                    else:
                        print(f"Error loading image: {filename}")
                else:
                    print(f"Image not found: {filename}")
            else:
                print(f"Invalid character: {char}")

        if images:
            merge_images(images)

    display_images_from_voice()

