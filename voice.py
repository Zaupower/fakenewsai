import speech_recognition as sr

def recognize_speech_from_mic():
    # Initialize recognizer class (for recognizing the speech)
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Adjusting for ambient noise, please wait...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening...")

        # Listen to the input from the user
        audio = recognizer.listen(source)

        try:
            # Recognize speech using Google Web Speech API
            print("Recognizing...")
            text = recognizer.recognize_google(audio)
            print(f"Recognized text: {text}")
            return text
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None
        except sr.UnknownValueError:
            print("Unknown error occurred")
            return None

# For testing purposes
if __name__ == '__main__':
    recognized_text = recognize_speech_from_mic()
    print(f"Recognized text: {recognized_text}")
