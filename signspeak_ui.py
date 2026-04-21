import cv2
import tkinter as tk
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
import tensorflow as tf
from gtts import gTTS
import os
import threading
import pygame 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class SignSpeakDashboard:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1100x950")
        self.window.configure(bg="#ffffff")
        
        self.running = True
        pygame.mixer.init() 

        # --- 1. LOAD MODELS ---
        self.model = tf.keras.models.load_model('signspeak_model.keras')
        self.label_map = np.load('label_map.npy', allow_pickle=True).item()
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

        # --- 2. SETUP MEDIAPIPE ---
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.cumulative_word = ""

        # --- 3. UI LAYOUT ---
        main_frame = tk.Frame(window, bg="#ffffff")
        main_frame.pack(expand=True, fill="both", padx=40, pady=20)

        # Video Label
        self.label_vid = tk.Label(main_frame, bg="#f0f0f0", borderwidth=1, relief="solid")
        self.label_vid.pack(pady=(0, 10))

        self.create_label(main_frame, "Predicted Alphabet")
        self.entry_alphabet = self.create_entry(main_frame, font_size=22)

        self.btn_submit = tk.Button(main_frame, text="Submit Letter", command=self.submit_letter,
                                   font=("Arial", 12, "bold"), bg="#d1d1d1", relief="flat", height=1)
        self.btn_submit.pack(fill="x", pady=(0, 15))

        self.create_label(main_frame, "Cumulative Word")
        self.entry_word = self.create_entry(main_frame, font_size=18)

        self.btn_clear = tk.Button(main_frame, text="Clear Word", command=self.clear_data,
                                   font=("Arial", 14, "bold"), bg="#e0e0e0", relief="flat", height=2)
        self.btn_clear.pack(fill="x", pady=5)

        self.btn_gen = tk.Button(main_frame, text="Generate Sentence", command=self.generate_sentence_logic,
                                 font=("Arial", 14, "bold"), bg="#e0e0e0", relief="flat", height=2)
        self.btn_gen.pack(fill="x", pady=5)

        self.create_label(main_frame, "Generated Sentence")
        self.entry_sentence = self.create_entry(main_frame, font_size=16)

        # --- 4. ENGINE ---
        self.vid = cv2.VideoCapture(0)
        self.update_loop()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_label(self, parent, text):
        tk.Label(parent, text=text, font=("Arial", 11), bg="#ffffff", fg="#666666").pack(anchor="w", pady=(5, 2))

    def create_entry(self, parent, font_size):
        ent = tk.Entry(parent, font=("Arial", font_size), bg="#ffffff", highlightthickness=1, relief="flat")
        ent.pack(fill="x", ipady=8, pady=(0, 5))
        return ent

    def update_loop(self):
        if not self.running: return
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.detector.detect(mp_image)

            if result.hand_landmarks:
                hand_lms = result.hand_landmarks[0]
                
                # --- DRAW RED BOX ---
                x_coords = [lm.x for lm in hand_lms]
                y_coords = [lm.y for lm in hand_lms]
                x1, y1 = int(min(x_coords) * w) - 15, int(min(y_coords) * h) - 15
                x2, y2 = int(max(x_coords) * w) + 15, int(max(y_coords) * h) + 15
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "HAND", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # --- PREDICTION LOGIC ---
                wrist = hand_lms[0]
                coords = []
                for lm in hand_lms:
                    coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
                
                max_val = max(max(coords), abs(min(coords))) or 1
                normalized = [c / max_val for c in coords]
                pred = self.model.predict(np.array([normalized]), verbose=0)
                
                if np.max(pred) > 0.85:
                    self.entry_alphabet.delete(0, tk.END)
                    self.entry_alphabet.insert(0, self.inv_label_map[np.argmax(pred)])
            
            # Show processed frame in UI
            display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(display_rgb).resize((480, 320))
            imgtk = ImageTk.PhotoImage(image=img)
            self.label_vid.imgtk = imgtk
            self.label_vid.configure(image=imgtk)
            
        self.window.after(10, self.update_loop)

    def submit_letter(self):
        letter = self.entry_alphabet.get()
        if letter:
            self.cumulative_word += letter
            self.entry_word.delete(0, tk.END)
            self.entry_word.insert(0, self.cumulative_word)

    def generate_sentence_logic(self):
        raw_word = self.entry_word.get().strip()
        
        if raw_word:
            # Remove spaces completely
            clean_word = "".join(raw_word.split()).lower()

            # Display sentence
            sentence = f"The person is signing: {clean_word}"
            self.entry_sentence.delete(0, tk.END)
            self.entry_sentence.insert(0, sentence)

            # Speak naturally
            threading.Thread(target=self._speak_silent, args=(clean_word,)).start()

    def _speak_silent(self, word):
        try:
            # Add context so TTS doesn't spell letters
            speak_text = f"{word}"

            tts = gTTS(text=speak_text, lang='en', slow=False)
            tts.save("voice.mp3")
            
            pygame.mixer.music.load("voice.mp3")
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                continue
                
            pygame.mixer.music.unload()

        except Exception as e:
            print(f"Audio Error: {e}")

    def clear_data(self):
        self.cumulative_word = ""
        self.entry_word.delete(0, tk.END)
        self.entry_alphabet.delete(0, tk.END)
        self.entry_sentence.delete(0, tk.END)

    def on_closing(self):
        self.running = False
        self.vid.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignSpeakDashboard(root, "Sign Language Dashboard")
    root.mainloop()