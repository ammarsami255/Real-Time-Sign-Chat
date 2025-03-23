import customtkinter as ctk
import threading
import main 

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("Real-Time sign chat")
root.geometry("300x150")
root.resizable(False, False) 

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = 300
window_height = 150

x_position = (screen_width - window_width) // 2
y_position = (screen_height - window_height) // 2

root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

def run_sign():
    threading.Thread(target=main.sign, daemon=True).start()

def run_voice():
    threading.Thread(target=main.voice_rec, daemon=True).start()

sign_btn = ctk.CTkButton(root, text="Sign Language Recognition", command=run_sign)
sign_btn.pack(pady=20)

voice_btn = ctk.CTkButton(root, text="Voice Recognition", command=run_voice)
voice_btn.pack(pady=20)

root.mainloop()
