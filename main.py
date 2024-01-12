import os.path
import datetime
import pickle

import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition

import custom_util as util
from face_authenticator import FaceAuthenticator


class FaceApp:
    def __init__(self):
        self.root_window = tk.Tk()
        self.root_window.geometry("1200x520+350+100")

        self.login_button = util.get_button(self.root_window, 'Login', 'green', self.login)
        self.login_button.place(x=750, y=200)

        self.logout_button = util.get_button(self.root_window, 'Logout', 'red', self.logout)
        self.logout_button.place(x=750, y=300)

        self.register_button = util.get_button(self.root_window, 'Register New User', 'gray', self.register_new_user, fg='black')
        self.register_button.place(x=750, y=400)

        self.webcam_label = util.get_img_label(self.root_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.add_webcam(self.webcam_label)

        self.database_dir = './user_database'
        if not os.path.exists(self.database_dir):
            os.mkdir(self.database_dir)

        self.log_path = './activity_log.txt'

    def add_webcam(self, label):
        if 'video_capture' not in self.__dict__:
            self.video_capture = cv2.VideoCapture(2)

        self.display_label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.video_capture.read()

        self.current_frame = frame
        img_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=pil_img)
        self.display_label.imgtk = imgtk
        self.display_label.configure(image=imgtk)

        self.display_label.after(20, self.process_webcam)

    def login(self):
        label = FaceAuthenticator.authenticate(
            image=self.current_frame,
            model_dir='/path/to/face/models',
            device_id=0
        )

        if label == 1:
            user_name = util.recognize(self.current_frame, self.database_dir)

            if user_name in ['unknown_person', 'no_persons_found']:
                util.msg_box('Oops...', 'Unknown user. Please register as a new user or try again.')
            else:
                util.msg_box('Welcome back!', 'Welcome, {}.'.format(user_name))
                self.log_activity(user_name, 'in')

        else:
            util.msg_box('Authentication Failed!', 'You are not recognized.')

    def logout(self):
        label = FaceAuthenticator.authenticate(
            image=self.current_frame,
            model_dir='/path/to/face/models',
            device_id=0
        )

        if label == 1:
            user_name = util.recognize(self.current_frame, self.database_dir)

            if user_name in ['unknown_person', 'no_persons_found']:
                util.msg_box('Oops...', 'Unknown user. Please register as a new user or try again.')
            else:
                util.msg_box('Goodbye!', 'Goodbye, {}.'.format(user_name))
                self.log_activity(user_name, 'out')

        else:
            util.msg_box('Authentication Failed!', 'You are not recognized.')

    def register_new_user(self):
        self.register_window = tk.Toplevel(self.root_window)
        self.register_window.geometry("1200x520+370+120")

        self.accept_button = util.get_button(self.register_window, 'Accept', 'green', self.accept_register)
        self.accept_button.place(x=750, y=300)

        self.try_again_button = util.get_button(self.register_window, 'Try again', 'red', self.try_again_register)
        self.try_again_button.place(x=750, y=400)

        self.capture_label = util.get_img_label(self.register_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

        self.entry_text = util.get_entry_text(self.register_window)
        self.entry_text.place(x=750, y=150)

        self.text_label = util.get_text_label(self.register_window, 'Please input username:')
        self.text_label.place(x=750, y=70)

    def try_again_register(self):
        self.register_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(self.current_frame))
        label.imgtk = imgtk
        label.configure(image=imgtk)

        self.register_capture = self.current_frame.copy()

    def accept_register(self):
        username = self.entry_text.get(1.0, "end-1c")

        embeddings = face_recognition.face_encodings(self.register_capture)[0]

        file_path = os.path.join(self.database_dir, '{}.pickle'.format(username))
        with open(file_path, 'wb') as file:
            pickle.dump(embeddings, file)

        util.msg_box('Success!', 'User registered successfully!')
        self.register_window.destroy()

    def log_activity(self, username, status):
        with open(self.log_path, 'a') as log_file:
            log_file.write('{},{},{}\n'.format(username, datetime.datetime.now(), status))

    def start(self):
        self.root_window.mainloop()


if __name__ == "__main__":
    app = FaceApp()
    app.start()
