import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from tensorflow import keras
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# Path to the trained EfficientNetB3 model (see README). Override with the
# MODEL_PATH environment variable or the first command-line argument.
MODEL_PATH = os.environ.get("MODEL_PATH") or (sys.argv[1] if len(sys.argv) > 1 else "model.h5")

model = None

def get_model():
    global model
    if model is None:
        model = keras.models.load_model(MODEL_PATH)
    return model

def classify_image(image_path):
    image = keras.preprocessing.image.load_img(image_path, target_size=(300, 440))
    image = keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    prediction = get_model().predict(image)[0]

    classes = ['Adeno carcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']
    classes = [class_name.replace(" ","\n") for class_name in classes]
    class_names = np.array(classes)
    class_idx = np.argsort(prediction)[::-1]
    class_probs = prediction[class_idx]

    top_classes = class_names[class_idx]
    top_probs = class_probs

    result_label.config(text="Predicted Class: {}".format(top_classes[0].replace("\n"," ")))


    fig, ax = plt.subplots(num="CT Scan Probabilities")
    y_pos = np.arange(len(top_classes))
    ax.barh(y_pos, top_probs, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_classes)
    ax.invert_yaxis()
    ax.set_xlabel('Probability')
    ax.set_title('Image Classification Results')
    for i, v in enumerate(top_probs):
        ax.text(v + 0.01, i, str(round(v, 2)), color='blue')

    plt.show()

def browse_button_clicked():
    plt.close("all")
    file_path = filedialog.askopenfilename(filetypes=[("Image File",'.jpg .png .jpeg')])
    if not file_path:  # dialog cancelled
        return
    file_path_entry.delete(0, tk.END)
    file_path_entry.insert(0, file_path)

    img = Image.open(file_path)
    img = img.resize((440, 300), Image.LANCZOS)
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

    file_name_label.configure(text="Selected file: " + file_path.split("/")[-1])
    result_label.config(text="")
    classify_button['state'] = "normal"

def classify_button_clicked():
    image_path = file_path_entry.get()
    try:
        classify_image(image_path)
    except (OSError, ValueError) as e:
        messagebox.showerror("Model not found", "Could not load model '{}'.\n\n{}".format(MODEL_PATH, e))

def clear_button_clicked():
    plt.close("all")
    classify_button['state'] = "disabled"
    file_name_label.config(text="")
    result_label.config(text="")
    file_path_entry.delete(0, tk.END)
    image_label.image = None

window = tk.Tk()
window.title("CT Scan Classification (research demo - not for medical use)")
window.geometry("700x550")

top_frame = tk.Frame(window)
top_frame.pack(side=tk.TOP)

file_path_entry = tk.Entry(top_frame, width=50)

browse_button = tk.Button(top_frame, text="Browse", command=browse_button_clicked)

clear_button = tk.Button(top_frame,text="Clear",command=clear_button_clicked)

image_label = tk.Label(window)

classify_button = tk.Button(window, text="Classify", command=classify_button_clicked)
classify_button['state'] = 'disabled'
result_label = tk.Label(window, text="")

file_name_label = tk.Label(window)

clear_button.pack(side=tk.RIGHT, padx=5)
browse_button.pack(side=tk.RIGHT, padx=5)
file_path_entry.pack(side=tk.LEFT,pady=10)
image_label.pack(pady=20)
file_name_label.pack()
result_label.pack(pady=20)
classify_button.pack(pady=10)

window.mainloop()
