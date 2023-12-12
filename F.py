import tkinter as tk
from tkinter import ttk
import torch
import tensorflow as tf
import numpy as np

def perform_computation(input_number):
    # PyTorch tensor operation
    pytorch_tensor = torch.tensor([input_number], dtype=torch.float32)
    pytorch_result = pytorch_tensor.pow(2).sqrt()

    # TensorFlow (Keras) model prediction
    keras_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,))
    ])
    keras_model.compile(optimizer='adam', loss='mean_squared_error')
    keras_result = keras_model.predict(np.array([input_number]))

    return pytorch_result.item(), keras_result[0][0]

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Computation App")

        self.frame = ttk.Frame(root, padding="10")
        self.frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)

        self.label = ttk.Label(self.frame, text="Enter a number:")
        self.label.grid(column=0, row=0, sticky=tk.W)

        self.number_entry = ttk.Entry(self.frame, width=10)
        self.number_entry.grid(column=1, row=0, sticky=tk.W)

        self.compute_button = ttk.Button(self.frame, text="Compute", command=self.compute)
        self.compute_button.grid(column=2, row=0, sticky=tk.W)

        self.result_label = ttk.Label(self.frame, text="")
        self.result_label.grid(column=0, row=1, columnspan=3, sticky=tk.W)

    def compute(self):
        try:
            input_number = float(self.number_entry.get())
            pytorch_result, keras_result = perform_computation(input_number)
            result_text = f"PyTorch Result: {pytorch_result:.4f}, Keras Result: {keras_result:.4f}"
            self.result_label.config(text=result_text)
        except ValueError:
            self.result_label.config(text="Invalid input. Please enter a valid number.")

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
