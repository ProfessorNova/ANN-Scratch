import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageDraw, ImageOps

from ANN import *

class DrawingApp:
    def __init__(self, network):
        self.root = tk.Tk()
        self.root.title("Digit Classifier")
        self.canvas_width = 560  # Increased canvas size
        self.canvas_height = 560
        self.canvas = Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.setup()
        self.network = network

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.button_classify = tk.Button(self.button_frame, text="Classify", command=self.classify, height=2, width=20)
        self.button_classify.pack(side=tk.LEFT, expand=True)

        self.button_clear = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas, height=2, width=20)
        self.button_clear.pack(side=tk.RIGHT, expand=True)

    def paint(self, event):
        paint_color = 'black'  # Paint color
        paint_width = 40  # Increased paint width for better visibility on larger canvas
        if self.old_x and self.old_y:
            self.canvas.create_line((self.old_x, self.old_y, event.x, event.y),
                                    width=paint_width, fill=paint_color, capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line((self.old_x, self.old_y, event.x, event.y), fill=paint_color, width=paint_width)

        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def classify(self):
        # Resize image to 28x28 for the neural network input
        resized_image = self.image.resize((28, 28), Image.LANCZOS)
        inverted_image = ImageOps.invert(resized_image)
        inverted_image.save("digit.png")
        image_array = np.array(inverted_image).reshape(1, 784) / 255.0
        # Classify the image using the neural network
        result = self.network.feed_forward(image_array)
        prediction = np.argmax(result)
        print(f"Predicted digit: {prediction}")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)

    def run(self):
        self.root.mainloop()

# Load your trained network
network = NeuralNetwork()
network.load("mnist_network.npy")

# Run the drawing and classification application
app = DrawingApp(network)
app.run()