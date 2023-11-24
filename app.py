import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
from run_sam import get_mask

def segment(image_path, pos_positions, neg_positions):
    print(f"Segmenting {image_path} with {len(pos_positions)} positive and {len(neg_positions)} negative points.")
    get_mask(image_path, pos_positions, neg_positions)


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Annotation Tool")
        
        # GUI Widgets
        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()
        
        self.canvas = tk.Canvas(root)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.pos_button = tk.Button(root, text="Positive Point", command=lambda: self.set_mode('pos'))
        self.pos_button.pack(side=tk.LEFT)
        
        self.neg_button = tk.Button(root, text="Negative Point", command=lambda: self.set_mode('neg'))
        self.neg_button.pack(side=tk.LEFT)
        
        # TODO: Implement undo button
        #self.undo_button = tk.Button(root, text="Undo", command=self.undo)
        #self.undo_button.pack(side=tk.RIGHT)
        
        self.segment_button = tk.Button(root, text="Segment", command=self.perform_segmentation)
        self.segment_button.pack(side=tk.RIGHT)
        
        # Event bindings
        self.canvas.bind("<Button-1>", self.place_point)
        
        # State
        self.image_path = None
        self.image = None
        self.tk_image = None
        self.pos_positions = []
        self.neg_positions = []
        self.mode = None

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if not file_path:
            return
        self.image_path = file_path
        self.image = Image.open(file_path).convert("RGB")
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
    def set_mode(self, mode):
        self.mode = mode

    def place_point(self, event):
        if not self.mode or not self.image:
            return
        
        if self.mode == 'pos':
            color = "blue"
            self.pos_positions.append((event.x, event.y))
        else:
            color = "red"
            self.neg_positions.append((event.x, event.y))
        
        draw = ImageDraw.Draw(self.image)
        draw.ellipse([event.x-5, event.y-5, event.x+5, event.y+5], fill=color)
        
        # Update displayed image
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def undo(self):
        if not self.pos_positions and not self.neg_positions:
            return

        # Refresh the image and re-draw all points except the last one
        self.image = Image.open(self.image_path).convert("RGB")
        draw = ImageDraw.Draw(self.image)
        if self.mode == 'pos' and self.pos_positions:
            self.pos_positions.pop()
        elif self.mode == 'neg' and self.neg_positions:
            self.neg_positions.pop()
        
        for point in self.pos_positions:
            draw.ellipse([point[0]-5, point[1]-5, point[0]+5, point[1]+5], fill="blue")
        for point in self.neg_positions:
            draw.ellipse([point[0]-5, point[1]-5, point[0]+5, point[1]+5], fill="red")

        # Update displayed image
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def perform_segmentation(self):
        if not self.image_path:
            messagebox.showerror("Error", "No image uploaded!")
            return
        segment(self.image_path, self.pos_positions, self.neg_positions)
        #messagebox.showinfo("Info", "Segmentation complete!")
            
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
