'''
skin_cancer_tkinter_final.py

Final Tkinter GUI for Skin Cancer Classification
- Monochrome design
- Left: Upload image
- Right: Results (Tumor Type + Probability)
- Responsive buttons
- Clean layout
'''

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
import base64
import io
from datetime import datetime

import torch
from PIL import Image, ImageTk
from torchvision import transforms

from ai_derm_model import OurCNN

class SkinCancerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.setup_gui()
        self.load_model()
        
        # Variables
        self.current_image = None
        self.current_image_path = None
        
    def setup_gui(self):
        self.root.title("AI Skin Dermatologist")
        self.root.geometry("800x600")
        self.root.configure(bg='#f5f5f5')
        
        # Header
        header_frame = tk.Frame(self.root, bg='#333333', height=60)
        header_frame.pack(fill='x', padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="AI Skin Dermatologist",
            font=('Arial', 20, 'bold'),
            bg='#333333',
            fg='white'
        )
        title_label.pack(pady=15)
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg='#f5f5f5')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Left side - Upload
        left_frame = tk.Frame(main_frame, bg='white', relief='solid', borderwidth=1)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Upload section
        upload_label = tk.Label(
            left_frame,
            text="Upload Image",
            font=('Arial', 16, 'bold'),
            bg='white',
            fg='#333333'
        )
        upload_label.pack(pady=20)
        
        # Image display area
        self.image_canvas = tk.Canvas(
            left_frame,
            width=350,
            height=350,
            bg='#f9f9f9',
            relief='solid',
            borderwidth=1
        )
        self.image_canvas.pack(pady=20)
        
        # Upload button
        self.upload_btn = tk.Button(
            left_frame,
            text="Select Image",
            command=self.select_image,
            font=('Arial', 12, 'bold'),
            bg='#333333',
            fg='white',
            activebackground='#555555',
            activeforeground='white',
            cursor='hand2',
            padx=30,
            pady=10
        )
        self.upload_btn.pack(pady=10)
        
        # Start button
        self.start_btn = tk.Button(
            left_frame,
            text="Start Classification",
            command=self.start_classification,
            font=('Arial', 12, 'bold'),
            bg='#333333',
            fg='white',
            activebackground='#555555',
            activeforeground='white',
            cursor='hand2',
            padx=30,
            pady=10,
            state='disabled'
        )
        self.start_btn.pack(pady=10)
        
        # Right side - Results
        right_frame = tk.Frame(main_frame, bg='white', relief='solid', borderwidth=1)
        right_frame.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        # Results section
        results_label = tk.Label(
            right_frame,
            text="Classification Results",
            font=('Arial', 16, 'bold'),
            bg='white',
            fg='#333333'
        )
        results_label.pack(pady=20)
        
        # Tumor Type
        type_frame = tk.Frame(right_frame, bg='white')
        type_frame.pack(fill='x', padx=30, pady=20)
        
        type_label = tk.Label(
            type_frame,
            text="Tumor Type:",
            font=('Arial', 14, 'bold'),
            bg='white',
            fg='#333333'
        )
        type_label.pack(side='left')
        
        self.type_result = tk.Label(
            type_frame,
            text="Unknown",
            font=('Arial', 14, 'bold'),
            bg='white',
            fg='#666666'
        )
        self.type_result.pack(side='right')
        
        # Probability
        prob_frame = tk.Frame(right_frame, bg='white')
        prob_frame.pack(fill='x', padx=30, pady=20)
        
        prob_label = tk.Label(
            prob_frame,
            text="Probability:",
            font=('Arial', 14, 'bold'),
            bg='white',
            fg='#333333'
        )
        prob_label.pack(side='left')
        
        self.prob_result = tk.Label(
            prob_frame,
            text="0.0%",
            font=('Arial', 14, 'bold'),
            bg='white',
            fg='#666666'
        )
        self.prob_result.pack(side='right')
        
        # Instructions
        instructions_frame = tk.Frame(right_frame, bg='white')
        instructions_frame.pack(fill='both', expand=True, padx=30, pady=20)
        
        instructions_label = tk.Label(
            instructions_frame,
            text="Instructions:",
            font=('Arial', 12, 'bold'),
            bg='white',
            fg='#333333'
        )
        instructions_label.pack(anchor='w', pady=(0, 10))
        
        instructions_text = """
1. Click 'Select Image' to choose a skin lesion image
2. Click 'Start Classification' to analyze the image
3. View the tumor type and probability results
4. Results will appear here after classification
        """
        
        instructions_display = tk.Label(
            instructions_frame,
            text=instructions_text.strip(),
            font=('Arial', 10),
            bg='white',
            fg='#666666',
            justify='left',
            anchor='nw'
        )
        instructions_display.pack(fill='both', expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            relief='sunken',
            anchor='w',
            bg='#e0e0e0',
            fg='#333333',
            font=('Arial', 9)
        )
        status_bar.pack(side='bottom', fill='x')
        
    def load_model(self):
        """Load the model in a separate thread"""
        self.status_var.set("Loading model...")
        
        def load():
            try:
                # Device selection
                if torch.backends.mps.is_available():
                    self.device = "mps"
                elif torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
                
                # Load model
                self.model = OurCNN().to(self.device)
                
                # Try to load fast model first, then normal model
                model_paths = [
                    '../Model/model_skin_cancer_fast.pth',
                    '../Model/model_skin_cancer.pth'
                ]
                
                model_loaded = False
                for path in model_paths:
                    if os.path.exists(path):
                        self.model.load_state_dict(torch.load(path, map_location=self.device))
                        self.model.eval()
                        model_loaded = True
                        break
                
                if model_loaded:
                    self.status_var.set(f"Model ready | Device: {self.device.upper()}")
                else:
                    self.status_var.set("No model found! Please train first.")
                    self.model = None
                    
            except Exception as e:
                self.status_var.set(f"Model load error: {str(e)}")
                self.model = None
        
        # Run in separate thread to avoid blocking GUI
        threading.Thread(target=load, daemon=True).start()
        
    def select_image(self):
        """Select and display an image"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load image
                img = Image.open(file_path).convert("RGB")
                self.current_image = img
                self.current_image_path = file_path
                
                # Resize for display
                display_img = img.copy()
                display_img.thumbnail((350, 350), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(display_img)
                
                # Display on canvas
                self.image_canvas.delete("all")
                self.image_canvas.create_image(175, 175, image=photo)
                self.image_canvas.image = photo  # Keep reference
                
                # Reset results
                self.type_result.config(text="Unknown", fg='#666666')
                self.prob_result.config(text="0.0%", fg='#666666')
                
                # Enable start button
                self.start_btn.config(state='normal')
                self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
                
    def start_classification(self):
        """Start image classification"""
        if self.model is None:
            messagebox.showerror("Model Error", "Model not loaded! Please train a model first.")
            return
            
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please select an image first!")
            return
        
        # Update button state
        self.start_btn.config(text="Processing...", state='disabled')
        self.status_var.set("Classifying image...")
        
        def classify():
            try:
                # Transform
                tfm = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                ])
                
                # Prepare image
                img_t = tfm(self.current_image).unsqueeze(0).to(self.device)
                
                # Predict
                with torch.no_grad():
                    outputs = self.model(img_t)
                    probs = torch.softmax(outputs, dim=1)
                    
                    conf, pred = torch.max(probs, dim=1)
                    confidence = conf.item()
                    
                    labels = ['Benign', 'Malignant']
                    predicted_class = labels[pred.item()]
                    
                    # Update UI in main thread
                    self.root.after(0, self.update_results, predicted_class, confidence)
                    
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Classification Error", f"Error: {str(e)}"))
            finally:
                # Reset button
                self.root.after(0, self.reset_button)
        
        # Run classification in separate thread
        threading.Thread(target=classify, daemon=True).start()
        
    def update_results(self, tumor_type, confidence):
        """Update the result display"""
        self.type_result.config(text=tumor_type, fg='#333333')
        self.prob_result.config(text=f"{confidence * 100:.1f}%", fg='#333333')
        self.status_var.set("Classification complete")
        
    def reset_button(self):
        """Reset the start button"""
        self.start_btn.config(text="Start Classification", state='normal')
        
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

if __name__ == "__main__":
    app = SkinCancerGUI()
    app.run()
