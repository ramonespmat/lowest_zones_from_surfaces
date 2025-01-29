# main.py

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk  # For the progress bar
from PIL import Image, ImageTk  # For handling images
import webbrowser  # To open folders
from processing import find_lowest_surfaces
import threading
import queue

def resource_path(relative_path):
    """ Get the absolute path to the resource, works for dev and PyInstaller """
    import sys
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def select_input_dxf():
    file_path = filedialog.askopenfilename(
        title="Select Input DXF File",
        filetypes=[("DXF Files", "*.dxf"), ("All Files", "*.*")]
    )
    input_dxf_var.set(file_path)
    if file_path:
        # Automatically set output folder to 'output' inside the input file's directory
        input_dir = os.path.dirname(file_path)
        output_folder = os.path.join(input_dir, "output")
        output_folder_var.set(output_folder)

def progress_callback_handler(q, message):
    """Handle special messages like opening folders."""
    if message.startswith("OPEN_FOLDER:"):
        folder_path = message.replace("OPEN_FOLDER:", "").strip()
        output_link.config(
            text=f"Open Output Folder: {folder_path}", 
            fg="blue", 
            cursor="hand2"
        )
        output_link.bind("<Button-1>", lambda e: open_output_folder(folder_path))
    else:
        # Regular log messages are already handled in progress_listener
        pass

def progress_listener(q):
    """Listen to the queue and update the GUI accordingly."""
    try:
        while True:
            message = q.get_nowait()
            if message.startswith("STEP:"):
                # Update the step label
                step_text = message.replace("STEP:", "").strip()
                step_label.config(text=step_text)
                # Increment the progress bar
                progress_bar['value'] += 1
            elif message.startswith("OPEN_FOLDER:"):
                # Handle special message to open folder
                progress_callback_handler(q, message)
            else:
                # It's a log message
                log_text.insert(tk.END, message + "\n")
                log_text.see(tk.END)
    except queue.Empty:
        pass
    root.after(100, lambda: progress_listener(q))

def start_processing():
    input_dxf = input_dxf_var.get()
    # output_folder = output_folder_var.get()  # Removed output folder selection
    snap_tolerance = snap_tolerance_var.get()
    decimation_reduction = decimation_reduction_var.get()
    visualize = visualize_var.get()

    if not input_dxf or not os.path.isfile(input_dxf):
        messagebox.showerror("Error", "Please select a valid input DXF file.")
        return

    # Automatically set output folder to 'output' inside the input file's directory
    input_dir = os.path.dirname(input_dxf)
    output_folder = os.path.join(input_dir, "output")
    output_dxf = os.path.join(output_folder, "lowest_zones.dxf")

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Reset progress bar and logs
    progress_bar['value'] = 0
    step_label.config(text="Starting...")
    log_text.delete(1.0, tk.END)
    output_link.config(text="", fg="blue", cursor="hand2")
    output_link.unbind("<Button-1>")

    # Disable the Start button to prevent multiple runs
    start_btn.config(state='disabled')

    # Create a queue to communicate with the GUI
    q = queue.Queue()

    # Start the listener
    root.after(100, lambda: progress_listener(q))

    def processing_thread():
        try:
            merged_polygons = find_lowest_surfaces(
                input_dxf=input_dxf,
                output_dxf=output_dxf,
                snap_tolerance=snap_tolerance,
                decimation_reduction=decimation_reduction,
                visualize=visualize,
                progress_callback=lambda msg: q.put(msg)
            )
            
            msg_lines = ["Processing complete!"]
            msg_lines.append(f"DXF output saved to: {output_dxf}")
            if merged_polygons:
                msg_lines.append(f"Total layers merged: {len(merged_polygons)}")
            else:
                msg_lines.append("No polygons were processed.")

            for line in msg_lines:
                q.put(line)
            
            # Add clickable link to output folder
            q.put(f"OPEN_FOLDER:{output_folder}")

        except Exception as e:
            q.put(f"Error occurred during processing: {e}")
            messagebox.showerror("Error", f"An error occurred:\n{e}")
        finally:
            # Re-enable the Start button after processing
            start_btn.config(state='normal')

    # Start the processing in a separate thread
    thread = threading.Thread(target=processing_thread, daemon=True)
    thread.start()

def open_output_folder(folder_path):
    webbrowser.open(folder_path)

root = tk.Tk()
root.title("Intersecciones Automáticas")

# Top Frame for Image and Title
top_frame = tk.Frame(root, padx=10, pady=10)
top_frame.pack(fill='both')

# Load navya.png
try:
    image_path = resource_path("navya.png")
    image = Image.open(image_path)
    image = image.resize((100, 100), Image.LANCZOS)  # Resize for consistency
    img_tk = ImageTk.PhotoImage(image)
    image_label = tk.Label(top_frame, image=img_tk)
    image_label.pack()
except Exception as e:
    print(f"Error loading image: {e}")
    image_label = tk.Label(top_frame, text="navya.png not found", fg="red")
    image_label.pack()

# Title
title_label = tk.Label(top_frame, text="Intersecciones Automáticas", 
                       font=("Helvetica", 16, "bold"), fg="black")
title_label.pack()

# Main Frame
frame = tk.Frame(root, padx=10, pady=10)
frame.pack(fill='both', expand=True)

input_dxf_var = tk.StringVar()
# output_folder_var = tk.StringVar()  # Removed output folder selection

# Input for DXF File
input_dxf_label = tk.Label(frame, text="Input DXF File:")
input_dxf_label.grid(row=0, column=0, sticky='w')

input_dxf_entry = tk.Entry(frame, textvariable=input_dxf_var, width=50)
input_dxf_entry.grid(row=1, column=0, sticky='we', pady=5)

input_dxf_browse_btn = tk.Button(frame, text="Browse...", command=select_input_dxf)
input_dxf_browse_btn.grid(row=1, column=1, padx=5)

# Removed Output Folder Widgets
# output_folder_label = tk.Label(frame, text="Output Folder:")
# output_folder_label.grid(row=2, column=0, sticky='w')

# output_folder_entry = tk.Entry(frame, textvariable=output_folder_var, width=50)
# output_folder_entry.grid(row=3, column=0, sticky='we', pady=5)

# output_folder_browse_btn = tk.Button(frame, text="Browse...", command=select_output_folder)
# output_folder_browse_btn.grid(row=3, column=1, padx=5)

# -- Add two sliders for snap and decimation tolerance --
snap_tolerance_var = tk.DoubleVar(value=0.1)
decimation_reduction_var = tk.DoubleVar(value=0.5)

# Snap Tolerance Slider
snap_tolerance_label = tk.Label(frame, text="Snap Tolerance:")
snap_tolerance_label.grid(row=2, column=0, sticky='w')

snap_tolerance_scale = tk.Scale(
    frame, from_=0.1, to=50.0, resolution=0.1, 
    orient='horizontal', variable=snap_tolerance_var
)
snap_tolerance_scale.grid(row=3, column=0, sticky='we', pady=5)

# Decimation Reduction Slider
decimation_reduction_label = tk.Label(frame, text="Decimation Reduction:")
decimation_reduction_label.grid(row=4, column=0, sticky='w')

decimation_reduction_scale = tk.Scale(
    frame, from_=0.1, to=1.0, resolution=0.05, 
    orient='horizontal', variable=decimation_reduction_var
)
decimation_reduction_scale.grid(row=5, column=0, sticky='we', pady=5)

# Add a checkbox for 3D visualization
visualize_var = tk.BooleanVar()
visualize_checkbox = tk.Checkbutton(frame, text="Show 3D Visualization", variable=visualize_var)
visualize_checkbox.grid(row=6, column=0, columnspan=2, pady=5)

# Start Button
start_btn = tk.Button(frame, text="Start Processing", command=start_processing)
start_btn.grid(row=7, column=0, columnspan=2, pady=10)

# Progress Label
step_label = tk.Label(frame, text="Ready", fg="blue", font=("Helvetica", 12, "bold"))
step_label.grid(row=8, column=0, columnspan=2, pady=(0,5))

# Progress Bar
progress_bar = ttk.Progressbar(frame, orient='horizontal', length=400, mode='determinate')
progress_bar.grid(row=9, column=0, columnspan=2, pady=5)
progress_bar['maximum'] = 10  # Updated to match the number of steps

# Log Text Widget with Scrollbar
log_frame = tk.Frame(frame)
log_frame.grid(row=10, column=0, columnspan=2, pady=5)

log_scroll = tk.Scrollbar(log_frame)
log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

log_text = tk.Text(log_frame, height=10, width=60, wrap='word', yscrollcommand=log_scroll.set)
log_text.pack(side=tk.LEFT, fill='both', expand=True)
log_scroll.config(command=log_text.yview)

# Output Link Label (Initially hidden)
output_link = tk.Label(frame, text="", fg="blue", cursor="hand2")
output_link.grid(row=11, column=0, columnspan=2, pady=10)

# Configure grid to make entries expandable
frame.columnconfigure(0, weight=1)

root.mainloop()
