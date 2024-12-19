import os
import numpy as np
import pandas as pd
import json
from utils import *
from tkinter import Tk
from tkinter.filedialog import askopenfilename

if __name__ == "__main__":
    with open('properties.json', 'r') as json_file:
        properties = json.load(json_file)

    Tk().withdraw()  
    file_path = askopenfilename(
        title="Select a .npz file",
        filetypes=[("NPZ files", "*.npz")],
        initialdir=os.path.join(os.getcwd(), "sample_datasets")
    )

    if file_path:
        data = np.load(file_path, allow_pickle=True)
        print(f"Loaded {os.path.basename(file_path)} with keys: {data.files}")
        
        vGait = data['pose_data']
        kpt_labels = data['kpt_labels'].tolist()
        vGait = filter_data(vGait, properties['fps'], properties['filter_cutoff'], properties['filter_order'], properties['max_gap'])
        vGait_events = step_detection(vGait, kpt_labels, properties)
        vGait_parameters = gait_analysis(vGait, vGait_events, kpt_labels, properties)
        display_results(vGait_parameters)
    else:
        print("No file selected.")