import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pickle
import pandas as pd

def plot_frame_seq(frames_dir, save_image_dir, name, indices, total_duration, total_frames):
    """
    Generate a plot of selected frames with captions indicating their time in the video.
    
    Args:
    - frames_dir (str): Directory containing the frames.
    - indices (list): List of specific frame indices to select.
    - total_duration (float): Total duration of the video in seconds.
    - total_frames (int): Total number of frames in the video.
    """
    # Get all frame files and sort them
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')])

    # Select the frames based on indices
    selected_frames = [frame_files[i] for i in indices]

    # Calculate the time per frame
    time_per_frame = total_duration / total_frames  # seconds per frame

    # Generate captions for the selected frames
    captions = [f"Frame {i} ({i * time_per_frame:.2f}s)" for i in indices]

    # Create a plot
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i, ax in enumerate(axes.flat):
        # Read and display the image
        img = mpimg.imread(selected_frames[i])
        ax.imshow(img)
        
        # Set the caption
        ax.set_title(captions[i], fontsize=10)
        
        # Remove the axis
        ax.axis('off')

    # Adjust layout
    plt.tight_layout()


    # Save image plot
    os.makedirs(save_image_dir, exist_ok=True)
    image_path = os.path.join(save_image_dir, f'{name}.png')
    plt.savefig(image_path)
    plt.close()
    plt.show()


def plot_probs(probs_array_path, captions, save_image_dir, total_frames, total_duration, name, indices=None, moving_avg_window=None):
    """
    Plot the probabilities of captions over frames and time, and save the plot to a file.
    
    Args:
    - probs_array_path (str): Path to the pickle file containing the array of probabilities with shape (num_frames, num_captions).
    - captions (list): List of captions corresponding to the probabilities.
    - save_image_dir (str): Directory where the plot image will be saved.
    - total_frames (int): Total number of frames in the video.
    - total_duration (float): Total duration of the video in seconds.
    - name (str): Name of the file to save the plot as.
    - indices (list, optional): List of frame numbers to mark with vertical red lines.
    """
    # Load the probabilities array from the pickle file
    with open(probs_array_path, 'rb') as f:
        probs_array = pickle.load(f)
    
    # Ensure the array has the correct shape
    probs_array = np.array(probs_array)
    expected_shape = (total_frames, len(captions))
    if probs_array.shape != expected_shape:
        print(f"Shape of probs_array: {probs_array.shape}")
        print(f"Expected shape: {expected_shape}")
        raise ValueError("Shape of probs_array does not match total_frames and number of captions")
    
    # Calculate time per frame
    time_per_frame = total_duration / total_frames
    times = np.linspace(0, total_duration, total_frames)

    # Plotting the probabilities
    plt.figure(figsize=(12, 6))

    for i, caption in enumerate(captions):
        #plt.plot(times, probs_array[:, i], label=f"'{caption}'")

        # Plot moving average if specified
        if moving_avg_window:
            moving_avg = np.convolve(probs_array[:, i], np.ones(moving_avg_window)/moving_avg_window, mode='valid')
            moving_avg_times = times[:len(moving_avg)]  # Adjust times array to match moving average length
            plt.plot(moving_avg_times, moving_avg, label=f"'{caption}'", linestyle='-')
            plt.legend(fontsize=34)  # Adjust fontsize as needed


    plt.xlabel('Time (seconds)', fontsize = 14)
    plt.ylabel('Probability', fontsize = 14)
    #plt.title('Probabilities of Captions Over Time and Frames')
    plt.legend()
    plt.grid(True)
    plt.xticks(fontsize=14)  # Adjust fontsize as needed
    plt.yticks(fontsize=14)  # Adjust fontsize as needed
    
    # Add secondary x-axis for frame numbers
    ax = plt.gca()
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(np.linspace(0, total_duration, num=5))
    ax2.set_xticklabels(np.linspace(0, total_frames, num=5, dtype=int))
    ax2.set_xlabel('Frame Number', fontsize=14)
    plt.xticks(fontsize=14)  # Adjust fontsize as needed

    # Add vertical red lines for specified indices
    if indices is not None:
        for index in indices:
            time = index * time_per_frame
            plt.axvline(x=time, color='red', linestyle='-')

    # Save the plot
    os.makedirs(save_image_dir, exist_ok=True)
    image_path = os.path.join(save_image_dir, f'{name}.png')
    plt.savefig(image_path)
    plt.close()

    # Show the plot
    plt.show()


def plot_probs_csv(probs_csv_path, captions, save_image_dir, total_frames, total_duration, name, indices=None, moving_avg_window=None):
    """
    Plot the probabilities of captions over frames and time from a CSV file, and save the plot to a file.
    
    Args:
    - probs_csv_path (str): Path to the CSV file containing the array of probabilities.
    - captions (list): List of captions corresponding to the probabilities.
    - save_image_dir (str): Directory where the plot image will be saved.
    - total_frames (int): Total number of frames in the video.
    - total_duration (float): Total duration of the video in seconds.
    - name (str): Name of the file to save the plot as.
    - indices (list, optional): List of frame numbers to mark with vertical red lines.
    - moving_avg_window (int, optional): Window size for moving average, if desired.
    """
    # Load the probabilities array from the CSV file
    probs_df = pd.read_csv(probs_csv_path, header=None)
    probs_array = probs_df.values
    
    # Ensure the array has the correct shape
    if probs_array.shape[1] != len(captions):
        print(f"Shape of probs_array: {probs_array.shape}")
        raise ValueError("Number of columns in the CSV does not match number of captions")

    # Calculate time per frame
    time_per_frame = total_duration / total_frames
    times = np.linspace(0, total_duration, total_frames)

    # Plotting the probabilities
    plt.figure(figsize=(12, 6))
    for i, caption in enumerate(captions):
        plt.plot(times, probs_array[:, i], label=caption)
        
        # Plot moving average if specified
        if moving_avg_window and moving_avg_window > 1:
            moving_avg = np.convolve(probs_array[:, i], np.ones(moving_avg_window)/moving_avg_window, mode='valid')
            plt.plot(times[:len(moving_avg)], moving_avg, label=f"{caption} (MA)", linestyle='--')

    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.title(f'Probabilities of {name} Over Time')
    plt.legend()
    plt.grid(True)
    
    # Secondary x-axis for frame numbers
    ax = plt.gca()
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(np.linspace(0, total_duration, num=5))
    ax2.set_xticklabels(np.linspace(0, total_frames, num=5, dtype=int))
    ax2.set_xlabel('Frame Number', fontsize=14)

    # Add vertical red lines for specified indices
    if indices:
        for index in indices:
            plt.axvline(x=index * time_per_frame, color='red', linestyle='--')

    # Save the plot
    os.makedirs(save_image_dir, exist_ok=True)
    image_path = os.path.join(save_image_dir, f'{name}.png')
    plt.savefig(image_path)
    plt.close()

    # Show the plot
    plt.show()

def check_prob_array_dimensions(probs_array_path):
    """
    Load the probability array from the given path and print its dimensions.
    
    Args:
    - probs_array_path (str): Path to the pickle file containing the probability array.
    """
    # Load the probabilities array from the pickle file
    with open(probs_array_path, 'rb') as f:
        probs_array = pickle.load(f)
    
    # Convert to numpy array if not already
    probs_array = np.array(probs_array)
    
    # Print the shape of the probabilities array
    print(f"Shape of probs_array: {probs_array.shape}")



#name = "sit_device_71s"
name = "cuff_placement_tubing_55s"
frames_dir = f'../frames/{name}'
indices_1 = [44, 74, 89, 121, 138, 197, 209, 234, 334, 350]
indices_2 = [34, 87, 114, 136, 149, 176, 189, 211, 241, 261]
total_duration = 71 # seconds
total_frames = 428  # number of frames

'''Plot frame sequence'''
#plot_frame_seq(frames_dir, f'../results/{name}', 'frame_seq_2', indices_2, total_duration, total_frames)

'''Plot probabilites'''
# captions_posture = ["sitting", "standing"]
# prob_array_path_posture = f'../prob_arrays/posture_{name}.pkl'
# plot_probs(prob_array_path_posture, captions_posture, f'../results/{name}', total_frames, total_duration, 'probs_posture_2_raw', indices=None, moving_avg_window=False)


# captions_cuff = ["holding cuff", "not holding cuff"]
# prob_array_path = f'../prob_arrays/cuff_{name}.pkl'
# plot_probs(prob_array_path, captions_cuff, f'../results/{name}', total_frames, total_duration, 'probs_cuff_2_ma_indices', indices_2, moving_avg_window=10)


'''Check shape of probs array'''
prob_array_path_cuff_placement = f'../prob_arrays/{name}.csv'
#check_prob_array_dimensions(prob_array_path_cuff_placement)


'''Plot probabilites of cuff placement'''
captions_cuff_placement = ["cuff on lower arm", "cuff on elbow", "cuff on upper arm"]
prob_array_path_cuff_placement = f'../prob_arrays/{name}.pkl'
plot_probs_csv(prob_array_path_cuff_placement, captions_cuff_placement, f'../results/{name}', 331, 55, 'probs_CP_raw', indices=None, moving_avg_window=None)