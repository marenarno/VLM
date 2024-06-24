import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.image as mpimg

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
    probs_array = probs_df.values.T  # Transpose to match the expected shape

    # Calculate time per frame
    time_per_frame = total_duration / total_frames
    times = np.linspace(0, total_duration, total_frames)

    # Plotting the probabilities
    plt.figure(figsize=(12, 6))
    for i, caption in enumerate(captions):
        #plt.plot(times, probs_array[i], label=caption)  # Assuming each row in CSV is a caption
        
        # Plot moving average if specified
        if moving_avg_window:
            moving_avg = np.convolve(probs_array[i], np.ones(moving_avg_window)/moving_avg_window, mode='valid')
            plt.plot(times[len(times) - len(moving_avg):], moving_avg, label=f"{caption}, ma={moving_avg_window}", linestyle='-')

    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    #plt.title(f'Probabilities for placement of cuff on arm')
    plt.legend()
    plt.grid(True)

    # Add vertical red lines for specified indices
    if indices:
        for index in indices:
            plt.axvline(x=index * time_per_frame, color='red', linestyle='-')

    # Secondary x-axis for frame numbers
    ax = plt.gca()
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(np.linspace(0, total_duration, num=5))
    ax2.set_xticklabels(np.linspace(0, total_frames, num=5, dtype=int))
    ax2.set_xlabel('Frame Number', fontsize=14)

    # Save the plot
    save_path = os.path.join(save_image_dir, f'{name}.png')
    os.makedirs(save_image_dir, exist_ok=True)
    plt.savefig(save_path)

    # Optionally show the plot
    print("Hei")
    plt.show()
    plt.close()


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

    # Decide how many rows and columns based on the number of indices
    rows = 2
    cols = 3 if len(indices) > 2 else len(indices)  # Use 3 columns if more than 2 indices, else use the number of indices

    # Create a plot
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3), squeeze=False)

    # Configure spacing
    fig.subplots_adjust(hspace=30, wspace=0)  # Adjust horizontal and vertical spacing

    # Loop through all plots (even potential empty ones if not a full grid)
    for ax, index in zip(axes.flat, range(rows * cols)):
        if index < len(selected_frames):
            img = mpimg.imread(selected_frames[index])
            ax.imshow(img)
            ax.set_title(captions[index], fontsize=16)
            ax.axis('off')
        else:
            ax.axis('off')  # Hide unused subplots

    # Adjust layout
    plt.tight_layout()

    # Save image plot
    os.makedirs(save_image_dir, exist_ok=True)
    image_path = os.path.join(save_image_dir, f'{name}.png')
    plt.savefig(image_path)
    plt.show()
    plt.close()

def get_cuff_placement_from_caps(caption, keywords):
    # List keywords by specificity: most specific to least specific
    for key, value in keywords:
        if key in caption:
            return value
    return 0  # Default category for captions that don't match any specified category

def plot_captions_over_time(captions_csv_path, keywords, save_image_dir, total_frames, total_duration, name, indices=None):
    """
    Plot the captions over frames and time from a CSV file, and save the plot to a file.
    """
    # Load the captions from the CSV file
    captions_df = pd.read_csv(captions_csv_path)

    # Define y-axis categories based on keyword matching with the order of specificity
    def get_category(caption):
        for key, value in keywords:
            if key in caption:
                return value
        return 0  # Default category for captions that don't match any specified category

    # Apply the function to get y-axis positions
    y_positions = captions_df['Caption'].apply(get_category)

    # Calculate frame times based on total duration
    frame_times = np.linspace(0, total_duration, num=len(captions_df))

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.scatter(frame_times, y_positions, alpha=0.5, c='blue', marker='o')  # Plot each frame as a point

    category_labels = [''] + [key for key, _ in keywords]  # Create labels from keywords
    plt.yticks(range(len(category_labels)), category_labels)  # Set y-axis labels
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Caption', fontsize=14)
    plt.title('Caption Placement Over Time')
    plt.grid(True)

    # Mark specific indices if provided
    if indices:
        time_indices = [idx * total_duration / total_frames for idx in indices]
        for t_idx in time_indices:
            plt.axvline(x=t_idx, color='red', linestyle='-', label='Special Frame')

    # Add legend if indices are provided
    if indices:
        plt.legend()

    # Save the plot
    os.makedirs(save_image_dir, exist_ok=True)
    plt.savefig(os.path.join(save_image_dir, f'{name}.png'))
    plt.show()
    plt.close()

''' Plot probabilites from csv file '''
name = 'xavier cuff og'
probs_csv_path = '../prob_arrays/cuff_placement_xavier og.csv'
captions = ["lower arm", "elbow", "upper arm"]
#captions = ["tubing facing up", "tubing facing down"]
#captions = ["wire facing sholder", "wire facing hand"]
save_image_dir = f'../results/xavier_og_cuff'
total_frames = 132
total_duration = 65
indices=[43, 54, 62, 81, 102, 128]
moving_avg_window=None
plot_probs_csv(probs_csv_path, captions, save_image_dir, total_frames, total_duration, name, indices, moving_avg_window=6)

''' Plot a sequence of frames '''
frames_dir = '../frames/tubing_55s'
name = 'cuff_placement_seq'
#plot_frame_seq(frames_dir, save_image_dir, name, indices, total_duration, total_frames)

''' Plot captions from csv'''
# path_to_captions = 'captions_tubing_rag_clustering.csv'
# path_to_save_captions = '../results/tubing_55s'
# name = 'rag_clustering_result'
# keywords = [
#         ('cuff on upper arm 2 inches above the elbow', 5),  # Most specific phrase first
#         ('cuff on upper arm', 4),
#         ('cuff on elbow', 3),
#         ('cuff in hand', 2),
#         ('cuff on lower arm', 1)
#     ]
#print(get_cuff_placement_from_caps('cuff on upper arm 2 inches above the elbow', keywords))
#plot_captions_over_time(path_to_captions, keywords, path_to_save_captions, total_frames, total_duration, name, indices)
