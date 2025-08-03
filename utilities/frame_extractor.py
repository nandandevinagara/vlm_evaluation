import cv2
import os
import random
import sys

def extract_random_frame_from_videos(base_path, output_base_folder, folder_range_str):
    """
    Goes into each specified folder and video, selects a random frame,
    and stores it in a separate folder.

    Args:
        base_path (str): The base directory containing the video folders.
        output_base_folder (str): The main folder to store all extracted frames.
        folder_range_str (str): A string indicating the range of folders to process
                                (e.g., "1-10", "5-5").
    """

    # Create the main output folder if it doesn't exist
    os.makedirs(output_base_folder, exist_ok=True)

    # Get all subdirectories (folders) in the base_path
    all_folders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

    # Parse the folder range string
    try:
        start_idx, end_idx = map(int, folder_range_str.split('-'))
        # Adjust to 0-based indexing
        start_idx -= 1
        end_idx -= 1
    except ValueError:
        print(f"Error: Invalid folder range format. Expected 'start-end', got '{folder_range_str}'")
        return

    if not (0 <= start_idx <= end_idx < len(all_folders)):
        print(f"Error: Folder range {folder_range_str} is out of bounds for {len(all_folders)} folders.")
        return

    # Select folders based on the provided range
    folders_to_process = all_folders[start_idx : end_idx + 1]

    print(f"Processing folders: {folders_to_process}")

    for folder_name in folders_to_process:
        folder_path = os.path.join(base_path, folder_name)
        print(f"\nEntering folder: {folder_path}")

        # Iterate through each file in the current folder
        for filename in os.listdir(folder_path):
            if filename.endswith(('.avi', '.mp4', '.mov', '.mkv')): # Add/remove video extensions as needed
                video_path = os.path.join(folder_path, filename)
                print(f"  Processing video: {video_path}")

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"  Failed to open video: {video_path}")
                    continue

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames == 0:
                    print(f"  No frames found in video: {video_path}")
                    cap.release()
                    continue

                # Select a random frame number
                random_frame_num = random.randint(0, total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_num)

                ret, frame = cap.read()
                if ret:
                    # Construct the output filename (e.g., v_Basketball_g01_c01.png)
                    # Remove the original extension and add .png
                    output_filename = os.path.splitext(filename)[0] + '.png'
                    output_filepath = os.path.join(output_base_folder, output_filename)

                    cv2.imwrite(output_filepath, frame)
                    print(f"  Extracted random frame ({random_frame_num}/{total_frames}) to {output_filepath}")
                else:
                    print(f"  Could not read random frame from {video_path}")

                cap.release()

# --- Example Usage ---
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python your_script_name.py <base_video_directory> <folder_range_string>")
        print("Example: python your_script_name.py /path/to/UCF-101 1-10")
        print("Example: python your_script_name.py /path/to/UCF-101 1-1")
        sys.exit(1)

    base_video_directory = sys.argv[1] # e.g., '/hkfs/work/workspace/scratch/st_st189656-myspace/UCF-101'
    folder_range = sys.argv[2]         # e.g., '1-10' or '1-1'
    frames_output_folder = 'frames_Extracted'

    extract_random_frame_from_videos(base_video_directory, frames_output_folder, folder_range)