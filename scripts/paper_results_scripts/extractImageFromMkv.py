import cv2


def extract_frame(video_path, time, output_imge_path):
    # Load the video
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print("Error: Could not open video file. Please check the file path and permissions.")
        return
    
    # Get the total number of frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    if time > duration:
        print(f"Error: Specified time {time} seconds exceeds video duration {duration} seconds.")
        return
    
    # Calculate the frame number to capture
    frame_number = int(time * fps)
    
    # Set the frame position
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    success, frame = video.read()
    
    if success:
        # Save the frame as an image
        cv2.imwrite(output_image_path, frame)
        print(f"Frame at {time} seconds was successfully saved as {output_image_path}")
    else:
        print(f"Failed to extract frame at {time} seconds")
    
    # Release the video capture object
    video.release()

# Example usage
video_path = "output_video.mkv"
time = 234 # Time in seconds
output_image_path = "extracted_frame.png"

extract_frame(video_path, time, output_image_path)
