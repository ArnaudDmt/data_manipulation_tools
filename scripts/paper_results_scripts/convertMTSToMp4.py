import ffmpeg

input_file= 'supplementaryVideo.mp4'
output_file = 'output_video.mp4'
bitrate = '800k'  # Example bitrate: 800 kbps

ffmpeg.input(input_file).output(
    output_file,
    b_v=bitrate,  # Enforce target video bitrate
    vcodec='libx264',  # Use H.264 codec for compression
    preset='fast',  # Adjust compression speed/quality
    an=None  # Remove audio
).run()