import ffmpeg

input_file= 'videoIJRR.mp4'
output_file = 'output_video.avi'
bitrate = '450k'  # Example bitrate: 800 kbps

ffmpeg.input(input_file).output(
    output_file,
    **{
        'b:v': bitrate,      # Correct key with colon
        'vcodec': 'libx264',
        'preset': 'fast',
        'an': None           # Remove audio
    }
).run()