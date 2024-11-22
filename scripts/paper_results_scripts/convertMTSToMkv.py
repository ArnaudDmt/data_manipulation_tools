import ffmpeg

input_file= 'rhps1_expe3_video.MTS'
output_file = 'output_video.mkv'

ffmpeg.input(input_file).output(output_file).run() 
