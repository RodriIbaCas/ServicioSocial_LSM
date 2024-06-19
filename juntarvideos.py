##Codigo auxiliar para juntar videos en un solo archivo

from moviepy.editor import VideoFileClip, concatenate_videoclips

# Load the videos
video1 = VideoFileClip('A.mp4')
video2 = VideoFileClip('B.mp4')
video3 = VideoFileClip('C.mp4')

# Concatenate the videos
final_video = concatenate_videoclips([video1, video2, video3])

# Save the merged video
final_video.write_videofile('test.mp4')

