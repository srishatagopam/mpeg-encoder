# mpeg-encoder
MPEG encoder and decoder framework that showcases many image processing techniques; namely frame sub/up-sampling,  image/video compression, and motion estimation. This project
was originally written in MATLAB, so I decided to rewrite it in Python as an exercise and to get used to image processing packages, such as cv2.

# Usage
Here are the command line arguments:
```
--file        # Filepath of input video
--extract     # Which frames to extract
```

# Example
```
$python mpeg.py --file '/filepath/video.avi' --extract 5 10
```

## Output
![alt text](https://github.com/srishatagopam/mpeg-encoder/blob/main/output.jpeg)
