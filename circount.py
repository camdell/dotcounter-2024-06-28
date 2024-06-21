import cv2
import numpy as np
import argparse
from matplotlib.pyplot import show, subplots, rc
from pathlib import Path

rc('font', size=20)

parser = argparse.ArgumentParser()
parser.add_argument('imagefile', type=Path)
parser.add_argument('--show', action='store_true')
parser.add_argument('--output', type=Path)

args = parser.parse_args()

dot_colors = {
    'red'   : [(0, 0, 240),(10, 10, 255)],
    'green' : [(0, 240, 0), (10, 255, 10)],
    'yellow': [(0, 240, 250), (10, 255, 255)],
}

fig, axes = subplots(2, 2, figsize=(10, 10))

img = cv2.imread(str(args.imagefile))
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original')

# apply medianBlur to smooth image before threshholding
#   smooth image by 7x7 pixels, may need to adjust a bit
blur = cv2.medianBlur(img, 7)

for ax, (color, (lower, upper)) in zip(axes.flat[1:], dot_colors.items()):
    # apply threshhold color to white (255, 255, 255) and the rest to black(0, 0, 0)
    mask = cv2.inRange(blur, lower, upper)

    circles = cv2.HoughCircles(
        image=mask,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=20,
        param2=8,
        minRadius=0,
        maxRadius=60
    )

    output = img.copy()
    index = 0
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image,
            #   then draw a rectangle corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 0, 0), 2)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 0), -1)
            index = index + 1


    ax.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    ax.set_title(f'{color.title()} = {index}')
    print(f'No. of {color} circles detected = {index}')


fig.suptitle('Counts of Red, Green, Yellow Dots')

for ax in axes.flat:
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

if args.output is not None:
    fig.savefig(args.output)

if args.show is not False:
    show()
