import cv2
import numpy as np
import matplotlib.pyplot as plt

# loading images
visible = cv2.imread("images/visible.jpg", cv2.IMREAD_GRAYSCALE)
hidden = cv2.imread("images/hidden.jpg", cv2.IMREAD_GRAYSCALE)

# resize hidden image to match visible image size
hidden = cv2.resize(hidden, (visible.shape[1], visible.shape[0]))

# Simulating painting over the hidden image
composite = (0.8 * visible + 0.2 * hidden).astype(np.uint8)

# edge detection of the composite image to reveal structure
edges = cv2.Canny(composite, 50, 150)

# Highlight differences between visible and composite images
difference = cv2.absdiff(visible, composite)
diff_norm = cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX)
heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)


# Show results
plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.title("Visible Painting")
plt.imshow(visible, cmap="gray")
plt.axis("off")

plt.subplot(2,3,2)
plt.title("Composite Image")
plt.imshow(composite, cmap="gray")
plt.axis("off")

plt.subplot(2,3,3)
plt.title("Edge Detection")
plt.imshow(edges, cmap="gray")
plt.axis("off")

plt.subplot(2,3,4)
plt.title("Difference Map")
plt.imshow(diff_norm, cmap="gray")
plt.axis("off")

plt.subplot(2,3,5)
plt.title("Difference Heatmap")
plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.savefig("results/output.png", dpi=300, bbox_inches="tight")
plt.show()