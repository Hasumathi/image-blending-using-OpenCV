import cv2
import numpy as np

# Load images
apple = cv2.imread('lak1.jpg')
orange = cv2.imread('yeluru1.jpg')

print(apple.shape)
print(orange.shape)

# Create a half-and-half image
apple_orange = np.hstack((apple[:, :256], orange[:, 256:]))

# Generate Gaussian pyramid for apple
apple_copy = apple.copy()
gp_apple = [apple_copy]
for i in range(6):
    apple_copy = cv2.pyrDown(apple_copy)
    gp_apple.append(apple_copy)

# Generate Gaussian pyramid for orange
orange_copy = orange.copy()
gp_orange = [orange_copy]
for i in range(6):
    orange_copy = cv2.pyrDown(orange_copy)
    gp_orange.append(orange_copy)

# Generate Laplacian pyramid for apple
apple_copy = gp_apple[5]
lp_apple = [apple_copy]
for i in range(5, 0, -1):
    gaussian_expanded = cv2.pyrUp(gp_apple[i])
    gaussian_expanded = cv2.resize(gaussian_expanded, (gp_apple[i-1].shape[1], gp_apple[i-1].shape[0]))
    laplacian = cv2.subtract(gp_apple[i-1], gaussian_expanded)
    lp_apple.append(laplacian)

# Generate Laplacian pyramid for orange
orange_copy = gp_orange[5]
lp_orange = [orange_copy]
for i in range(5, 0, -1):
    gaussian_expanded = cv2.pyrUp(gp_orange[i])
    gaussian_expanded = cv2.resize(gaussian_expanded, (gp_orange[i-1].shape[1], gp_orange[i-1].shape[0]))
    laplacian = cv2.subtract(gp_orange[i-1], gaussian_expanded)
    lp_orange.append(laplacian)

# Combine the halves of the two images at each level of the Laplacian pyramids
apple_orange_pyr = []
for apple_lap, orange_lap in zip(lp_apple, lp_orange):
    cols, rows, ch = apple_lap.shape
    laplacian = np.hstack((apple_lap[:, :cols//2], orange_lap[:, cols//2:]))
    apple_orange_pyr.append(laplacian)

# Reconstruct the image from the combined Laplacian pyramid
apple_orange_reconst = apple_orange_pyr[0]
for i in range(1, 6):
    apple_orange_reconst = cv2.pyrUp(apple_orange_reconst)
    apple_orange_reconst = cv2.resize(apple_orange_reconst, (apple_orange_pyr[i].shape[1], apple_orange_pyr[i].shape[0]))
    apple_orange_reconst = cv2.add(apple_orange_pyr[i], apple_orange_reconst)

# Display results
cv2.imshow("apple", apple)
cv2.imshow("orange", orange)
cv2.imshow("apple_orange", apple_orange)
cv2.imshow("apple_orange_reconstruct", apple_orange_reconst)

cv2.waitKey(0)
cv2.destroyAllWindows()
