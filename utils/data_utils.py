import cv2
import numpy as np
from skimage import filters
from matplotlib.path import Path

def get_fg_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = filters.threshold_triangle(gray)
    mask = ((gray <= thresh) * 255.0).astype(np.uint8)
    
    return mask

def get_color_histogram(image, mask, bin_size):
    bins = [256 // bin_size]
    hist_b = cv2.calcHist([image[:, :, 0]], [0], mask, bins, [0, 256])
    hist_g = cv2.calcHist([image[:, :, 1]], [0], mask, bins, [0, 256])
    hist_r = cv2.calcHist([image[:, :, 2]], [0], mask, bins, [0, 256])
    
    # hist_b = np.log(hist_b / hist_b.sum(), dtype=np.float32)
    # hist_g = np.log(hist_g / hist_g.sum(), dtype=np.float32)
    # hist_r = np.log(hist_r / hist_r.sum(), dtype=np.float32)
    
    hist = np.concatenate((hist_r, hist_g, hist_b), axis=1)
    hist = np.transpose(hist, (1, 0))
    
    return hist

def normalize_image(img):
    return (img - 128.0) / 128.0

def PolyArea2D(pts):
    lines = np.hstack([pts,np.roll(pts,-1,axis=0)])
    area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
    return area

def get_hull_mask(hulls):
    x, y = np.meshgrid(np.arange(300), np.arange(300)) # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T 
    
    hull_mask = None
    
    for sample_hull in hulls:
        p = Path(np.squeeze(sample_hull, axis=1).tolist()) # make a polygon
        grid = p.contains_points(points)
        mask = grid.reshape(300,300)
        
        if hull_mask is None:
            hull_mask = mask
        else:
            hull_mask = hull_mask | mask
            
    return hull_mask

def get_visual_hull(mask, low_area=500, high_area=80000):
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # color_contours = (0, 255, 0) # green - color for contours
    # color = (255, 0, 0) # blue - color for convex hull
    # drawing = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    
    hull = []
    for i in range(len(contours)):
        hull_tmp = cv2.convexHull(contours[i], False)
        area = PolyArea2D(hull_tmp[:, 0].tolist())
        
        if area < low_area or area > high_area:
            continue
            
        hull.append(hull_tmp)
        # cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
        # cv2.drawContours(drawing, hull, i, color, 1, 8)
            
    return hull

def get_new_mask(image_path):
    img = cv2.imread(image_path)
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = filters.threshold_otsu(img_gray)
    mask = ((img_gray <= thresh)*255.0).astype(np.uint8)
    hulls = get_visual_hull(mask)
    
    hull_mask = get_hull_mask(hulls)
    
    return img, img_gray, mask, hulls, hull_mask