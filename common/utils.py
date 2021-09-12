import numpy as np
from scipy import interpolate
import cv2
import torch

def spline_interp(xk, yk, z, xg, yg):
    # Interpolate knots with bicubic spline
    tck = interpolate.bisplrep(xk, yk, z)
    # Evaluate bicubic spline on (fixed) grid
    zint = interpolate.bisplev(xg[:,0], yg[0,:], tck)
    # zint is between [-1, 1]
    zint = np.clip(zint, -1, 1)
    # Convert spline values to binary image
    C = 255/2; thresh = C
    img = np.array(zint*C+C).astype('uint8')
    # Thresholding give binary image, which gives better contour
    _, thresh_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    return thresh_img

def isoperi_reward_from_img(img):
    # Extract contours and calculate perimeter/area   
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = 0; peri = 0
    for cnt in contours:
        area -= cv2.contourArea(cnt, oriented=True)
        peri += cv2.arcLength(cnt, closed=True)
    if peri == 0:
        return 0
    return np.sqrt(abs(area))/abs(peri)

def isoperi_reward(xk, yk, z, xg, yg):
    img = spline_interp(xk, yk, z, xg, yg)
    return isoperi_reward_from_img(img)

# compute nabla -g(q). Wish to minimize g
def grad_isoperi_reward(xk, yk, z, xg, yg):
    # Basis
    B = np.eye(z.shape[0]*z.shape[1]); eps = 1e-5
    ret = np.zeros(z.shape[0]*z.shape[1])
    # Numerically estimate gradient of reward function
    for i in range(len(ret)):
        ret[i] = (isoperi_reward(xk, yk, z+B[i].reshape(4, 4)*eps, xg, yg)\
                  -isoperi_reward(xk, yk, z-B[i].reshape(4, 4)*eps, xg, yg))/(2*eps)
    
    return ret

def generate_coords(dim=16, num_samples=1024, xk=None, yk=None, xg=None, yg=None, total_random=True):
    if xk is None:
        xk, yk = np.mgrid[-1:1:4j, -1:1:4j]
    if xg is None:
        xg, yg = np.mgrid[-1:1:50j, -1:1:50j]
    if total_random:
        # Totally random
        q = np.random.rand(num_samples, dim) 
    else:
        # Random with zero padding
        width = int(np.sqrt(dim))
        q = np.zeros((num_samples, width, width)) 
        q[:, 1:(width-1), :(width-1)] = np.random.rand(num_samples, width-2, width-1)
        q = q.reshape(num_samples, -1)
    q -= .5
    p = np.zeros((num_samples, dim))
    
    for i in range(num_samples):
        p[i] = grad_isoperi_reward(xk, yk, q[i].reshape(4, 4), xg, yg)
    
    return torch.cat((torch.tensor(q, dtype=torch.float), torch.tensor(p, dtype=torch.float)), dim=1)        