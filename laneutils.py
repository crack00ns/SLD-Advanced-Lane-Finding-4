import numpy as np
import cv2, glob, pickle, os, pdb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sys import platform

''' Constants to be used in the project '''
FILE_SEPARATOR = "/" if platform != "win32" else "\\"  # File Separator based on OS (useful while testing on windows)

# Binary Thesholds
KSIZE = 3                       # Sobel Kernel: Choose a larger odd number for Sobel kernel size to smooth gradient measurements
GRADX_THRES = (50 , 200)        # SobelX Thresholds
GRADY_THRES = (50 , 200)        # SobelY Thresholds
MAG_THRES   = (50 , 200)        # Sobel Magnitude Thresholds
DIR_THRES   = (0.3, 1.3)        # Sobel Direction Thresholds
H_THRES     = (20,  100)        # H Thresholds in HLS Space
L_THRES     = (200, 255)        # L Thresholds in HLS Space
S_THRES     = (170, 255)        # S Thresholds in HLS Space

# Mask region paddings parameters
X_PAD = 100 
Y_PAD = 85

# For finding curvature
YM_PER_PIX = 30/720  # meters per pixel in y dimension
XM_PER_PIX = 3.7/550 # meters per pixel in x dimension (550 pixel is the distance between lanes in warped image)

# Max x and y value in an image
Y_MAX = 720  # Max y-value in an image
X_MAX = 1280 # Max x-value in an image

# Define the source and destination points for image warping (Obtained after testing in jupyter notebook)
SRC = np.float32([[208,720],[580,460],[705,460],[1120,720]])
DST = np.float32([[350, 720],[350, 0],[900, 0],[900, 720]])

# Sliding window constants
NUM_WINDOWS = 9 # Number of sliding windows
MINPIX = 50     # Set minimum number of pixels found to recenter sliding window

# Lane constants
THRESH_PARALLEL  = (0.0004, 0.56)  # threshold for determining whether two lane lines are parallel
THRESH_LANE_DIST = (480, 620)      # threshold for determining if the lane distance is right or not

FONT_SIZE_FIG = 15 # Font size for figures

'''Calibrate the camera using checkerboard images and also perform bird's-eye view perspective transform '''
class CalibrateCamPerspectiveTransform:
    def __init__(self, cal_path, out_path, pickle):
        '''Initialise class variables'''
        self.cal_path = cal_path
        self.out_path = out_path
        self.pickle = pickle
        self.caliberated = False
        self.nx = 9 # number of inside corners in x
        self.ny = 6 # number of inside corners in y

        # Compute the (inverse) perspective transform (Matrices computed first time class is created)
        self.M = cv2.getPerspectiveTransform(SRC, DST)
        self.Minv = cv2.getPerspectiveTransform(DST, SRC)

    def caliberate(self):
        print('Calibrating the camera...')
        nx  = self.nx # number of inside corners in x
        ny  = self.ny # number of inside corners in y
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((ny*nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob(self.cal_path+'calibration*.jpg')

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = mpimg.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
                write_name = self.out_path+'corners_found_'+fname.split(FILE_SEPARATOR)[-1]
                mpimg.imsave(write_name, img)
            else:
                print('Error detecting checkerboard corners {}'.format(fname))
        img_size = (img.shape[1], img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        self.mtx = mtx
        self.dist = dist
        self.caliberated = True
        print("Caliberation Matrix Generated!")

        # Pickle file to hold caliberations 
        cal_data = {}
        cal_data["mtx"], cal_data["dist"] = mtx, dist
        pickle.dump(cal_data, open(self.out_path + '/' + self.pickle, "wb"))

    # Camera Undistort for a given image
    def undistort(self,img, visualize = False):
        if not self.caliberated: # Load from pickle file or caliberate if pickle file doesn't exist
            if os.path.isfile(self.out_path + '/' + self.pickle):
                print("Loading from picked file...")
                with open(self.out_path + '/' + self.pickle, mode='rb') as f:
                    cal_data = pickle.load(f)
                self.mtx = cal_data["mtx"]
                self.dist = cal_data["dist"]
                self.caliberated = True
            else:
                self.caliberate()
        undst_img = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        
        # Visualize undistortion
        if visualize:
            show_images_sbs(img, undst_img,'Original Image','Undistorted Image')
        return undst_img
    
    # Warp Image to Bird's eye view
    def warp(self, image):
        return cv2.warpPerspective(image, self.M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    # UnWarp Image to Bird's eye view
    def unwarp(self, image):
        return cv2.warpPerspective(image, self.Minv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

''' Line Class holds parameters for each line of lane. Implements helper functions
    to check if lanes are parallel and find distance between two lines'''
class Line():
    def __init__(self, n_frames=5):
        self.n_frames = n_frames                        # Number of previous frames used to smooth the current frame
        self.best_fit = None                            # polynomial coefficients averaged over the last n iterations
        self.best_fit_poly = None                       # Polynomial coefficients averaged over the last n iterations
        self.current_fit = None                         # polynomial coefficients for the most recent fit
        self.current_fit_poly = None                    # polynomial coefficients for the most recent fit
        self.allx = None                                # x values for detected line pixels
        self.ally = None                                # y values for detected line pixels
    
    def update_fit(self, x, y):
        '''Update lane line x and y points and corresponding currentfit and bestfit'''
        self.allx = x
        self.ally = y
        self.current_fit = np.polyfit(self.ally, self.allx, 2)
        if self.best_fit is None:
            self.best_fit = self.current_fit
        else:
            self.best_fit = (self.best_fit * (self.n_frames - 1) + self.current_fit) / self.n_frames
        self.current_fit_poly = np.poly1d(self.current_fit)
        self.best_fit_poly    = np.poly1d(self.best_fit)

    def lines_parallel_or_not(self, line2):
        '''Determine whether two lane lines are parallel by comparing fitted polynomial coefficients'''
        diff_coeff_first = np.abs(self.current_fit[0] - line2.current_fit[0])
        diff_coeff_second = np.abs(self.current_fit[1] - line2.current_fit[1])
        return diff_coeff_first < THRESH_PARALLEL[0] and diff_coeff_second < THRESH_PARALLEL[1]

    def distance_between_current_fit(self, line2):
        '''Calculate the distance between the currently fitted polynomials of two lines'''
        return np.abs(self.current_fit_poly(Y_MAX) - line2.current_fit_poly(Y_MAX))

''' LaneDetector Class detects lane in an image frame, plots/fills the lane''' 
class LaneDetector():
    def __init__(self, calibrator_perspective_transform, num_frames, debug = False):
        self.calibrator_perspective_transform = calibrator_perspective_transform
        self.num_frames = num_frames
        self.left_line = None
        self.right_line = None
        self.lane_was_detected = False
        self.debug = debug

    def peak_detection(self, binary_warped):
        ''' Find the peak of the left and right halves of the histogram. These 
            will be the starting point for the left and right lines '''
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Plot Histogram if debug is enabled
        if self.debug: 
            plt.figure(figsize=(8,5))
            plt.plot(histogram)
            plt.title("Histogram of Bottom half Image")
            plt.xlim(0,X_MAX)
            plt.show()
        return leftx_base, rightx_base

    def are_these_lane_lines(self, line1, line2):
        ''' Check if two lines are parallel and separated by valid lane distances'''
        are_parallel = line1.lines_parallel_or_not(line2)
        lane_dist_valid = THRESH_LANE_DIST[0]<line1.distance_between_current_fit(line2)<THRESH_LANE_DIST[1]
        return are_parallel and lane_dist_valid

    def verify_lane_points(self, leftx, lefty, rightx, righty):
        ''' Verify if left and right points found belong to valid lane lines'''
        left_found, right_found = False, False
        left_line, right_line  = Line(), Line()
        left_line.update_fit(leftx, lefty)
        right_line.update_fit(rightx, righty)
        if self.are_these_lane_lines(left_line, right_line): left_found, right_found  = True, True
        elif self.right_line is not None and self.are_these_lane_lines(left_line, self.right_line): left_found  = True
        elif self.left_line is not None and self.are_these_lane_lines(self.left_line, right_line): right_found = True
        return (left_found, right_found)

    def detect_and_plot_lanes(self, frame_orig, binary_warped):
        '''Detect Lanes and plot lane lines'''
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Set the width of the windows +/- margin
        margin = 100

        if self.debug: # if debug is enabled, Create an output image to draw on and visualize the result
            out_img = np.uint8(np.dstack((binary_warped, binary_warped, binary_warped))*255)
            window_img = np.zeros_like(out_img)

        if not self.lane_was_detected: # If lane was not detected use the sliding window method to find left and right lane points
            leftx_base, rightx_base = self.peak_detection(binary_warped) # Find left and right lane x points using peak histogram detection method
            window_height = np.int(binary_warped.shape[0]/NUM_WINDOWS) # Set height of windows

            # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base

            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(NUM_WINDOWS):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window+1)*window_height
                win_y_high = binary_warped.shape[0] - window*window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                if self.debug: # if debug is enabled, draw the windows on the visualization image
                    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
                    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > MINPIX pixels, recenter next window on their mean position
                if len(good_left_inds) > MINPIX:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > MINPIX:        
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        else: # if lane was detected, no need to search using sliding windows again
            left_lane_inds = ((nonzerox > (self.left_line.best_fit_poly(nonzeroy)-margin)) & (nonzerox < (self.left_line.best_fit_poly(nonzeroy)+margin)))
            right_lane_inds = ((nonzerox > (self.right_line.best_fit_poly(nonzeroy)-margin)) & (nonzerox < (self.right_line.best_fit_poly(nonzeroy)+margin)))
            if self.debug: # if debug is enabled, plot the lane search area on visualization images: See course notes
                ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
                left_fitx = self.left_line.best_fit_poly(ploty)
                right_fitx = self.right_line.best_fit_poly(ploty)
                left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
                left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
                left_line_pts = np.hstack((left_line_window1, left_line_window2))
                right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
                right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
                right_line_pts = np.hstack((right_line_window1, right_line_window2))
                cv2.fillPoly(window_img, np.int_([left_line_pts]), (255,0, 0))
                cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
                out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        if self.debug: # plot visualization if debug is enabled
            out_img[lefty, leftx] = [255, 0, 0]
            out_img[righty, rightx] = [0, 0, 255]
            ploty = np.linspace(0,Y_MAX-1, num=Y_MAX)
            plt.figure(figsize=(8,8))
            plt.imshow(out_img)
            plt.title('Detected Lanes in Warped Image', fontsize=15)
            plt.plot(np.poly1d(np.polyfit(lefty, leftx, 2))(ploty), ploty, color='yellow')
            plt.plot(np.poly1d(np.polyfit(righty, rightx, 2))(ploty), ploty, color='yellow')
            plt.xlim(0,X_MAX)
            plt.ylim(Y_MAX,0)
            plt.show()

        left_found, right_found =  self.verify_lane_points(leftx, lefty, rightx, righty)
        self.lane_was_detected = left_found&right_found

        # Fit a second order polynomial to each
        if left_found:
            if self.left_line is None: self.left_line = Line()
            self.left_line.update_fit(leftx, lefty)
        if right_found:
            if self.right_line is None: self.right_line = Line()
            self.right_line.update_fit(rightx, righty)
        if self.left_line is not None and self.right_line is not None:
            frame_orig = self.lane_fill_add_info(frame_orig,binary_warped)
        return frame_orig

    def lane_fill_add_info(self, image, binary_warped):
        ''' Fill Lane and warp and add to original image '''
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        y = np.linspace(0, Y_MAX-1)
        left_fitx  = self.left_line.best_fit_poly(y)
        right_fitx = self.right_line.best_fit_poly(y)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.calibrator_perspective_transform.unwarp(color_warp) 
        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

        # Write the curvature and vehicle positon on the image frame
        left_curvature  = calculate_curvature(self.left_line.best_fit_poly) 
        right_curvature = calculate_curvature(self.right_line.best_fit_poly)
        average_curvature =  (left_curvature+right_curvature)/2
        vehicle_position = (X_MAX/2 - (self.right_line.best_fit_poly(Y_MAX)+self.left_line.best_fit_poly(Y_MAX))/2)*XM_PER_PIX
        left_or_right = 'left' if vehicle_position<0 else "right"
        overlay = result.copy()
        cv2.rectangle(overlay, (10, 10), (610, 100), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.3, result, 1 - 0.3, 0, result)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, "Lane Curvature:  {:0.2f}m".format(average_curvature), (50, 40), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result, "Vehicle Position:  {:0.2f}m to the {}".format(abs(vehicle_position), left_or_right), (50, 80), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return result

    def find_lane_img_frame(self, frame):
        # 1. Undistort the Image
        undst_img = self.calibrator_perspective_transform.undistort(frame, visualize = self.debug)

        # 2. Create thresholded binary image
        thresh_img = threshold_pipeline(undst_img, show_pipeline = self.debug)

        # 3. Perspective transform to create a bird's-eye view with parallel lanes
        binary_warped = self.calibrator_perspective_transform.warp(thresh_img)
        if self.debug: show_images_sbs(thresh_img,binary_warped,'Thresholded', 'Binary Warped')

        # 4. Detect and plot the lane for each frame
        images_with_lanes = self.detect_and_plot_lanes(undst_img, binary_warped)
        return images_with_lanes

def show_images_sbs(img1, img2, title1="First Image", title2="Second Image"):
    '''Show Images Side by Side'''
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img1, cmap="gray")
    ax1.set_title(title1, fontsize=20)
    ax2.imshow(img2, cmap="gray")
    ax2.set_title(title2, fontsize=20)
    plt.show()


def show_single_image(img,title=""):
    ''' Show Single Image '''
    plt.figure(figsize=(8,8))
    plt.imshow(img, cmap = "gray")
    plt.title(title, fontsize=15)
    plt.xlim(0,X_MAX)
    plt.ylim(Y_MAX,0)
    plt.show()

def abs_sobel_threshold(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    ''' Threshold gradient magnitude x and y '''
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient=='x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_threshold(img, sobel_kernel=3, thresh=(0, 255)):
    ''' Threshold gradient magnitude '''
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.sqrt(np.square(sobelx)+np.square(sobely))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    ''' Threshold gradient direction '''
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    arc_sobel = np.arctan2(np.abs(sobely),np.abs(sobelx))
    binary_output = np.zeros_like(arc_sobel)
    binary_output[(arc_sobel >= thresh[0]) & (arc_sobel <= thresh[1])] = 1
    return binary_output


def hls_h_threshold(img, thresh=(0, 180)):
    ''' Threshold H color channel '''
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= thresh[0]) & (h_channel <= thresh[1])] = 1
    return h_binary

def hls_l_threshold(img, thresh=(0, 255)):
    ''' Threshold L color channel '''
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= thresh[0]) & (l_channel <= thresh[1])] = 1
    return l_binary

def hls_s_threshold(img, thresh=(0, 255)):
    ''' Threshold S color channel '''
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    return s_binary

def threshold_pipeline(img, show_pipeline = False):
    ''' Image Threshold Pipeline '''
    gradx = abs_sobel_threshold(img, orient='x', sobel_kernel=KSIZE, thresh=GRADX_THRES)
    grady = abs_sobel_threshold(img, orient='y', sobel_kernel=KSIZE, thresh=GRADY_THRES)
    mag_binary = mag_threshold(img, sobel_kernel=KSIZE, thresh=MAG_THRES)
    dir_binary = dir_threshold(img, sobel_kernel=KSIZE, thresh=DIR_THRES)
    h_binary = hls_h_threshold(img, thresh=H_THRES)
    l_binary = hls_l_threshold(img, thresh=L_THRES)
    s_binary = hls_s_threshold(img, thresh=S_THRES)
    
    # Combine the results with some experimentation
    combined = np.zeros_like(gradx)
    combined[((gradx == 1)&(grady ==1))|((mag_binary == 1) & (dir_binary == 1))|(l_binary == 1) |(s_binary == 1)] = 1
    final_binary  = region_of_interest(combined)
    # Plot the intermediate binaries to see visually whats going on
    if show_pipeline:
        f, ax = plt.subplots(4, 3, figsize=(30,20))
        ax[0,0].imshow(img)
        ax[0,0].set_title('Original Image', fontsize=20)

        ax[0,1].imshow(gradx, cmap='gray')
        ax[0,1].set_title('GradX', fontsize=20)

        ax[0,2].imshow(grady, cmap='gray')
        ax[0,2].set_title('GradY', fontsize=20)

        ax[1,0].imshow(mag_binary, cmap='gray')
        ax[1,0].set_title('Absolute', fontsize=20)

        ax[1,1].imshow(dir_binary, cmap='gray')
        ax[1,1].set_title('Sobel Direction', fontsize=20)

        ax[1,2].imshow(h_binary, cmap='gray')
        ax[1,2].set_title('H Binary', fontsize=20)

        ax[2,0].imshow(l_binary, cmap='gray')
        ax[2,0].set_title('L Binary', fontsize=20)

        ax[2,1].imshow(s_binary, cmap='gray')
        ax[2,1].set_title('S Binary', fontsize=20)

        ax[2,2].imshow(combined, cmap='gray')
        ax[2,2].set_title('Combined', fontsize=20)

        ax[3,0].imshow(final_binary, cmap='gray')
        ax[3,0].set_title('Final Masked Image', fontsize=20)

        ax[3,1].axis("off")
        ax[3,2].axis("off")
        plt.show()
    return region_of_interest(combined)

def region_of_interest(image):
    '''
    Applies an image mask. Only keeps the region of the image defined by the polygon formed from `vertices`.
    Function borrowed from other udacity resources and users, Credit due to Unknown authors.
    '''
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]),                                    # bottom left
                          (imshape[1] / 2 - X_PAD, imshape[0] / 2 + Y_PAD),   # top left
                          (imshape[1] / 2 + X_PAD, imshape[0] / 2 + Y_PAD),   # top right
                          (imshape[1], imshape[0])]],                         # bottom right
                        dtype=np.int32)

    mask = np.zeros_like(image)

    # Define a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # Fill pixels inside the polygon defined by "vertices" with the fill color (i.e. white)
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # Return the image only where mask pixels are nonzero
    return cv2.bitwise_and(image, mask)

def calculate_curvature(fit_crv):
    '''Calculate the line curvature (in meters)'''
    fit_crv = np.poly1d(fit_crv)
    y = np.array(np.linspace(0, Y_MAX-1, num=Y_MAX))
    x = np.array([fit_crv(x) for x in y])
    y_eval = np.max(y)

    fit_crv = np.polyfit(y * YM_PER_PIX, x * XM_PER_PIX, 2)
    curvature_radius = ((1 + (2 * fit_crv[0] * Y_MAX / 2. + fit_crv[1]) ** 2) ** 1.5) / np.absolute(2 * fit_crv[0])
    return curvature_radius