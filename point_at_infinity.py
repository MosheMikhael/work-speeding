import numpy as np
import cv2

import random
import os
import sys
from collections import deque
import draw
import service_functions as sf
import base_classes as bc


def find_interest_points(img, quality_level = 0.3, max_corners = 500):
    """
    Finds edges in an image using the Canny86 algorithm.

    :param img: source image, array of coordinates of points.
    :param quality_level: parameter of searching with Canny86 algorithm.
    :param max_corners: max number of corners.
    :return: array of points.
    """
    feature_params = dict(maxCorners=max_corners, qualityLevel=quality_level, 
                                              minDistance=10, blockSize=10)
    points = cv2.goodFeaturesToTrack(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
                                             mask=None, **feature_params)
    return points


def find_opt_flow_lk_with_points(img1, img2, pts, winSize_x=30, winSize_y=30):
    """
    The function searches for feature points and compute
    the optical flow for them using Lucas–Kanade method

    :param img1: source first image, array of coordinates of points.
    :param img2: source second image, array of coordinates of points.
    :param pts: array of points from first image.
    :return: two arrays of points, that built by optical flow Lucas–Kanade 
    :method.
    """
    if len(pts) == 0:
        return np.array([], np.float32), np.array([], np.float32)
    lk_params = dict(winSize=(winSize_x, winSize_y), maxLevel=2, 
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.3))
    points, st, err = cv2.calcOpticalFlowPyrLK(
                 cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
                 cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),
                 pts, None, **lk_params
             ) 
#    print('err=', err, flush = True)
    out_new = points[st == 1]
    out_old = pts[st == 1]
#    print(st)
    L = len(out_old)
    out1 = np.zeros([L, 2])
    out2 = np.array(out1)
    for i in range(0, L):
        out2[i,:] = out_new[i]
        out1[i,:] = out_old[i]    
    
    return out1, out2


def find_opt_flow_lk(img1, img2, quality_level=0.3, max_corners=500):
    """
    The function searches for feature points and computes
    the optical flow for them using Lucas–Kanade method.

    :param img1: source first image, array of coordinates of points.
    :param img2: source second image, array of coordinates of points.
    :param quality_level: quality level parameter.
    :param max_corners: max number of corners.
    :return: two arrays of points, that built by optical flow Lucas–Kanade method.
    """
    
    pts = find_interest_points(img1, quality_level, max_corners)
#    print(pts.shape)
#    if pts.shape[0] == 0:
#        sys.exit(-123)
    if type(pts) is type(None):
        print('None')
        return None, None
    pts1, pts2 = find_opt_flow_lk_with_points(img1, img2, pts)
    
    return pts1, pts2


def find_opt_flow_orb(img1, img2, good_match_percent=0.3, max_corners=500):
    """
    The function searches for feature points and computes
    the optical flow for them using ORB method.

    :param img1: source first image, array of coordinates of points.
    :param img2: source second image, array of coordinates of points.
    :param good_match_percent: how much points will be in use.
    :param max_corners: max number of corners.
    :return: two arrays of points, that built by optical flow ORB method.
    """
    im1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(max_corners)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create('BruteForce-Hamming')
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * good_match_percent)
    matches = matches[:numGoodMatches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
        
    #points1 = np.array(points1)         #??????????? why doing that ?
    #points2 = np.array(points2)         # idem
    
    return points1, points2


def find_opt_flow(img1, img2, prop=0.3, max_corners=500, method='orb'):
    """
    The function searches for feature points and computes
    the optical flow by chosen method.

    :param img1: source first image, array of coordinates of points.
    :param img2: source second image, array of coordinates of points.
    :param prop: property for method.
    :param max_corners: max number of corners.
    :param method: param, which method is using. ('ord' or 'lk')
    :return: two arrays of points, that built by optical flow.
    """
    if method == 'orb':
        return find_opt_flow_orb(img1, img2, prop, max_corners)
        
    if method == 'lk':
        return find_opt_flow_lk(img1, img2, prop, max_corners)
        
    return None, None


def find_OF_crossing_pt(img1, img2, num_cycles = 30, num_rnd_lines = 20,
                 delta = 15, method = 'lk', trace = False, path = 'output/'):
    """
    Ransac algorithm gets 2 frames and returns a point of infinity.

    :param img1: source first image, array of coordinates of points.
    :param img2: source second image, array of coordinates of points.
    :param num_cycles: number of cycles in ransac.
    :param num_rnd_lines: size of subset in ransac.
    :param delta: max distance for inliers in ransac.
    :param method: optical flow finding method.
    :param trace: boolean value trace, saving points in file.
    :param path: folder for tracing points.
    
    :return: point of infinity coordinates and algorithm status: 
     np.array([[], []]), True\False
    """

    frame1 = img1.copy()
    frame2 = img2.copy()
    cycle = 0  # iterator
    min_points = 50
    #max length of deque below ???????????????????????????????????????????????????
    dist = deque()  # vector characteristic for all points 
    points_of_cross = deque()

    pts1, pts2 = np.array([]), np.array([]) # Temporary arrays of points, that containing optical flow

    qL = 0.15
    N = 500
    itt = 0

    points = bc.Points()
    
    # Trying to generate points and build optical flow
    while len(pts1) < min_points:
        if itt > 5:  # limit of iteration
            return [np.NaN, np.NaN], False
        pts1, pts2 = find_opt_flow(frame1, frame2, np.power(0.5, itt) * qL, 
                                       N * np.power(2, itt), method=method)
        itt += 1
    points.add_points(pts1, pts2)
    points.make_lines()

    # Generating img with all lines and current trace folder.
    if trace:
        out = img2.copy()
        inliers = points.get_indexes_of_inliers()
        for id in inliers:
            point = points.get_point_by_id(id)
            k, b = point.get_line()
            x1, y1 = point.get_pt1()
            x2, y2 = point.get_pt2()
            out = draw.draw_line(out, k, b, color=draw.red)
            out = draw.draw_point(out, np.array([x1, y1]), color=draw.green)
            out = draw.draw_point(out, np.array([x2, y2]), 
                                                      color=draw.dark_green)
            cv2.imwrite(path + " all_lines.jpg", out)
            
    # Cycles of ransac
    while cycle < num_cycles:
        # Generating subset of lines
        subset = points.get_subset_of_rnd_lines(num_rnd_lines)
        # Draw current subset, if need
        if trace:
            out = img2.copy()
            color = draw.blue
            for s in subset:
                k, b = points.get_point_by_id(s).get_line()
                out = draw.draw_line(out, k, b, color=color)
                
        # Trying to find current point of cross
        pt = points.point_of_crossing(subset)
        if not np.isnan(pt[0]):
            points_of_cross.append(pt)
            dist.append(points.get_sum_dist(pt, subset))
            
            if trace:
                out = draw.draw_point(out, pt, color=draw.red, thickness=1, 
                                                                  radius=10)
                cv2.imwrite(path + str(cycle) + ".jpg", out)
                
        # if there are not so much lines, is sufficient 1 subset(all lines)
        if points.get_number_of_inliers() <= num_rnd_lines:
            break
        cycle = cycle + 1
    # if was a some error
    if len(dist) == 0:
        return [np.NaN, np.NaN], False
    # Main point of cross
    id_temp_point = list(dist).index(min(dist))
    temp_point = points_of_cross[id_temp_point]
    # Marking outliers
    points.check_for_lines_in_interesting_area(temp_point, delta)
    inliers = points.get_indexes_of_inliers()
    pt = points.point_of_crossing(inliers)
    # if wasn't found point of infinity (some error in numpy.linalg.lstsq)
    if np.isnan(pt[0]):
        return pt, False
        
    # Drawing inliers and point of infinity
    if trace:
        out = img2.copy()
        for id in inliers:
            k, b = points.get_point_by_id(id).get_line()
            out = draw.draw_line(out, k, b)
            out = draw.draw_point(out, pt, radius=10, thickness=1)
        cv2.imwrite(path + "result.jpg", out)
        out = img2.copy()
        for i in points.get_indexes_of_inliers():
            point = points.get_point_by_id(i)
        out = draw.draw_point(out, pt, color=draw.blue, thickness=1, radius=2)
        cv2.imwrite(path + 'unit_vector.jpg', out)
        
    return pt, True


def is_speed_ok(s1, s2, threshold_value_of_speed=20):
    if s1 < threshold_value_of_speed or s2 < threshold_value_of_speed:    
        return False
    else:
        return True


def collect_OF_crossing_pts(
               cap, # Capture object
               analysed_frame_nb,
               threshold_value_of_speed = 20, 
               mask = None,
               method = 'lk', 
               trace = False, 
               save_points = False,
               output = 'out/calibration/',
               verbose = True
               ):
    """
    This function collect the crossing point steming from the optical flow, 
    during a certain number of frames set by argument "analysed_frame_nb".
    
    INPUT:
      * cap: an object with at least 3 properties cap.current_frame, 
        cap.current_time and cap.current_speed, and a method cap.capture()
        that updates these properties at each call;
      * analysed_frame_nb: the desired number of frame pair to be processed;
      * threshold_value_of_speed: the speed under which the optical flow 
        will not be taken into account;
      * mask: a 2-list the form [top_left, bottom_right], where top_left
        is the top_left vertex of the desired mask in the image, and 
        bottom_right is its bottom right vertex. If mask is None, then all
        the image is taken into account;
      * method = 'lk': optical flow finding method;
      * trace = False: boolean value trace, saving points in a file;
      * save_points = False : if it is desired to save the OF in a file
      * output = 'out/calibration/': a folder address to store the data
        in the case where save_points has been set to True;
      * verbose = True: be or don't be verbose  
    
    OUTPUT:
      * A list of points representing the computed point at infinity 
       at each frame. The real point at infinity is then very close to be 
       the mean of these points.
    """
                     
    itt = 0
    frame_itt = 0
    file = -1
    n_frames = analysed_frame_nb
    
    if trace and not os.path.exists('out'):
        os.mkdir('out')
    if trace and not os.path.exists('out/calibration/'):
        os.mkdir('out/calibration/')
    if save_points:
        out = os.path.join(output, 'points.txt')
        file = open(out, "w")
        file.close()

    pts_out = np.zeros([n_frames, 2])
    
    cap.capture() #capture current frame and related info    
    frame1 = cap.current_frame
    speed1 = cap.current_speed
    
    if save_points:
        file = open(os.path.join(output, 'points.txt'), "a")
        
    while frame_itt < n_frames:
        
        cap.capture() #capture current frame and related info
        frame2 = cap.current_frame
        if frame2 is None:
            break
        
        speed2 = cap.current_speed
        
        if trace and not os.path.exists('out/calibration/' + str(itt) + '/'):
            os.mkdir('out/calibration/' + str(itt) + '/')
            
        if is_speed_ok(speed1, speed2, threshold_value_of_speed):
            area1 = sf.get_box(frame1, mask)
            area2 = sf.get_box(frame2, mask)
            pt, st = find_OF_crossing_pt(
                            area1, 
                            area2, 
                            method=method, 
                            trace=trace, 
                            path='out/calibration/' + str(itt) + '/') 
        else:
           pt, st = [np.NaN, np.NaN], False
           if verbose:
               print('bad speed', flush = True)

        if st:
            pt = pt + mask[0]
            print('CALIBRATION: {} %'.format(frame_itt / n_frames * 100))
            pts_out[frame_itt, :] = pt
            frame_itt += 1
            
            if save_points:
                file.write("{}|{}|{}\n".format(itt, pt[0], pt[1]))

        frame1 = frame2
        speed1 = speed2
        itt += 1
        
    if save_points:
        file.close()
    return pts_out

#-----------------

def get_pt_at_infinity(
                       cap, 
                       img_dim,
                       analysed_frame_nb, 
                       method = 'lk', 
                       probability_threshold = 0.8,
                       mask_width_ratio = 1/2, 
                       trace = False,
                       save_points = False,
                       verbose = True
                       ):
    
    """
    This function obtains the point at infinity (in the direction of the 
    camera movement). 
    INPUT:
      * cap: an object with at least 3 properties cap.current_frame, 
        cap.current_time and cap.current_speed, and a method cap.capture()
        that updates these properties at each call;
      * img_dim: the dimension of the image;
      * analysed_frame_nb: the desired number of frame pair to be processed;
      * param method = 'lk': optical flow finding method;
      * probability_threshold = 0.8 : a number between 0 and 1, that allow
        refining the point at infinity. The more the number is close to 0, 
        the more the refining is be tight, but also the more the number of 
        samples is small. So, don't decrease it to much.
      * mask_width_ratio = 1/2 : only the optical flow in the mask will be
        taken into account. The height of the mask is the height of the 
        image, and the mask is horizontally centered, of width equal to 
        mask_width_ration-times-the width of the image;
      * trace = False: boolean value trace, saving points in file;
      * save_points = False : if it is desired to save the OF in a file
      * verbose = True: be or don't be verbose

     OUPUT:
      * m : the coordinate of the point at infinity;
      * K : the covariance matrix of all the points at infinity in a number of 
        frames that were used to compute the point at infinity
      * mask :a 2-list the form [top_left, bottom_right], where top_left
        is the top_left vertex of the desired mask in the image, and 
        bottom_right is its bottom right vertex (the mask may be all the image). 
    """


    height, width = img_dim[0:2] # size of frame
    ratio = (1 - mask_width_ratio)/2
    x1, y1, x2, y2 = ratio*width//1, 0, (1-ratio)*width//1, height//1  # points of rectangle that will be analysed in ransac
    pt_rec1 = np.array([x1, y1])
    pt_rec2 = np.array([x2, y2])    
    mask = [pt_rec1, pt_rec2]
    
    pts = collect_OF_crossing_pts(
                                cap, 
                                analysed_frame_nb,
                                method = method,
                                threshold_value_of_speed = 20, 
                                mask = mask,
                                trace = trace, 
                                save_points = save_points,
                                verbose = verbose)

    # if desired, read points from file
    # X, Y, _ = sf.read_stat('out/calibration/points.txt')    

    r_sq = \
        sf.get_mahalanobis_distance_sq_by_probability(probability_threshold)  # getting mahalanobis radius by probability

    m, K = sf.stat(pts)       
    
    # generating of new sample by filter, that used as criterion mahalanubis distance
    good_pts_ind = \
        sf.indexes_of_points_that_mahalanobis_dist_smaller_than(
                                                            pts, m, K, r_sq)
     
    good_pts = pts[good_pts_ind, :]
        
    good_m, good_K = sf.stat(good_pts)  # getting statistics characteristics 
    
    return good_m, good_K, mask


#--------------------------------------------------------------------------

if __name__ == '__main__':
    
    video = 'calibration_video.mp4'  # calibration video
    #video = 'test1.mp4'                # video
    video_length = 700
    frame_rate = 25

    min_analysed_frame_nb = 300  # number of frames which will be used in video analysis.
    
    #-----------------------------    
 
    tempCap = cv2.VideoCapture(video)     
    img_dim = tempCap.read()[1].shape                             
      
    cap = bc.Capture(video)
 
    
    m, K, mask = get_pt_at_infinity(
                       cap, 
                       img_dim,
                       min_analysed_frame_nb, 
                       method = 'lk',
                       mask_width_ratio = 1/3, 
                       trace = False,
                       save_points = False
                       )
       
    #------------
    #plotting:
    K_inv = np.linalg.inv(K)            
    r_sq = sf.get_mahalanobis_distance_sq_by_probability(0.8)  # getting mahalanubis radius by probability
    r = np.sqrt(r_sq)
    path = 'out/'
    if not os.path.exists(path):
        os.mkdir(path)

    inf_pt_list = deque(maxlen = 5000) 

    i = 0
    capture = cv2.VideoCapture(video)  
    is_captured1, f1 = capture.read()
    is_captured2, f2 = capture.read()    
   
    is_captured = is_captured1 and is_captured2
    
    while is_captured:        
        area1 = sf.get_box(f1, mask)
        area2 = sf.get_box(f2, mask)
       
        pt, st = find_OF_crossing_pt(area1, area2, method='lk')        
        print(i, flush = True)
        
        if st:
            pt = pt + mask[0]
            inf_pt_list.append(pt)
            out = f2.copy()
            r_i_sq = sf.get_mahalanobis_distance_sq(pt, m, K_inv)
            r_i = np.sqrt(r_i_sq)
            # drawing point, arrow and text
            text = dict(text='r = ' + str(r_i), font_scale=1, line_type=2, 
                                                            color=draw.blue)
            if r_i_sq > r_sq:
                text['color'] = draw.red
            out = draw.draw_text(out, (0, 50), **text)
            text.pop('text')
            text.pop('color')
            out = draw.draw_text(out, (0, 80), text='x = ' + str(pt[0]), 
                                          color=draw.black, **text)
            out = draw.draw_text(out, (0, 110), 
                        text='y = ' + str(pt[1]), color=draw.black, **text)
            out = draw.draw_rectangle(out, mask, color=draw.cyan)
            out = draw.draw_arrow(out, m, pt)
            out = draw.draw_point(out, pt, radius=3, thickness=6, 
                                                          color=draw.red)
            #  mahalanobis ellipses
            out = draw.draw_mahalanobis_ellipse(out, r, m, K, 
                    color=draw.blue, draw_center_pt = True, 
                    draw_axes = True, draw_extremum_pts = True)
           
            out = draw.draw_mahalanobis_ellipse(out, 3*r, m, K, 
                    color=draw.red, draw_center_pt = False, 
                    draw_axes = False, draw_extremum_pts = False)           

            p = path + '{:05d}.jpg'.format(i)
            cv2.imwrite(p, out)
            
        f1 = f2
        i += 1    
        is_captured, f2 = capture.read()

    #------------------
       
    #Now, we plot all the points on a "0000000.jpg" image.   
    capture = cv2.VideoCapture(video)    
    is_captured, f1 = capture.read()
    out = f1.copy()
    
    for pt in inf_pt_list:
       out = draw.draw_point(out, pt, radius=1, thickness=1, 
                                                      color=draw.red)
    #  mahalanobis ellipses
    out = draw.draw_mahalanobis_ellipse(out, r, m, K, 
                        color=draw.blue, draw_center_pt = True, 
                            draw_axes = True, draw_extremum_pts = True)
    p = path + '0000000.jpg'
    cv2.imwrite(p, out)    
    