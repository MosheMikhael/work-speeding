import numpy as np
import cv2
import random
from collections import deque
import draw

class Point:
    """
        This class keeps a coordinates of point on two frames, properties of line and inlier status.
    """
    def __init__(self, pt1, pt2):
        """
            Initialization.
        """
        self.pt1 = np.array(pt1)
        self.pt2 = np.array(pt2)        
        self.vect = self.pt2 - self.pt1
        self.vect_norm = np.linalg.norm(self.vect)
        if self.vect_norm > 0:
            self.unit_vect = self.vect / self.vect_norm
        else:
            self.unit_vect = None
        self.inlier = False
        self._k = None
        self._b = None


    def is_inlier(self):
        """
            Inlier status.

        :return: inlier value.
        """
        return self.inlier
    
    def is_outlier(self):
        return not self.inlier

#  WARNING, make using vectors!!!
    def distance(self, pt):
        """
            Returns a distance from point to line.

        :param pt: point.
        :return: distance value.
        """
        if self.unit_vect is not None:
            M = np.concatenate(([pt - self.pt1], [self.unit_vect]), axis = 0)
            return(np.abs(np.linalg.det(M)))
        else:
            return None
        

    def make_line(self):
        """
            Make a line coefficients. (y = k*x + b)

        :return: None.
        """
        x1, y1 = self.pt1
        x2, y2 = self.pt2
        self._k = (y2 - y1) / (x2 - x1)
        self._b = y1 - self._k * x1
#        if self.inlier:
#            (x1, y1) = self.pt1
#            if self.vect[0] == 0:
#                self.vect[0] = 0.00000000001 #vertical line case
#                self._k = np.float32(self.vect[1] / self.vect[0])
#                self._b = np.float32(y1 - self._k * x1)            


    def get_pt1(self):
        """
            Returns a first point.

        :return: point.
        """

        return self.pt1


    def get_pt2(self):
        """
            Returns a second point.

        :return: point.
        """

        return self.pt2
 

    def get_line(self):
        """
            Returns a line-coefficients. (y = k*x + b)

        :return: k, b.
        """
        return self._k, self._b


    def check_for_line_in_interesting_area(self, pt, delta):
        """
            Mark line as outlier if it is out of interesting area in delta-area around the point.

        :param pt: point.
        :param delta: delta value.
        """
        dist = self.distance(pt) 
        
        if dist and dist <= delta:
            self.inlier = True
        else:
            self.inlier = False


    def check_for_sufficient_flow(self, delta = 1):
        """
            Mark as outlier if the distance between points on different frames is less than delta.

        :param delta: delta.
        :return: None.
        """
        #if np.sqrt((self._x1 - self._x2) ** 2 + (self._y1 - self._y2) ** 2) < delta:
        if np.linalg.norm(self.vect) < delta:
            self.inlier = False
        else:
            self.inlier = True


    def get_unit_vector(self): 
  
        return self.unit_vect

    def set_inlier(self):
        self.inlier = True
    
    def set_outlier(self):
        self.inlier = False

    def __str__(self):
        """
            For trace this class.

        :return: string.
        """
        x1, y1 = self.pt1
        x2, y2 = self.pt2
        return "pt1=({}, {}) | pt2=({}, {}) | inlier={} | k={} | b={}".\
            format(x1, y1, x2, y2, self.inlier, self._k, self._b)    


class Points:
    """
        This class keeps a set of points between two frames and realised interface of using for this set.
    """
    def __init__(self, max_len = 10000):
        """
            Initialization.
        """
        self._points = deque(maxlen = max_len)
        

    def get_point_by_id(self, id):
        """
            Returns 'Point' object by id.

        :param id: id.
        :return: point.
        """
        return self._points[id]


    def get_number_of_points(self):
        """
            Size of points array.

        :return: size.
        """
        return len(self._points)


    def get_number_of_inliers(self):
        """
            Returns a number of inliers in array of points.

        :return: size.
        """
        counter = 0
        for pt in self._points:
            if pt.is_inlier():
                counter += 1
        
        return counter


    def add_point(self, pt1, pt2):
        """
            Add point to array.

        :param pt1: point coordinates form first frame.
        :param pt2: point coordinates form second frame.
        :return: None.
        """
        point = Point(pt1, pt2)
        self._points = np.append(self._points, point)


    def add_points(self, pts1, pts2):
        """
            Add points to array.

        :param pts1:  array of point coordinates form first frame.
        :param pts2:  array of point coordinates form second frame.
        :return: None.
        """
        for i in range(len(pts1)):
            self.add_point(pts1[i], pts2[i])


    def make_lines(self):
        """
            Mark all points as outlier if the distance between points on different frames is less than delta and then
            if it inlier making line coefficients (y = k*x + b).

        :return: None.
        """
        for i in range(len(self._points)):
            self._points[i].check_for_sufficient_flow()
            self._points[i].make_line()


    def get_indexes_of_inliers(self):
        """
            Return a array of inlier id-s.

        :return: array of id-s.
        """     
        L = len(self._points)
        inliers = deque(maxlen = L)
        for i in range(L):
            if self._points[i].is_inlier():
                inliers.append(i)
                
        return inliers        
        

    def get_subset_of_rnd_lines(self, number_of_rnd_lines=20):
        """
            Getting a array of id-s that randomly selected from inliers.

        :param number_of_rnd_lines: max size of output array.
        :return: array of id-s.
        """
        indexes = self.get_indexes_of_inliers()
        #subset = []
        if self.get_number_of_inliers() < number_of_rnd_lines:
            return indexes

        return random.sample(list(indexes), number_of_rnd_lines)


    def point_of_crossing(self, indexes):
        """
            Сalculating the crossing of lines in subset.

        :param indexes: array of id-s.
        :return: point.
        """
        L = len(indexes)
        M = np.zeros([L, 2]) # memory allocation
        v = np.zeros(L) # idem
        
        if len(indexes) < 2:
            return np.array([np.NaN, np.NaN])

        for i in range(L):
            pt = self.get_point_by_id(indexes[i])
            k, b = pt.get_line()
            M[i][0] = -k
            M[i][1] = 1
            v[i] = b
#            point = self.get_point_by_id(indexes[i])
# 
#            v1 = point.get_pt1()
#            v2 = point.get_pt2()
#            
#            
#            v1_hort = np.array([-v1[1], v1[0]])
#            b_ = v1_hort.dot(v2) / point.vect_norm
#
#            vect_hort = np.array([point.vect[1], -point.vect[0]])
#            m_ = vect_hort / point.vect_norm
#
#            v[i] = b_
#            M[i,:] = m_
            
        out = np.linalg.lstsq(M, v, rcond=-1)[0]

        return out
        
    def get_OF_by_max_dens(self):
        m00 = 0
        m01 = 0
        m10 = 0
        m11 = 0
        b0 = 0
        b1 = 0
        indexes = self.get_indexes_of_inliers()
        print(len(indexes))
        for ind in indexes:
            
            a, b = self._points[ind].get_line()
#            print(a, b)
            z = a ** 2 + 1
            m00 += - a ** 2 / z
            m01 += a / z
            m10 += - a / z
            m11 += 1 / z
            b0 += a * b / z
            b1 += b / z
        M = np.array([[m00, m01], [m10, m11]], np.float32)
        b = np.array([b0, b1], np.float32)
        out = np.linalg.lstsq(M, b, rcond=-1)[0]
        if np.isnan(out[0]):
            return out, False
        return out, True
        dist = []
        for i in range(len(indexes)):
            dist += [(self.get_point_by_id(indexes[i]).distance(out), indexes[i])]
        dist.sort()
        if len(dist) > 10:
            for i in range(10, len(dist)):
                self.get_point_by_id(dist[i][1]).set_outlier()
        m00 = 0
        m01 = 0
        m10 = 0
        m11 = 0
        b0 = 0
        b1 = 0
        indexes = self.get_indexes_of_inliers()
        print(len(indexes))
        for ind in indexes:    
            a, b = self._points[ind].get_line()
            z = a ** 2 + 1
            m00 += - a ** 2 / z
            m01 += a / z
            m10 += - a / z
            m11 += 1 / z
            b0 += a * b / z
            b1 += b / z
        M = np.array([[m00, m01], [m10, m11]], np.float32)
        b = np.array([b0, b1], np.float32)
        out = np.linalg.lstsq(M, b, rcond=-1)[0]
        if np.isnan(out[0]):
            return out, False, None
        
        return out, True, dist[:10]
        
    def get_sum_dist(self, pt, subset):
        """
            Returns a summary distance from point to lines in subset.

        :param pt: current point.
        :param subset: array of id-s lines.
        :return: summary distance.
        """
        dist = 0
        for i in subset:
            dist += self._points[i].distance(pt)
            
        return dist


    def check_for_lines_in_interesting_area(self, pt, delta):
        """
        Marking all lines as outlier if them out of interesting area in delta-area around the point.

        :param pt: interesting point.
        :param delta: delta distance.
        :return:
        """
        for i in range(len(self._points)):
            self.get_point_by_id(i).check_for_line_in_interesting_area(pt, delta)


    def get_middle_unit_vector(self):
        
        out = np.array([0, 0])
        for id in range(self.get_number_of_inliers()):
            point = self.get_point_by_id(id)
            out = (id * out + point.get_unit_vector()) / (id + 1)
            
        return out
    def find_PAI(self):
        self.make_lines()
        inliers = self.get_indexes_of_inliers()
        pt = self.point_of_crossing(inliers)
        if np.isnan(pt[0]):
            return pt, False
        return pt, True
    def find_OF_crossing_pt(self, num_cycles = 30, num_rnd_lines = 20,
                     delta = 15, trace = False, path = 'output/', img = None):
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
    
        cycle = 0  # iterator
        #max length of deque below ???????????????????????????????????????????????????
        dist = deque()  # vector characteristic for all points 
        points_of_cross = deque()
    
        
        self.make_lines()
    
        # Generating img with all lines and current trace folder.
        if trace:
            out = img.copy()
            inliers = self.get_indexes_of_inliers()
            for id in inliers:
                point = self.get_point_by_id(id)
                k, b = point.get_line()
                x1, y1 = point.get_pt1()
                x2, y2 = point.get_pt2()
                out = draw.draw_line(out, k, b, color=draw.red)
                out = draw.draw_point(out, np.array([x1, y1]), color=draw.green)
                out = draw.draw_point(out, np.array([x2, y2]), color=draw.dark_green)
                cv2.imwrite(path + " all_lines.jpg", out)        
                
        # Cycles of ransac
        while cycle < num_cycles:
            # Generating subset of lines
            subset = self.get_subset_of_rnd_lines(num_rnd_lines)
            # Draw current subset, if need
            if trace:
                out = img.copy()
                color = draw.blue
                for s in subset:
                    k, b = self.get_point_by_id(s).get_line()
                    out = draw.draw_line(out, k, b, color=color)
                    
            # Trying to find current point of cross
            pt = self.point_of_crossing(subset)
            if not np.isnan(pt[0]):
                points_of_cross.append(pt)
                dist.append(self.get_sum_dist(pt, subset))
                
                if trace:
                    out = draw.draw_point(out, pt, color=draw.red, thickness=1, 
                                                                      radius=10)
                    cv2.imwrite(path + str(cycle) + ".jpg", out)
                    
            # if there are not so much lines, is sufficient 1 subset(all lines)
            if self.get_number_of_inliers() <= num_rnd_lines:
                break
            cycle = cycle + 1
        # if was a some error
        if len(dist) == 0:
            return [np.NaN, np.NaN], False
        # Main point of cross
        id_temp_point = list(dist).index(min(dist))
        temp_point = points_of_cross[id_temp_point]
        # Marking outliers
        self.check_for_lines_in_interesting_area(temp_point, delta)
        inliers = self.get_indexes_of_inliers()
        pt = self.point_of_crossing(inliers)
        # if wasn't found point of infinity (some error in numpy.linalg.lstsq)
        if np.isnan(pt[0]):
            return pt, False
            
        # Drawing inliers and point of infinity
        if trace:
            out = img.copy()
            for id in inliers:
                k, b = self.get_point_by_id(id).get_line()
                out = draw.draw_line(out, k, b)
                out = draw.draw_point(out, pt, radius=10, thickness=1)
            cv2.imwrite(path + "result.jpg", out)
            out = img.copy()
            for i in self.get_indexes_of_inliers():
                point = self.get_point_by_id(i)
            out = draw.draw_point(out, pt, color=draw.blue, thickness=1, radius=2)
            cv2.imwrite(path + 'unit_vector.jpg', out)
            
        return pt, True

    def mark_inlier_all(self):
        for i in range(self.get_number_of_points()):
                self.get_point_by_id(i).set_inlier()
                
    def __str__(self):
        """
            Trace this class.

        :return: string.
        """
        out = ""
        for i in range(len(self._points)):
            out += str(self._points[i]) + "\n"
            
        return out

class Capture:
    def __init__(self, video):
        self._videoCap = cv2.VideoCapture(video)
        video_length = np.int(self._videoCap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = np.int(self._videoCap.get(cv2.CAP_PROP_FPS))
        self._speeds = [random.randint(0, 120) \
                        for i in range(video_length)]                                 
        self._time_array = np.arange(video_length) / frame_rate        
        self.counter = 0
        
        self.current_frame = None
        self.current_time = None
        self.current_speed = None
            
    def capture(self):
        try:
            is_captured, self.current_frame = self._videoCap.read()
        except:
            self.current_frame = None
        try:
            self.current_time = self._time_array[self.counter]
        except:
            self.current_time = None
        try:
            self.current_speed = self._speeds[self.counter]
        except:
            self.current_speed = None
        
        self.counter += 1            

class Time:
    def __init__(self, h, m, s):
        self.h = h
        self.m = m
        self.s = s
    def get_time(self, sec):
        _s = self.s + sec
        _m = self.m + _s // 60
        _s = int(_s % 60)
        _h = self.h + int(_m // 60)
        _m = int(_m % 60)
        return _h, _m, _s
    def get_sec(self):
        return self.s + 60 * self.m + 3600 * self.h