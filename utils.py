import argparse
import cv2
import torch
import numpy as np
import pandas as pd

def crop_center(image):
    # crop the center of an image and matching the height with the width of the image
    shape = image.shape[:-1]
    max_size_index = np.argmax(shape)
    diff1 = abs((shape[0] - shape[1]) // 2)
    diff2 = shape[max_size_index] - shape[1 - max_size_index] - diff1
    return image[:, diff1: -diff2] if max_size_index == 1 else image[diff1: -diff2, :]


def get_dtype():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev)
    if dev == 'cuda':
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    print(f'Using device {device}')
    return dtype


def get_video_properties(video):
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # get videos properties
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        length = int(video.get(cv2.cv.CAP_PROP_FRAME_COUNT))
        v_width = int(video.get(cv2.cv.CAP_PROP_FRAME_WIDTH))
        v_height = int(video.get(cv2.cv.CAP_PROP_FRAME_HEIGHT))
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        v_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, length, v_width, v_height


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_stickman_line_connection():
    # stic kman line connection with keypoints indices for R-CNN
    line_connection = [
        (7, 9), (7, 5), (10, 8), (8, 6), (6, 5), (15, 13), (13, 11), (11, 12), (12, 14), (14, 16), (5, 11), (12, 6)
    ]
    return line_connection


def interpolation(coords):
    coords = coords.copy()
    x, y = [x[0] if x[0] is not None else np.nan for x in coords], [x[1] if x[1] is not None else np.nan for x in coords]



    xxx = np.array(x)  # x coords
    yyy = np.array(y)  # y coords

    nons, yy = nan_helper(xxx)
    xxx[nons] = np.interp(yy(nons), yy(~nons), xxx[~nons])
    nans, xx = nan_helper(yyy)
    yyy[nans] = np.interp(xx(nans), xx(~nans), yyy[~nans])

    newCoords = [*zip(xxx, yyy)]

    return newCoords

def diff_xy(coords):
    coords = coords.copy()
    diff_list = []
    for i in range(0, len(coords) - 1):
        if coords[i] is not None and coords[i + 1] is not None:
            point1 = coords[i]
            point2 = coords[i + 1]
            diff = [abs(point2[0] - point1[0]), abs(point2[1] - point1[1])]
            diff_list.append(diff)
        else:
            diff_list.append(None)

    xx, yy = np.array([x[0] if x is not None else np.nan for x in diff_list]), np.array(
        [x[1] if x is not None else np.nan for x in diff_list])

    return xx, yy

def remove_outliers(x, y, coords):
  ids = set(np.where(x > 50)[0]) & set(np.where(y > 50)[0])
  for id in ids:
    left, middle, right = coords[id-1], coords[id], coords[id+1]
    if left is None:
      left = [0]
    if  right is None:
      right = [0]
    if middle is None:
      middle = [0]
    MAX = max(map(list, (left, middle, right)))
    if MAX == [0]:
      pass
    else:
      try:
        coords[coords.index(tuple(MAX))] = None
      except ValueError:
        coords[coords.index(MAX)] = None


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        # >>> # linear interpolation of NaNs
        # >>> nans, x= nan_helper(y)
        # >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]

def from_2d_array_to_nested( X, index=None, columns=None, time_index=None, cells_as_numpy=False ):

    if (time_index is not None) and cells_as_numpy:
        raise ValueError(
            "`Time_index` cannot be specified when `return_arrays` is True, "
            "time index can only be set to "
            "pandas Series"
        )
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    container = np.array if cells_as_numpy else pd.Series

    # for 2d numpy array, rows represent instances, columns represent time points
    n_instances, n_timepoints = X.shape

    if time_index is None:
        time_index = np.arange(n_timepoints)
    kwargs = {"index": time_index}

    Xt = pd.DataFrame(
        pd.Series([container(X[i, :], **kwargs) for i in range(n_instances)])
    )
    if index is not None:
        Xt.index = index
    if columns is not None:
        Xt.columns = columns
    return Xt