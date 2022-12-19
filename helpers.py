import numpy as np
import open3d as o3d
import open3d.visualization as viz
import open3d.visualization.rendering as rendering

mesh = o3d.geometry.TriangleMesh # less typing

# ***************************
# * Open3D Helper Functions *
# ***************************

def rotate_obj (obj, x, y, z):
    """
    Rotates an open3d mesh `obj` by x, y, and z.
    """
    R = obj.get_rotation_matrix_from_xyz([x, y, z])
    obj.rotate(R)


def print_material_attributes(material):
  for elt in dir(material):
    if elt[0] == '_': # no dunder attributes
      continue
    print(f"{elt} = {eval(f'material.{elt}')}")
  
def renderPointLight(env, pos, color = [255,255,255], intensity = 60000, falloff = 60):
  """
  Renders `env` (an open3d `OffScreenRenderer`) with one additional point light at the specified `pos`, `intensity`, and `falloff`.
  """
  env.scene.scene.remove_light("rand_point")
  # add_point_light(name, color, position, intensity, falloff, cast_shadows)
  env.scene.scene.add_point_light("rand_point", color, pos, intensity, falloff, True)
  return env.render_to_image()

def convert_to_float(color, norm = 255):
  """
  Takes an 8-bit color and converts it to a float for 32-bit color needs.
  """
  return (np.array(color)/norm).astype('float32')

def convert_from_float(color, norm = 255):
  """
  Takes a 32-bit color and converts it to a float for 8-bit color needs.
  """
  return (np.array(color) * norm).astype('int8')

def create_material(color = [255,255,255,255], makefloat = True, roughness = 1.0):
  """
  Creates a standard "defaultLit" material using the specified `color` and `roughness`.
  """
  if makefloat:
    color = convert_to_float(color)
  
  out = rendering.MaterialRecord()
  out.shader = 'defaultLit' # makes the material responsive to light
  out.base_color = color # 32-Bit (R,G,B,A)
  out.base_roughness = roughness
  return out

def make_plane(length = 1, width = None):
  """
  Creates a plane of side length `length` (and an optional side length `width`) for rendering
  """
  plane_depth = .05 # you can't have a depth of 0, since the create_box function throws an error.

  if width is None:
    plane = mesh.create_box(length, plane_depth, length)

  else:
    plane = mesh.create_box(length, plane_depth, width)

  plane.compute_vertex_normals(normalized=True)
  return plane

# *******************
# * Image Comparing *
# *******************

from scipy.stats import multivariate_normal as mult_norm

def gaussian_compare(observed, baseline, scale = 1, log = False):
  """
  Returns product of gaussian distributions at each element at `observed` centered at each element in `baseline`.
  (also known as the multivariate normal distribution centered at `baseline`)
  """
  if log:
    return mult_norm.logpdf(observed.ravel(), mean = baseline.ravel(), cov = scale) 
  
  return mult_norm.pdf(observed.ravel(), mean = baseline.ravel(), cov = scale)

def gauss_image_likelihood(observed, truth, scale = 1, log = False, grayscale = False):
  """
  Image Likelihood on RGB Images is the product of Gaussian distributions on each of the R G and B values at each pixel.
  `grayscale` determines if image RGB values are averaged before comparing.
  `log` determines if log likelihood is computed instead of standard likelihood.
  """
  o_arr = np.asarray(observed)
  t_arr = np.asarray(truth)

  if grayscale:
    o_arr = np.average(o_arr, 2)
    t_arr = np.average(t_arr, 2)
  
  if log:
    out = 0
    update = lambda x, y: x + y

  else:
    out = 1
    update = lambda x, y: x * y

  for i in range(o_arr.shape[0]): # iterate over rows to save memory
    out = update(out, gaussian_compare(o_arr[i], t_arr[i], scale = scale, log = log))
  
  return out

def img_diff(img1, img2):
  """ Computes Manhattan Pixel-Wise difference between two images. """
  # calculate the difference and its norms
  diff = np.asarray(img1) - np.asarray(img2)
  m_norm = np.sum(np.abs(diff)) # Manhattan norm
  return m_norm

# ************
# * PLOTTING *
# ************

def o3d_plot(points, target, geometries = []):
  o3d_points=o3d.utility.Vector3dVector(points)
  point_cloud = o3d.geometry.PointCloud(o3d_points)
  original_point = mesh.create_sphere(radius = .04, resolution = 2)
  original_point.translate(target, False)
  original_point.paint_uniform_color(convert_to_float([30, 10, 200]))

  point_cloud.paint_uniform_color(convert_to_float([100, 123, 60]))
  viz.draw_plotly([point_cloud, original_point].extend(geometries), mesh_show_wireframe = False)


# importing required libraries
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.ticker as plticker

def pyplot_3d(point_coords, vals, target, bounds = None, step = .5, minor = False, res = 150, zdir = 'z'):
  """
  Graphs `points` in 3d using matplotlib.
  If `bounds` is specified, the x, y, and z axes fall within those bounds with ticks controlled by `step` and `minor`. Otherwise, the default scatter plot bounds are used.
  `point_coords` is a list of 3-length arrays.
  `vals` should be an array of the same length as point_coords, corresponding to values for that particular point.
  `target` is a point (x,y,z), that is graphed as a star.
  `step` determines the tickmark intervals across the graph.
  `minor` controls if minor tick marks appear.
  """

  # transform tuple of points into n x 3 array of x, y, z coords
  viridis = mpl.colormaps['viridis'].resampled(8) # create colormap
  norm = mpl.colors.Normalize(vmin=vals[0], vmax=vals[-1]) # create normalization from the minimum and maximum values

  # creating figure
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  xs = point_coords[:,0]
  ys = point_coords[:,1]
  zs = point_coords[:,2]

  # creating the plot
  plot = ax.scatter3D(*target, color='#29438D', s = 100, marker = '*', zdir = zdir)
  ax.scatter3D(xs, ys, zs, s = 2, c = vals, norm = norm, cmap = viridis, zdir = zdir)


  # setting title and labels
  ax.set_xlabel('X')
  
  if zdir.lower() == 'y':
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
  
  else:
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
  


  # Limits and Ticks
  if bounds is not None:
    lower = bounds[0]
    higher = bounds[1]
    ax.set_xlim(lower,higher)
    ax.set_ylim(lower,higher)
    ax.set_zlim(lower,higher)

    loc = plticker.MultipleLocator(base = step) # puts ticks at regular intervals
    loc_minor = plticker.MultipleLocator(base = step/2) 
    
    # put major ticks
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    ax.zaxis.set_major_locator(loc)

    if minor: # put minor ticks
      ax.xaxis.set_minor_locator(loc_minor)
      ax.yaxis.set_minor_locator(loc_minor)
      ax.zaxis.set_minor_locator(loc_minor)

  fig.suptitle("MCMC Inference over Light Positions", x = .57)
  # Creating Colorbar
  cb = fig.colorbar(plot, pad = .11)
  cb.set_label("Normalized Point Score", labelpad=10, size = 8)

  # displaying the plot
  plt.tight_layout()
  fig.set_dpi(res)
  plt.show()