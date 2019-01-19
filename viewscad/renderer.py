import solid
import os
import tempfile
import platform
import traceback
from IPython.display import display
from ipywidgets import HTML, Text, Output, VBox
from traitlets import link, dlink
import math as pymath
import hashlib
import time
import subprocess
import pythreejs as pjs
from IPython.display import display, SVG
import numpy as np

def col_from_hex(c):
    return [int(c[1:3], 16) / 256, int(c[3:5], 16) / 256, int(c[5:7], 16) / 256]

OBJ_COLOR = '#f7d62c'
#OBJ_COLOR = '#156289'
OBJ_RGB = col_from_hex(OBJ_COLOR)
SELECTED_FACE_COLOR = '#ff0000'
SELECTED_FACE_RGB = col_from_hex(SELECTED_FACE_COLOR)
SELECTED_EDGE_COLOR = '#6666ff'
SELECTED_EDGE_COLOR_INT = int(SELECTED_EDGE_COLOR[1:], 16)
SELECTED_VERTEX_COLOR = '#00ff00'
SELECTED_VERTEX_RGB = col_from_hex(SELECTED_VERTEX_COLOR)
BACKGROUND_COLOR = '#ffffff'
DEFAULT_FN = 20
DEBUG = False

def rotate_around_mx(v, radians):
    v_norm = v / np.sqrt(v.dot(v))
    mx = np.array([[0, -v_norm[2], v_norm[1]], [v_norm[2], 0, -v_norm[0]], [-v_norm[1], v_norm[0], 0]])
    return np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.]]) + np.sin(radians) * mx + 2 * (np.sin(radians/2)**2) * mx @ mx

def rotate_into_mx(v1, v2):
    v1n = np.array(v1)
    v1l = np.sqrt(v1n.dot(v1n))
    if v1l < 1.0E-9:
        return np.identity(3)
    v1n /= v1l
    v2n = np.array(v2)
    v2l = np.sqrt(v2n.dot(v2n))
    if v2l < 1.0E-9:
        return np.identity(3)
    v2n /= v2l
    v3 = np.cross(v1n, v2n)
    l = np.sqrt(v3.dot(v3))
    if l < 1.0E-9:
        if v1.dot(v2) > 0:
            return np.identity(3)
        else:
            for i in range(3):
                rot_axis = np.zeros(3)
                rot_axis[0] = 1.0
                if np.abs(rot_axis.dot(v1)) > 1.0E-6:
                    break
            v_parallel = rot_axis.dot(v1)
            v1n = v1 / np.sqrt(v1.dot(v1))
            rot_axis -= v1n * v1n.dot(rot_axis)
            return rotate_around_mx(rot_axis, np.pi)            
    cosang = v1n.dot(v2n)
    ang = np.arccos(cosang)
    
    return rotate_around_mx(v3, ang)
    
# return an OpenSCADObject rotation to bring v1 into v2 by rotating about an axis
# perpendicular to both:
def rotate_into(v1, v2):
    v1n = np.array(v1)
    v1l = np.sqrt(v1n.dot(v1n))
    if v1l < 1.0E-9:
        return solid.rotate(0.0)
    v1n /= v1l
    v2n = np.array(v2)
    v2l = np.sqrt(v2n.dot(v2n))
    if v2l < 1.0E-9:
        return solid.rotate(0.0)
    v2n /= v2l
    v3 = np.cross(v1n, v2n)
    l = np.sqrt(v3.dot(v3))
    if l < 1.0E-9:
        if v1.dot(v2) > 0:
            return solid.rotate(0.0)
        else:            
            for i in range(3):
                rot_axis = np.zeros(3)
                rot_axis[0] = 1.0
                if np.abs(rot_axis.dot(v1)) > 1.0E-6:
                    break
            v_parallel = rot_axis.dot(v1)
            v1n = v1 / np.sqrt(v1.dot(v1))
            rot_axis -= v1n * v1n.dot(rot_axis)
            return solid.rotate(180.0, rot_axis)
    cosang = v1n.dot(v2n)
    ang = np.arccos(cosang)
    
    return solid.rotate(ang * 180.0 / np.pi, v=v3)

def make_arrow(cyl_mesh, head_mesh, start_vec, end_vec, width, head_width, head_length, color):
    start_vec = np.array(start_vec)
    end_vec = np.array(end_vec)
    v = end_vec - start_vec
    l = np.sqrt(v.dot(v))
    v_norm = v / l
    cyl_height = l - head_length
    z_rot = np.arccos(v_norm.dot(np.array([0.0, 1.0, 0.0])))
    y_rot = np.arctan2(v[2], v[0])
    
    cyl_center = start_vec + cyl_height/2 * v_norm
    head_center = start_vec + (cyl_height + head_length/2) * v_norm
    cyl_geom = pjs.CylinderGeometry(radiusTop=width, radiusBottom=width, height=cyl_height)
    cyl_mesh.geometry = cyl_geom
    cyl_mesh.material = pjs.MeshLambertMaterial(color=color)
    cyl_mesh.position = cyl_center.tolist()
    cyl_mesh.setRotationFromMatrix([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    cyl_mesh.rotateY(-y_rot)
    cyl_mesh.rotateZ(-z_rot)
    
    head_geom = pjs.CylinderGeometry(radiusTop=0.0, radiusBottom=head_width, height=head_length)
    head_mesh.geometry = head_geom
    head_mesh.material = pjs.MeshLambertMaterial(color=color)
    head_mesh.position = head_center.tolist()
    head_mesh.setRotationFromMatrix([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    head_mesh.rotateY(-y_rot)
    head_mesh.rotateZ(-z_rot)


class RenderedObject:
    def __init__(self, stl_file):
        self.faces = []
        self.vertices = []
        self.normals = []
        self.scad_digest = None
        vert_str_to_index = {}
        ind = 0
        for ln in stl_file:
            if ln.find('facet normal') >= 0:
                normal = [float(x) for x in ln.split()[2:]]
                self.normals.append(normal)
            if ln.find('outer loop') >= 0:
                cur_face = []
            elif ln.find('vertex') >= 0:
                vert_str = ln.split('vertex ')[-1]
                if vert_str not in vert_str_to_index:
                    vert_str_to_index[vert_str] = ind
                    self.vertices.append([float(x) for x in vert_str.split()])
                    v_ind = ind
                    ind += 1
                else:
                    v_ind = vert_str_to_index[vert_str]
                cur_face.append(v_ind)
            elif ln.find('endloop') >= 0:
                self.faces.append(cur_face)    
        self.faces = np.array(self.faces)
        self.vertices = np.array(self.vertices)
        self.face_verts = self.vertices[self.faces] # indices face #, face-vertex-#, coord #
        self.nFaces = self.face_verts.shape[0]
        self.plot_verts = self.face_verts.reshape((self.nFaces * 3, 3)).astype('float32')
        self.base_cols = np.tile(np.array(OBJ_RGB), (self.nFaces * 3, 1)).astype('float32')
        self.v1 = self.face_verts[:, 1, :] - self.face_verts[:, 0, :]
        self.v2 = self.face_verts[:, 2, :] - self.face_verts[:, 1, :]
        self.face_normals = np.cross(self.v1, self.v2)
        norm_lens = np.sqrt(np.sum(self.face_normals**2, axis=1)[:, np.newaxis])
        norm_lens = np.where(norm_lens > 1.0E-8, norm_lens, 1.0)
        self.face_normals /= norm_lens
        self.face_normals = np.repeat(self.face_normals, 3, axis=0)
                                 
        

class Renderer:
    '''This class will render an OpenSCAD object within a jupyter notebook.
    The 'pythreejs' module must be installed (see directions a
    https://github.com/jupyter-widgets/pythreejs).

    This class needs to know the path to the openscad command-line tool.
    You can set the path with the OPENSCAD_EXEC environment variable, or with the 'openscad_exec'
    keyword in the constructor.  If these are omitted, the class makes an attempt at
    finding the executable itself.
    other keyword arguments: 'width', 'height', 'draw_grids' (True/False), and 'grid_lines_width' (default 1)
    Primarily this class is used to render a SolidPython object in a Jupyter window, but it
    also can be used to create a 3D object file (STL/OFF/CSG/DXF)
    directly by calling Renderer.render(outfile='outfilename').
    '''
    def __init__(self, **kw):
        self.openscad_exec = None
        self.openscad_tmp_dir = None
        if 'OPENSCAD_EXEC' in os.environ: self.openscad_exec = os.environ['OPENSCAD_EXEC']
        if 'OPENSCAD_TMP_DIR' in os.environ: self.openscad_tmp_dir = os.environ['OPENSCAD_TMP_DIR']
        if 'openscad_exec' in kw: self.openscad_exec = kw['openscad_exec']
        if self.openscad_exec is None:
            self._try_detect_openscad_exec()
        if self.openscad_exec is None:
            raise Exception('openscad exec not found!')
        self.width = kw.get('width', 600)
        self.height = kw.get('height', 600)
        self.draw_grids = kw.get('draw_grids', True)
        self.grid_lines_width = kw.get('grid_lines_width', 1)

    def _try_executable(self, executable_path):
        if os.path.isfile(executable_path):
            self.openscad_exec = executable_path

    def _try_detect_openscad_exec(self):
        self.openscad_exec = None
        platfm = platform.system()
        if platfm == 'Linux':
            self._try_executable('/usr/bin/openscad')
            if self.openscad_exec is None:
                self._try_executable('/usr/local/bin/openscad')
        elif platfm == 'Darwin':
            self._try_executable('/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD')
        elif platfm == 'Windows':
            self._try_executable(os.path.join(
                    os.environ.get('Programfiles(x86)','C:'),
                    'OpenSCAD\\openscad.exe'))

    def _conv_stl(self, stl_file_name):
        fl = open(stl_file_name)
        faces = []
        vertices = []
        vert_str_to_index = {}
        ind = 0
        for ln in fl:
            if ln.find('outer loop') >= 0:
                cur_face = []
            elif ln.find('vertex') >= 0:
                vert_str = ln.split('vertex ')[-1]
                if vert_str not in vert_str_to_index:
                    vert_str_to_index[vert_str] = ind
                    vertices.append([float(x) for x in vert_str.split()])
                    v_ind = ind
                    ind += 1
                else:
                    v_ind = vert_str_to_index[vert_str]
                cur_face.append(v_ind)
            elif ln.find('endloop') >= 0:
                faces.append(cur_face)

        return vertices, faces

    def _get_extents(self, vertices):
        extents = np.zeros((3, 2))
        extents[:, 0] = np.min(vertices, axis=0)
        extents[:, 1] = np.max(vertices, axis=0)
        
        return extents

    def _get_grid_lines(self, axis1, start, step, N, axis2, start2, end2):
        w = start
        vertices = []
        for i in range(N):
            pt1 = [0, 0, 0]
            pt1[axis1] = start + i * step
            pt1[axis2] = start2
            pt2 = [0, 0, 0]
            pt2[axis1] = start + i * step
            pt2[axis2] = end2
            vertices.append(pt1)
            vertices.append(pt2)
        return vertices

    def _get_grids(self, obj_vertices):
        extents = self._get_extents(obj_vertices)
        grid_verts = []
        deltas = [extent[1] - extent[0] for extent in extents]
        max_extent = max(deltas)
        space1 = 10.0**pymath.floor(pymath.log(max_extent) / pymath.log(10.0) - 0.5)
        space2 = 2 * 10.0**pymath.floor(pymath.log(max_extent / 2.0) / pymath.log(10.0) - 0.5)
        space = space2
        if max_extent / space2 < 5: space = space1
        N = int(pymath.floor(max_extent / space + 2.0))
        grid_cols = []
        axis_cols = ['#ff3333', '#33ff33', '#3333ff']
        ends = []
        for axis1 in range(3):
            start = pymath.floor(extents[axis1][0] / space) * space
            ends.append(start + space * N)
            for axis2 in range(3):
                axis3 = [x for x in [0,1,2] if x not in [axis1, axis2]][0]
                if axis1 == axis2: continue
                delta = extents[axis1][1] - extents[axis1][0]
                
                start2 = pymath.floor(extents[axis2][0] / space) * space
                end2 = start2 + (N - 1) * space
                verts = self._get_grid_lines(axis1, start, space, N, axis2,
                                             start2, end2)
                grid_verts.extend(verts)
                grid_cols.extend([axis_cols[axis3] for vert in verts])
                
        # now draw the X,Y,Z labels:
        char_width = max_extent * 0.05
        char_lines = []
        # X:
        char_lines_x = []
        char_lines_x.append([[0.0, 0.0], [1.0, 1.0]])
        char_lines_x.append([[0.0, 1.0], [1.0, 0.0]])
        char_lines.append(char_lines_x)
        # Y:
        char_lines_y = []
        char_lines_y.append([[0.5, 0.0], [0.5, 0.5]])
        char_lines_y.append([[0.5, 0.5], [0.0, 1.0]])
        char_lines_y.append([[0.5, 0.5], [1.0, 1.0]])
        char_lines.append(char_lines_y)
        # Z:
        char_lines_z = []
        char_lines_z.append([[1.0, 1.0], [0.0, 1.0]])
        char_lines_z.append([[0.0, 1.0], [1.0, 0.0]])
        char_lines_z.append([[1.0, 0.0], [0.0, 0.0]])
        char_lines.append(char_lines_z)

        for iaxis in range(3):
            ax1 = [0, 1, 2][iaxis]
            ax2 = [2, 2, 1][iaxis]
            char_lns = char_lines[iaxis]
            segs = [[[0,0], [ends[iaxis] + char_width, 0]],
                   [[ends[iaxis] + char_width, 0], [ends[iaxis] + 0.5 * char_width, 0.5 * char_width]],
                   [[ends[iaxis] + char_width, 0], [ends[iaxis] + 0.5 * char_width, -0.5 * char_width]]]
            for seg in segs:
                for pt in seg:
                    pt3 = [0, 0, 0]
                    pt3[ax1] += pt[0]
                    pt3[ax2] += pt[1]                    
                    grid_verts.append(pt3)
                    grid_cols.append('#000000')

            for seg in char_lns:
                for pt in seg:
                    pt3 = [0, 0, 0]
                    pt3[iaxis] += ends[iaxis] + 2 * char_width
                    pt3[ax1] += pt[0] * char_width
                    pt3[ax2] += 1.2 * (pt[1] - 0.5) * char_width
                    grid_verts.append(pt3)
                    grid_cols.append('#000000')
            
        lines_geom = pjs.Geometry(vertices=grid_verts, colors =grid_cols)
        lines = pjs.LineSegments(geometry=lines_geom,
                 material=pjs.LineBasicMaterial(linewidth=self.grid_lines_width, transparent=True,
                 opacity=0.5, dashSize=10,
                 gapSize=10, vertexColors='VertexColors'),
                 type='LinePieces')

        return lines, space

    def render_to_file(self, openscad_str, fl_name, **kw):
        scad_prepend = ''        
        if 'dollar_sign_vars' in kw:
            for var_name, value in kw['dollar_sign_vars'].items():
                scad_prepend += '${}={};\n'.format(var_name, value)
        else:
            if not kw.get('rough', False):
                scad_prepend += '$fn=120;\n'
                
        scad_tmp_file = os.path.join(self.tmp_dir, 'tmp.scad')
        try:
            of = open(scad_tmp_file, 'w')
            of.write(scad_prepend)
            of.write(openscad_str)
            of.close()

            # now run openscad to generate stl:
            cmd = [self.openscad_exec, '-o', fl_name, scad_tmp_file]
            out = subprocess.check_output(cmd)
            if out != b'': print(out)
            #if return_code < 0:
            #    raise Exception('openscad command line returned code {}'.format(return_code))
        except Exception as e:
            raise e
        finally:
            if os.path.isfile(scad_tmp_file):
                os.remove(scad_tmp_file)        

    def render(self, in_obj, **kw):
        if 'dollar_sign_vars' not in kw:
            kw['dollar_sign_vars'] = {'fn': DEFAULT_FN}
        else:
            if 'fn' not in kw['dollar_sign_vars']:
                kw['dollar_sign_vars']['fn'] = DEFAULT_FN
                
        if isinstance(in_obj, solid.OpenSCADObject):
            kw['scad_object'] = in_obj
            self.create_rendered_obj_if_needed(in_obj, **kw)
            if hasattr(in_obj, 'rendered_object'):
                self._render_obj(in_obj.rendered_object, **kw)
            
        elif isinstance(in_obj, str):
            self.render_openscad_str(in_obj, **kw)
            
    def render_stl(self, stl_fname):
        rendered_obj = RenderedObject(open(stl_fname))
        self._render_obj(rendered_obj)
        
    def _get_digest(self, scad_str):
        hasher = hashlib.md5()
        hasher.update(scad_str.encode('utf-8'))
        return hasher.digest()
    
    def get_vertex(self, openSCADObj, vert_num):
        self.create_rendered_obj_if_needed(openSCADObj)
        return openSCADObj.rendered_object.face_verts[vert_num//3, vert_num%3]
    
    def get_edge_vec(self, openSCADObj, edge_num):
        self.create_rendered_obj_if_needed(openSCADObj)
        face_num = edge_num//3
        v1ind = edge_num - face_num*3
        v1 = openSCADObj.rendered_object.face_verts[face_num, v1ind]
        v2 = openSCADObj.rendered_object.face_verts[face_num, (v1ind+1)%3]
        return v2 - v1
    
    def get_face_norm(self, openSCADObj, face_num):
        self.create_rendered_obj_if_needed(openSCADObj)
        return openSCADObj.rendered_object.face_normals[face_num*3]
    
    def get_face_centroid(self, openSCADObj, face_num):
        self.create_rendered_obj_if_needed(openSCADObj)
        return openSCADObj.rendered_object.face_verts[face_num].mean(axis=0)
    
    def rotate_face_into(self, openSCADObj, face_num, v):
        return rotate_into(self.get_face_norm(openSCADObj, face_num), v)
    
    def rotate_face_down(self, openSCADObj, face_num):
        return self.rotate_face_into(openSCADObj, face_num, [0.0, 0.0, -1.0])
    
    def orient_relative(self, **kw):
        '''
        This returns a function which produces a transformation (translation+rotation) of scadObj1
        so that:
        (1) u_vec1 becomes parallel to u_vec2, 
        (2) the projection of v_vec1_prime onto the plane perpendicular to u_vec is rotated with respect to 
          the projection of v_vec2 onto that plane by azimuth_deg, and
        (3) origin1 is translated into origin2.
        
        '''
        scadObj1 = kw['obj1']
        scadObj2 = kw['obj2']
        origin1 = kw['point1']
        origin2 = kw['point2']
        u_vec1 = kw['align_vec1']
        u_vec2 = kw['align_vec2']
        do_azimuth = False
        if 'azimuth_vec1' in kw and 'azimuth_vec2' in kw:
            v_vec1 = kw['azimuth_vec1']
            v_vec2 = kw['azimuth_vec2']
            do_azimuth = True
            azimuth_deg = kw.get('azimuth_deg', 0.0)
                    
        u_vec1 = np.array(u_vec1)        
        u_vec2 = np.array(u_vec2)        
            
        trans1 = solid.translate(-origin1)
        rot1 = rotate_into(u_vec1, u_vec2)
        mx1 = rotate_into_mx(u_vec1, u_vec2)
        
        if do_azimuth:
            v_vec1 = np.array(v_vec1)
            v_vec2 = np.array(v_vec2)
            v_vec1_prime = mx1 @ v_vec1
            u_vec_norm = u_vec2 / np.sqrt(u_vec2.dot(u_vec2))
            v_vec1_perp_u = v_vec1_prime - u_vec_norm * u_vec_norm.dot(v_vec1_prime)
            l = np.sqrt(v_vec1_perp_u.dot(v_vec1_perp_u))
            if l < 1.0E-9: 
                raise Exception('v_vec1 appears to be parallel to u_vec1!')
            v = v_vec1_perp_u / l
            v_vec2_perp_u = v_vec2 - u_vec_norm * u_vec_norm.dot(v_vec1_prime)
            l = np.sqrt(v_vec2_perp_u.dot(v_vec2_perp_u))
            if l < 1.0E-9: 
                raise Exception('v_vec2 appears to be parallel to u_vec2!')
            x = v_vec2_perp_u / l
            z = u_vec_norm
            y = np.cross(z, x)
            x_comp = x.dot(v)
            y_comp = y.dot(v)
            current_azimuth_radians = np.arctan2(y_comp, x_comp)
            rotate_angle_deg = azimuth_deg - current_azimuth_radians * 180 / np.pi
            rot2 = solid.rotate(rotate_angle_deg, v=z)
        else:
            rot2 = solid.rotate(0.0)
            
        trans2 = solid.translate(origin2)
        transformation = lambda o: trans2 (rot2 (rot1 (trans1 (o))))
        
        return transformation
        
    def place_on(self, obj1, obj2, **kw):
        if 'vertex1' in kw:
            origin1 = self.get_vertex(obj1, kw['vertex1'])
        elif 'point1' in kw:
            origin1 = kw['point1']
        else:
            raise Exception('There must be a "vertex1" or "origin1" keyword!')
        if 'vertex2' in kw:
            origin2 = self.get_vertex(obj2, kw['vertex2'])
        elif 'point2' in kw:
            origin2 = kw['point2']
        else:
            raise Exception('There must be a "vertex2" or "origin2" keyword!')    
            
        if 'align_edge1' in kw:
            align_vec1 = self.get_edge_vec(obj1, kw['align_edge1'])
        elif 'align_face1' in kw:
            align_vec1 = -self.get_face_norm(obj1, kw['align_face1'])
        elif 'align_vec1' in kw:
            align_vec1 = kw['align_vec1']
        else:
            raise Exception('There must be a "align_edge1" or "align_face1" or "align_vec1" keyword!')
        if 'align_edge2' in kw:
            align_vec2 = self.get_edge_vec(obj2, kw['align_edge2'])
        elif 'align_face2' in kw:
            align_vec2 = self.get_face_norm(obj2, kw['align_face2'])
        elif 'align_vec2' in kw:
            align_vec2 = kw['align_vec2']
        else:
            raise Exception('There must be a "align_edge1" or "align_face1" or "align_vec1" keyword!')    
            
        align_vec1 = np.array(align_vec1)
        align_vec2 = np.array(align_vec2)
        
        do_azimuth = True
        if 'azimuth_edge1' in kw:
            azimuth_vec1 = self.get_edge_vec(obj1, kw['azimuth_edge1'])
        elif 'azimuth_face1' in kw:
            azimuth_vec1 = self.get_face_norm(obj1, kw['azimuth_face1'])
        else:
            do_azimuth = False
                        
        if 'azimuth_edge2' in kw:
            azimuth_vec2 = self.get_edge_vec(obj2, kw['azimuth_edge2'])
        elif 'azimuth_face2' in kw:
            azimuth_vec2 = self.get_face_norm(obj2, kw['azimuth_face2'])
        else:
            do_azimuth = False               
            
        azimuth_deg = 0.0
        if do_azimuth:
            azimuth_vec1 = np.array(azimuth_vec1)
            azimuth_vec2 = np.array(azimuth_vec2)
            if 'azimuth_deg' in kw: 
                azimuth_deg = kw.get('azimuth_deg')
            elif 'azimuth_rad' in kw:
                azimuth_deg = 180 / np.pi * kw.get('azimuth_rad')

        fudge_amount = kw.get('fudge_amount', 1.0E-6)
        
        align_vec2_norm = align_vec2 / np.sqrt(align_vec2.dot(align_vec2))
        
        if do_azimuth:
            trans = self.orient_relative(obj1=obj1, obj2=obj2, point1=origin1, point2=origin2, 
                                        align_vec1=align_vec1, align_vec2=align_vec2,
                                        azimuth_vec1=azimuth_vec1, azimuth_vec2=azimuth_vec2, azimuth_deg=azimuth_deg)
        else:
            trans = self.orient_relative(obj1=obj1, obj2=obj2, point1=origin1, point2=origin2, 
                                         align_vec1=align_vec1, align_vec2=align_vec2)
            
        obj1_moved = solid.translate(-fudge_amount * align_vec2_norm) (trans(obj1))
        
        return obj1_moved + obj2       
                
    def create_rendered_obj_if_needed(self, openSCADObj, **kw):
        need_create_obj = False        
        scad_str = None
        if DEBUG:
            print('create_rendered kw:', kw)
        if 'outfile' in kw: 
            need_create_obj = True
        if not hasattr(openSCADObj, 'rendered_object'):
            if DEBUG:
                print('OpenSCADObject does not have a rendered_object, need to create.')
            need_create_obj = True
        else:
            scad_str = solid.scad_render(openSCADObj)
            digest = self._get_digest(scad_str)
            if digest != openSCADObj.rendered_object.scad_digest:
                if DEBUG:
                    print('OpenSCADObject.rendered_object digest does not match, need to recreate.')
                need_create_obj = True
                
        if need_create_obj:
            if scad_str is None:
                scad_str = solid.scad_render(openSCADObj)
            rend_obj = self.render_openscad_str(scad_str, **kw)
            if rend_obj is not None:
                openSCADObj.rendered_object = rend_obj
                openSCADObj.rendered_object.scad_digest = self._get_digest(scad_str)
            else:
                if hasattr(openSCADObj, 'rendered_object'):
                    delattr(openSCADObj, 'rendered_object')                    
                
    
    def render_openscad_str(self, openscad_str, **kw):

        if self.openscad_tmp_dir is not None:
            self.tmp_dir = self.openscad_tmp_dir
        else:
            self.tmp_dir = tempfile.mkdtemp()
        self.saved_umask = os.umask(0o077)        
        return_obj = None
        
        do_tmp_file = True
        if 'outfile' in kw:
            do_tmp_file = False
            openscad_out_file = kw['outfile']
        else:
            openscad_out_file = os.path.join(self.tmp_dir, 'tmp.stl')                    
        try:
            kw['rough'] = True            
            self.render_to_file(openscad_str, openscad_out_file, **kw)
            if openscad_out_file.find('.stl') >= 0:                
                #self._render_stl(openscad_out_file)
                return_obj = RenderedObject(open(openscad_out_file))
            else:
                print('No rendering if non-STL file is being created.')
        except Exception as e:
            raise e
        finally:
            if do_tmp_file:
                if os.path.isfile(openscad_out_file):
                    os.remove(openscad_out_file)
            if self.openscad_tmp_dir is None:
                os.rmdir(self.tmp_dir)
                
        return return_obj
    
    def _render_obj(self, rendered_obj, **kw):        
        obj_geometry = pjs.BufferGeometry(attributes=dict(position=pjs.BufferAttribute(rendered_obj.plot_verts), 
                                                         color=pjs.BufferAttribute(rendered_obj.base_cols),
                                                         normal=pjs.BufferAttribute(rendered_obj.face_normals.astype('float32'))))
        vertices = rendered_obj.vertices
        
        # Create a mesh. Note that the material need to be told to use the vertex colors.        
        my_object_mesh = pjs.Mesh(
            geometry=obj_geometry,
            material=pjs.MeshLambertMaterial(vertexColors='VertexColors'),
            position=[0, 0, 0],   
        )
        
        line_material = pjs.LineBasicMaterial(color='#ffffff', transparent=True, opacity=0.3, linewidth=1.0)
        my_object_wireframe_mesh = pjs.LineSegments(
            geometry=obj_geometry,
            material=line_material,
            position=[0, 0, 0], 
        )

        n_vert = vertices.shape[0]
        center = vertices.mean(axis=0)
        
        extents = self._get_extents(vertices)
        max_delta = np.max(extents[:, 1] - extents[:, 0])
        camPos = [center[i] + 4 * max_delta for i in range(3)]
        light_pos = [center[i] + (i+3)*max_delta for i in range(3)]
    
        # Set up a scene and render it:
        camera = pjs.PerspectiveCamera(position=camPos, fov=20,
                                   children=[pjs.DirectionalLight(color='#ffffff',
                                   position=light_pos, intensity=0.5)])
        camera.up = (0,0,1)

        v = [0.0, 0.0, 0.0]
        if n_vert > 0: v = vertices[0].tolist()
        select_point_geom = pjs.SphereGeometry(radius=1.0)
        select_point_mesh = pjs.Mesh(select_point_geom,
                 material=pjs.MeshBasicMaterial(color=SELECTED_VERTEX_COLOR),
                 position=v, scale=(0.0, 0.0, 0.0))
        
        #select_edge_mesh = pjs.ArrowHelper(dir=pjs.Vector3(1.0, 0.0, 0.0), origin=pjs.Vector3(0.0, 0.0, 0.0), length=1.0,
        #                                  hex=SELECTED_EDGE_COLOR_INT, headLength=0.1, headWidth=0.05)
        
        arrow_cyl_mesh = pjs.Mesh(geometry=pjs.SphereGeometry(radius=0.01), material=pjs.MeshLambertMaterial())
        arrow_head_mesh = pjs.Mesh(geometry=pjs.SphereGeometry(radius=0.001), material=pjs.MeshLambertMaterial())
        
        scene_things = [my_object_mesh, my_object_wireframe_mesh, select_point_mesh, arrow_cyl_mesh, arrow_head_mesh,
                        camera, pjs.AmbientLight(color='#888888')]
        
        if self.draw_grids:
            grids, space = self._get_grids(vertices)
            scene_things.append(grids)

        scene = pjs.Scene(children=scene_things, background=BACKGROUND_COLOR)
        
        
        click_picker = pjs.Picker(controlling=my_object_mesh, event='dblclick')
        out = Output()
        top_msg = HTML()
        
        
        def on_dblclick(change):    
            if change['name'] == 'point':                
                try:
                    point = np.array(change['new'])
                    face = click_picker.faceIndex
                    face_points = rendered_obj.face_verts[face]                    
                    face_vecs = face_points - np.roll(face_points, 1, axis=0)
                    edge_lens = np.sqrt((face_vecs**2).sum(axis=1))
                    point_vecs = face_points - point[np.newaxis, :]
                    point_dists = (point_vecs**2).sum(axis=1)                    
                    min_point = np.argmin(point_dists)                    
                    v1s = point_vecs.copy()
                    v2s = np.roll(v1s, -1, axis=0)
                    edge_mids = 0.5 * (v2s + v1s)
                    edge_mid_dists = (edge_mids**2).sum(axis=1)                    
                    min_edge_point = np.argmin(edge_mid_dists)
                    edge_start = min_edge_point
                    edge = face*3 + edge_start                    
                    close_vert = rendered_obj.face_verts[face, min_point]
                    edge_start_vert = rendered_obj.face_verts[face, edge_start]
                    edge_end_vert = rendered_obj.face_verts[face, (edge_start+1)%3]
                                     
                    vertex = face*3 + min_point
                    radius = min([edge_lens.max()*0.02, 0.1 * edge_lens.min()])
                    
                    edge_head_length = radius * 4
                    edge_head_width = radius * 2
                    select_point_mesh.scale = (radius, radius, radius)
                    top_msg.value = '<font color="{}">selected face: {}</font>, <font color="{}">edge: {}</font>, <font color="{}"> vertex: {}</font>'.format(SELECTED_FACE_COLOR, face, SELECTED_EDGE_COLOR, edge, SELECTED_VERTEX_COLOR, vertex)   
                    newcols = rendered_obj.base_cols.copy()
                    newcols[face*3:(face+1)*3] = np.array(SELECTED_FACE_RGB, dtype='float32')
                    select_point_mesh.position = close_vert.tolist()
                    obj_geometry.attributes['color'].array = newcols
                    
                    with out:   
                        make_arrow(arrow_cyl_mesh, arrow_head_mesh, edge_start_vert, edge_end_vert, radius/2, radius, radius*3, SELECTED_EDGE_COLOR) 
                    
                except:
                    with out:
                        print(traceback.format_exc())
                
    
        click_picker.observe(on_dblclick, names=['point'])

        renderer_obj = pjs.Renderer(camera=camera, background='#cccc88',
            background_opacity=0, scene=scene,
            controls=[pjs.OrbitControls(controlling=camera), click_picker],
            width=self.width,
            height=self.height)

        
        display_things = [top_msg, renderer_obj, out]
        if self.draw_grids:
            s = """
<svg width="{}" height="30">
<rect width="20" height="20" x="{}" y="0" style="fill:none;stroke-width:1;stroke:rgb(0,255,0)" />
    <text x="{}" y="15">={:.1f}</text>
  Sorry, your browser does not support inline SVG.
</svg>""".format(self.width, self.width//2, self.width//2+25, space)
            display_things.append(HTML(s))

        display(VBox(display_things))
        

    

