import os.path
import time

import numpy as np
import open3d as o3d
from cv2 import VideoWriter_fourcc, VideoWriter, cvtColor, COLOR_BGR2RGB
from tqdm import tqdm

from mesh_sequence_player.FPSCounter import FPSCounter
from mesh_sequence_player.FastGeometryLoader import load_meshes_fast, load_meshes_safe, load_pointclouds_safe, \
    load_pointclouds_fast
from mesh_sequence_player.geometries.BaseGeometry import BaseGeometry
from mesh_sequence_player.geometries.Geometry import Geometry
from mesh_sequence_player.geometries.LazyGeometry import LazyGeometry
from mesh_sequence_player.utils import get_files_in_path


class MeshSequencePlayer:
    def __init__(self, fps: int = 24, loop: bool = True):
        self.fps = fps
        self.loop = loop
        self.geometries: [BaseGeometry] = []
        self.pcds = []
        self.linesets: [BaseGeometry] = []
        self.linesetsPermanent: [BaseGeometry] = []
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.background_color = [255, 255, 255]

        self.debug = False
        self.load_safe = False
        self.lazy_loading = False

        self.render = False
        self.output_path = "render.mp4"
        self.render_index = 0

        self.vis = o3d.visualization.Visualizer()

        self._is_playing: bool = False
        self._index: int = 0
        self._last_update_ts = 0
        self._current_geometry = None
        self._current_lineset = None
        self._current_lineset_permanent = None

        self._writer: VideoWriter = None
        self._progress_bar: tqdm = None

        self._fps_counter = FPSCounter()

    def load_meshes(self, mesh_folder: str, mesh_format: str = "*.obj"):
        files = sorted(get_files_in_path(mesh_folder, extensions=[mesh_format]))

        if self.lazy_loading:
            self.geometries = [LazyGeometry(os.path.abspath(file), o3d.io.read_triangle_mesh) for file in files]
            return

        if self.load_safe:
            meshes = load_meshes_safe(files)
        else:
            meshes = load_meshes_fast(files)

        self.geometries = [Geometry(mesh) for mesh in meshes]

    def load_pointclouds(self, pcl_folder: str, pcl_format: str = "*.ply"):
        files = sorted(get_files_in_path(pcl_folder, extensions=[pcl_format]))

        if self.lazy_loading:
            self.geometries = [LazyGeometry(os.path.abspath(file), o3d.io.read_point_cloud) for file in files]
            return

        if self.load_safe:
            pcds = load_pointclouds_safe(files)
        else:
            pcds = load_pointclouds_fast(files)
            self.pcds = pcds

        self.geometries = [Geometry(pcd) for pcd in pcds]
        self.linesets, self.linesetsPermanent = self.load_linesets(pcl_folder)

    def load_linesets(self, labels_folder: str, labels_format: str = "*.txt"):
        files = sorted(get_files_in_path(labels_folder, extensions=[labels_format]))
        linesets = []
        linesetsPermanent = []
        for index, file in enumerate(files):
            labels = np.loadtxt(file, dtype='int')
            max_label = max(labels)
            line_set = o3d.geometry.LineSet()
            line_set_permanent = o3d.geometry.LineSet()
            # Take labels array and sort each label into its own array
            for clusterIndex in range(max_label):
                clusterIndices = np.where(labels == clusterIndex)
                pointsInCluster = np.asarray(self.pcds[index].points)[clusterIndices]

                if len(pointsInCluster) == 0:
                    continue

                # For each sub-array compute centeroid, min, max
                xmin = min(pointsInCluster[:, 0])
                ymin = min(pointsInCluster[:, 1])
                xave = np.average(pointsInCluster[:, 0])
                yave = np.average(pointsInCluster[:, 1])
                xmax = max(pointsInCluster[:, 0])
                ymax = max(pointsInCluster[:, 1])

                line_set += self.lineset_from_bounds(xmin, ymin, xmax, ymax)
                line_set_permanent += self.lineset_from_point(xave, yave)
            linesets.append(line_set)
            linesetsPermanent.append(line_set)
        return linesets, linesetsPermanent

    def lineset_from_bounds(self, xmin, ymin, xmax, ymax):
        points = [
            [xmin, ymin, 0],
            [xmax, ymin, 0],
            [xmax, ymax, 0],
            [xmin, ymax, 0],
        ]
        lines = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
        ]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        return line_set

    def lineset_from_point(self, xave, yave):
        points = [
            [xave, yave, 0],
        ]
        lines = [
            [0, 1],
        ]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        return line_set

    def open(self, window_name: str = 'Mesh Sequence Player',
             width: int = 1080, height: int = 1080,
             visible: bool = True):
        self.vis.create_window(window_name=window_name,
                               width=width,
                               height=height,
                               visible=visible)

        if len(self.geometries) == 0:
            print("No meshes to show!")
            return

        if len(self.linesets) == 0:
            print("No linesets to show!")

        if self.render:
            fourcc = VideoWriter_fourcc(*'mp4v')
            self._writer = VideoWriter(self.output_path, fourcc, self.fps, (width, height))
            self._progress_bar = tqdm(total=len(self.geometries), desc="rendering")

            # make rendering as fast as possible
            self.fps = 10000.0

        # set background color
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray(self.background_color)

        # add first mesh
        self._current_geometry = self.geometries[self._index].get()
        self._current_lineset = self.linesets[self._index]
        self._current_lineset_permanent = self.linesetsPermanent[self._index]
        self.vis.add_geometry(self._current_geometry, reset_bounding_box=True)
        self.vis.add_geometry(self._current_lineset, reset_bounding_box=False)
        self.vis.add_geometry(self._current_lineset_permanent, reset_bounding_box=False)

    def close(self):
        self._is_playing = False
        self.vis.destroy_window()

    def play(self):
        self._is_playing = True
        self._play_loop()

    def pause(self):
        self._is_playing = False

    def jump(self, index: int):
        self._index = index

    def _play_loop(self):
        self._fps_counter.reset()

        while self._is_playing:
            # rotation
            ctr = self.vis.get_view_control()
            ctr.rotate(self.rotation_x, self.rotation_y)

            # events
            if not self.vis.poll_events():
                break

            self.vis.update_renderer()

            # skip if no meshes available
            if len(self.geometries) == 0:
                continue

            # render
            if self.render:
                color = self.vis.capture_screen_float_buffer(False)
                color = np.asarray(color)
                color = np.uint8(color * 255.0)
                im_rgb = cvtColor(color, COLOR_BGR2RGB)
                self._writer.write(im_rgb)

                self.render_index += 1
                self._progress_bar.update()

            # frame playing
            current = self._millis()
            if (current - self._last_update_ts) > (1000.0 / self.fps):
                self._next_frame()
                self._last_update_ts = current

            # keep track of fps
            self._fps_counter.update()

            if self.debug:
                tqdm.write("FPS: %0.2f" % self._fps_counter.fps)

    def _next_frame(self):
        if not self.loop and self._index == len(self.geometries) - 1:
            if self.render:
                self._writer.release()
                self._progress_bar.close()

            self._is_playing = False

        self.vis.remove_geometry(self._current_geometry, reset_bounding_box=False)
        self.vis.remove_geometry(self._current_lineset, reset_bounding_box=False)
        self.vis.remove_geometry(self._current_lineset_permanent, reset_bounding_box=False)
        self._index = (self._index + 1) % len(self.geometries)
        self._current_geometry = self.geometries[self._index].get()
        self._current_lineset = self.linesets[self._index]
        self._current_lineset_permanent = self.linesetsPermanent[self._index]
        self.vis.add_geometry(self._current_geometry, reset_bounding_box=False)
        self.vis.add_geometry(self._current_lineset, reset_bounding_box=False)
        self.vis.add_geometry(self._current_lineset_permanent, reset_bounding_box=False)

    @staticmethod
    def _millis() -> int:
        return round(time.time() * 1000)
