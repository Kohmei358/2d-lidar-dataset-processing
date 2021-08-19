# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import copy
import time

import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
import os

from sklearn.cluster import OPTICS, AgglomerativeClustering

from open3d.cpu.pybind.visualization.gui import Label3D
from open3d.visualization import gui

from mesh_sequence_player.geometries.Geometry import Geometry
from motpy import Detection, MultiObjectTracker

from mesh_sequence_player.MeshSequencePlayer import MeshSequencePlayer

SubFolderString = "PCD_AK_Both"


# create a multi object tracker with a specified step time of 100ms

model_spec = {
        'order_pos': 2, 'dim_pos': 2, # position is a center in 2D space; under constant velocity model
        'order_size': 0, 'dim_size': 2, # bounding box is 2 dimensional; under constant velocity model
        'q_var_pos': 1000., # process noise
        'r_var_pos': 0.1 # measurement noise
    }

tracker = MultiObjectTracker(dt=1, model_spec=model_spec)

# Define hasmap to associate ID with label value
# ID to label#
ID_to_Label_dict = {}
final_label_index = 0

files = []
file_index = 0

batchProcess = False
batchProcessLimit = 100000

cameraResetTriggered = False



def write_ndarray_to_file(array, filename):
    with open(filename, 'w') as f:
        for item in array:
            f.write("%s\n" % item)


def lineset_from_bounds(xmin, ymin, xmax, ymax, height):
    points = [
        [xmin, ymin, height],
        [xmax, ymin, height],
        [xmax, ymax, height],
        [xmin, ymax, height],
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


def callback(vis):
    vis.remove_geometry("PCD")
    vis.remove_geometry("LS1")
    vis.remove_geometry("LS2")

def processAllFiles(vis):
    global batchProcess
    batchProcess = True
    processNextFile(vis)

def process2(vis):
    global batchProcess, batchProcessLimit
    batchProcess = True
    batchProcessLimit = file_index + 2
    processNextFile(vis)

def process10(vis):
    global batchProcess, batchProcessLimit
    batchProcess = True
    batchProcessLimit = file_index + 10
    processNextFile(vis)

def process50(vis):
    global batchProcess, batchProcessLimit
    batchProcess = True
    batchProcessLimit = file_index + 50
    processNextFile(vis)

def process600(vis):
    global batchProcess, batchProcessLimit
    batchProcess = True
    batchProcessLimit = file_index + 600
    processNextFile(vis)


def processNextFile(vis):
    # Compute detections and bounding boxes
    global file_index
    filename = files[file_index]
    file_index = file_index + 1

    vis.title = str(file_index)

    print(filename)

    vis.clear_3d_labels()

    raw_clusters = o3d.geometry.LineSet()
    active_tracks = o3d.geometry.LineSet()
    matched_clusters = o3d.geometry.LineSet()
    detections_lineset = o3d.geometry.LineSet()

    # Read pcd file as a point cloud, only loading location + color
    pcd = o3d.io.read_point_cloud("data/" + SubFolderString + "/" + filename)



    # Label clusters
    # labels = np.array(pcd.cluster_dbscan(eps=0.040, min_points=5, print_progress=False))
    clustering = AgglomerativeClustering(n_clusters=140).fit(np.asarray(pcd.points))

    # clustering = OPTICS(min_samples=6,xi=0.2).fit(np.asarray(pcd.points))

    labels = clustering.labels_


    max_label = labels.max() + 1
    # print("I see " + str(max_label) + " labels from DBSCAN")

    # Take labels array and sort each label into its own array
    max_final_label = max(labels)
    # print("Generating bounding boxed for up to " + str(max_final_label + 1) + " raw clusters")
    for clusterIndex in range(max_final_label + 1):
        clusterIndices = np.where(labels == clusterIndex)
        pointsInCluster = np.asarray(pcd.points)[clusterIndices]

        if len(pointsInCluster) == 0:
            continue

        # For each sub-array compute centeroid, min, max
        xmin = min(pointsInCluster[:, 0])
        ymin = min(pointsInCluster[:, 1])
        # xave = np.average(pointsInCluster[:, 0])
        # yave = np.average(pointsInCluster[:, 1])
        xmax = max(pointsInCluster[:, 0])
        ymax = max(pointsInCluster[:, 1])

        raw_clusters += lineset_from_bounds(xmin, ymin, xmax, ymax, -0.01)
        # vis.add_3d_label([xmax, ymax, 0], "RAW:{}".format(clusterIndex))

    detections = []

    # Take labels array and sort each label into its own array
    for clusterIndex in range(max_label):
        clusterIndices = np.where(labels == clusterIndex)
        pointsInCluster = np.asarray(pcd.points)[clusterIndices]
        number_of_points = len(pointsInCluster)

        delete_this_cluster = False

        if (number_of_points > 100):
            delete_this_cluster = True

        if (number_of_points < 5):
            delete_this_cluster = True

        if number_of_points is 0:
            print("HELP! this cluster to be processed has no points")

        # For each sub-array compute centeroid, min, max
        xmin = min(pointsInCluster[:, 0])
        ymin = min(pointsInCluster[:, 1])
        # xave = np.average(pointsInCluster[:, 0])
        # yave = np.average(pointsInCluster[:, 1])
        xmax = max(pointsInCluster[:, 0])
        ymax = max(pointsInCluster[:, 1])

        max_cluster_size = 0.6 #meters in either direciton
        min_cluster_size = 0.0  # meters in either direciton
        if max_cluster_size > abs(xmin - xmax) > min_cluster_size and \
                max_cluster_size > abs(ymin - ymax) > min_cluster_size:
            detections.append([xmin, ymin, xmax, ymax, clusterIndex])
            # detections_lineset += lineset_from_bounds(xmin, ymin, xmax, ymax, -0.01)
            # vis.add_3d_label([xmax, ymax, 0], "DET:{}".format(clusterIndex))
        else:
            delete_this_cluster = True


        if delete_this_cluster:
            # print("cluster " + str(clusterIndex) + " is too large with " + str(number_of_points) + " points")
            indices = np.where(labels == clusterIndex)
            np.put(labels, indices, -1)

    # Detections (Label#, box)
    # print(detections)
    # print("Now adding " + str(len(detections)) + " detections as observations to the tracker")

    # Add Detections to object tracker
    # update the state of the multi-object-tracker tracker
    # with the list of bounding boxes
    box_size = 0.5
    tracker.step(detections=[Detection(box=[(bounding_box[0]+bounding_box[2]-box_size)/2,(bounding_box[1]+bounding_box[3]-box_size)/2,
                                            (bounding_box[0]+bounding_box[2]+box_size)/2,(bounding_box[1]+bounding_box[3]+box_size)/2]) for bounding_box in detections])

    # Trakcs (id, box)
    # retrieve the active tracks from the tracker (you can customize
    # the hyperparameters of tracks filtering by passing extra arguments)
    tracks = tracker.active_tracks()

    # print('MOT tracker tracks %d active tracks' % len(tracks))
    # print('first track box: %s' % str(tracks[0].box))

    # For each cluster with ID#, find closest active track
    # Make a copy of labeled points
    final_labels = np.full(len(labels), -1)

    for row in detections:
        bounding_box = row[0:4]
        cluster_index = row[4]
        closest_index = None
        closest_index_dist = 999999
        box_center = np.array(
            ((bounding_box[0] + bounding_box[2]) / 2, (bounding_box[1] + bounding_box[3]) / 2))
        for index, track in enumerate(tracks):
            track_center = np.array(((track.box[0] + track.box[2]) / 2, (track.box[1] + track.box[3]) / 2))
            dist = np.linalg.norm(box_center - track_center)
            if dist < closest_index_dist:
                closest_index = index
                closest_index_dist = dist

        # Found closest match
        # the best track for this bounding box is:
        track = tracks[closest_index]

        label_for_this_track = ID_to_Label_dict.get(track.id)

        if label_for_this_track is None:
            global final_label_index
            label_for_this_track = final_label_index
            ID_to_Label_dict.update({track.id: label_for_this_track})
            # print("newly matched "+str(track.id)+" to "+str(label_for_this_track))
            final_label_index = final_label_index + 1
        # else:
            # print("matched "+str(track.id)+" to "+str(label_for_this_track))

        indices = np.where(labels == cluster_index)
        np.put(final_labels, indices, [label_for_this_track])

    # Remove the '.pcd' extention and write the labels as a .txt file
    stripped_filename = filename[:-4]
    write_ndarray_to_file(final_labels, "data/" + SubFolderString + "/" + stripped_filename + ".txt")
    # print("saved with " + str(len(np.unique(final_labels)) - 1) + " clusters:")
    # print(np.unique(final_labels))

    # Color PCD

    # Convert Labels to colors
    # max_label = 9
    # colors = plt.get_cmap("tab10")(final_labels % 10 / max_label)
    # colors[labels < 0] = 0  # Background should be black
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # Write colored point cloud to file
    # o3d.io.write_point_cloud("data/" + SubFolderString + "/" + filename, pcd)
    #

    # Also show MOT tracks
    for track in tracks:
        # For each sub-array compute centeroid, min, max
        bb = track.box

        # active_tracks += lineset_from_bounds(bb[0], bb[1], bb[2], bb[3], 0.01)
        # vis.add_3d_label([bb[0], bb[1], 0], "{}".format(track.id[:5]))

    # Take labels array and sort each label into its own array
    max_final_label = max(final_labels)
    # print("Generating bounding boxed for up to " + str(max_final_label + 1) + " clusters")
    for clusterIndex in range(max_final_label + 1):
        clusterIndices = np.where(final_labels == clusterIndex)
        pointsInCluster = np.asarray(pcd.points)[clusterIndices]

        if len(pointsInCluster) == 0:
            continue

        # For each sub-array compute centeroid, min, max
        xmin = min(pointsInCluster[:, 0])
        ymin = min(pointsInCluster[:, 1])
        xmax = max(pointsInCluster[:, 0])
        ymax = max(pointsInCluster[:, 1])

        matched_clusters += lineset_from_bounds(xmin, ymin, xmax, ymax, 0.02)
        vis.add_3d_label([(xmin+xmax)/2, (ymin+ymax)/2, 0], "{}".format(clusterIndex))


    external_labels = np.loadtxt('data/' + SubFolderString + '/FINAL_' + stripped_filename + '.txt',dtype=int)
    max_external_labels = max(external_labels)
    for clusterIndex in range(max_external_labels + 1):
        clusterIndices = np.where(external_labels == clusterIndex)
        pointsInCluster = np.asarray(pcd.points)[clusterIndices]

        if(len(pointsInCluster) == 0):
            continue

        # For each sub-array compute centeroid, min, max
        xmin = min(pointsInCluster[:, 0])
        ymin = min(pointsInCluster[:, 1])
        xmax = max(pointsInCluster[:, 0])
        ymax = max(pointsInCluster[:, 1])

        detections_lineset += lineset_from_bounds(xmin, ymin, xmax, ymax, 0.02)

    max_label = 9
    colors = plt.get_cmap("tab10")(external_labels % 10 / max_label)
    colors[external_labels < 0] = 0  # Background should be black
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.io.write_point_cloud("data/" + SubFolderString + "/" + filename, pcd)



    colors = [[1, 0, 0] for i in range(len(active_tracks.lines))]
    active_tracks.colors = o3d.utility.Vector3dVector(colors)

    colors = [[0.4, 0.4, 0.4] for i in range(len(raw_clusters.lines))]
    raw_clusters.colors = o3d.utility.Vector3dVector(colors)

    colors = [[0, 1, 0] for i in range(len(matched_clusters.lines))]
    matched_clusters.colors = o3d.utility.Vector3dVector(colors)

    colors = [[0, 0, 1] for i in range(len(detections_lineset.lines))]
    detections_lineset.colors = o3d.utility.Vector3dVector(colors)

    vis.remove_geometry("Point Cloud")
    vis.remove_geometry("Raw Clusters (grey)")
    vis.remove_geometry("Active Tracks (red)")
    vis.remove_geometry("Matched Clusters (green)")
    vis.remove_geometry("Detected Clusters (blue)")

    vis.add_geometry("Point Cloud", pcd)
    vis.add_geometry("Raw Clusters (grey)", raw_clusters)
    # vis.add_geometry("Active Tracks (red)", active_tracks)
    # vis.add_geometry("Matched Clusters (green)", matched_clusters)
    vis.add_geometry("Detected Clusters (blue)", detections_lineset)

    global cameraResetTriggered
    if not cameraResetTriggered:
        vis.reset_camera_to_default()
        cameraResetTriggered = True

    global batchProcess
    if batchProcess and batchProcessLimit > file_index:
        processNextFile(vis)
    # o3d.visualization.draw_geometries([line_set, pcd])

    # o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':


    # vis.add_geometry("Points", points)
    # for idx in range(0, len(points.points)):
    #     vis.add_3d_label(points.points[idx], "{}".format(idx))
    # vis.reset_camera_to_default()

    # player = MeshSequencePlayer(fps=10, loop=False)
    # player.load_pointclouds("data/" + SubFolderString + "/", "*.pcd")
    # player.open(window_name="Mesh Sequence Player - %s" % SubFolderString,
    #             width=1920, height=1080, visible=True)
    #
    # player.play()
    # player.close()
    # time.sleep(100)

    for filename in sorted(os.listdir('data/' + SubFolderString)):
        if filename.endswith(".pcd"):
            files.append(filename)

    app = gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1920, 1080)

    vis.show_settings = True

    vis.add_action("Next Frame", processNextFile)
    vis.add_action("Skip 2", process2)
    vis.add_action("Skip 10", process10)
    vis.add_action("Skip 50", process50)
    vis.add_action("Skip 600", process600)
    vis.add_action("Process all", processAllFiles)

    app.add_window(vis)
    vis.add_3d_label([0, 0, 0], "Press \"Next Frame\" to show first frame!")
    app.run()


