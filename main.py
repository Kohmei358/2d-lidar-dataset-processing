# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import copy
import time

import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
import os
from motpy import Detection, MultiObjectTracker

from mesh_sequence_player.MeshSequencePlayer import MeshSequencePlayer

SubFolderString = "PCD_AK_Both"


def write_ndarray_to_file(array, filename):
    with open(filename, 'w') as f:
        for item in array:
            f.write("%s\n" % item)


def lineset_from_bounds(xmin, ymin, xmax, ymax):
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


if __name__ == '__main__':
    player = MeshSequencePlayer(fps=10, loop=False)
    player.load_pointclouds("data/" + SubFolderString + "/", "*.pcd")
    player.open(window_name="Mesh Sequence Player - %s" % SubFolderString,
                width=1920, height=1080, visible=True)

    player.play()
    player.close()
    time.sleep(100)

    # create a multi object tracker with a specified step time of 100ms
    tracker = MultiObjectTracker(dt=0.1)

    # Define hasmap to associate ID with label value
    # ID to label#
    ID_to_Label_dict = {}
    final_label_index = 0

    for filename in sorted(os.listdir('data/' + SubFolderString)):
        if filename.endswith(".pcd"):
            print(filename)

            # Compute detections and bounding boxes

            # Read pcd file as a point cloud, only loading location + color
            pcd = o3d.io.read_point_cloud("data/" + SubFolderString + "/" + filename)

            # Label clusters
            labels = np.array(pcd.cluster_dbscan(eps=0.08, min_points=5, print_progress=False))

            max_label = labels.max()+1
            print("I see "+str(max_label)+" labels from DBSCAN")

            detections = []

            # Take labels array and sort each label into its own array
            for clusterIndex in range(max_label):
                clusterIndices = np.where(labels == clusterIndex)
                pointsInCluster = np.asarray(pcd.points)[clusterIndices]
                number_of_points = len(pointsInCluster)

                if(number_of_points < 999):

                    if number_of_points is 0:
                        print("HELP")

                    # For each sub-array compute centeroid, min, max
                    xmin = min(pointsInCluster[:, 0])
                    ymin = min(pointsInCluster[:, 1])
                    # xave = np.average(pointsInCluster[:, 0])
                    # yave = np.average(pointsInCluster[:, 1])
                    xmax = max(pointsInCluster[:, 0])
                    ymax = max(pointsInCluster[:, 1])

                    detections.append([xmin, ymin, xmax, ymax])

                else:
                    print("cluster "+str(clusterIndex) +" is too large with "+ str(number_of_points)+" points")
                    indices = np.where(labels == clusterIndex)
                    np.put(labels, indices, -1)

            # Detections (Label#, box)
            # print(detections)
            print("Now adding "+str(len(detections))+" detections as observations to the tracker")

            # Add Detections to object tracker
            # update the state of the multi-object-tracker tracker
            # with the list of bounding boxes
            tracker.step(detections=[Detection(box=bounding_box) for bounding_box in detections])

            # Trakcs (id, box)
            # retrieve the active tracks from the tracker (you can customize
            # the hyperparameters of tracks filtering by passing extra arguments)
            tracks = tracker.active_tracks()

            print('MOT tracker tracks %d active tracks' % len(tracks))
            # print('first track box: %s' % str(tracks[0].box))

            # For each cluster with ID#, find closest active track
            # Make a copy of labeled points
            final_labels = copy.deepcopy(labels)

            for cluster_index, bounding_box in enumerate(detections):
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
                    label_for_this_track = final_label_index
                    ID_to_Label_dict.update({track.id: label_for_this_track})
                    print("newly matched "+str(track.id)+" to "+str(label_for_this_track))
                    final_label_index = final_label_index + 1
                else:
                    print("matched "+str(track.id)+" to "+str(label_for_this_track))

                indices = np.where(labels == cluster_index)
                np.put(final_labels, indices, [label_for_this_track])

            # Remove the '.pcd' extention and write the labels as a .txt file
            stripped_filename = filename[:-4]
            write_ndarray_to_file(final_labels, "data/" + SubFolderString + "/" + stripped_filename + ".txt")
            print("saved with "+str(len(np.unique(final_labels))-1)+" clusters:")
            print(np.unique(final_labels))

            # Color PCD

            # Convert Labels to colors
            max_label = 9
            colors = plt.get_cmap("tab10")(final_labels % 10 / max_label)
            colors[labels < 0] = 0 # Background should be black
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

            # Write colored point cloud to file
            o3d.io.write_point_cloud("data/" + SubFolderString + "/" + filename, pcd)
            #
            line_set = o3d.geometry.LineSet()

            # Take labels array and sort each label into its own array
            max_final_label = max(final_labels)
            print("Generating bounding boxed for up to "+str(max_final_label+1)+" clusters")
            for clusterIndex in range(max_final_label+1):
                clusterIndices = np.where(final_labels == clusterIndex)
                pointsInCluster = np.asarray(pcd.points)[clusterIndices]

                if len(pointsInCluster) == 0:
                    continue

                # For each sub-array compute centeroid, min, max
                xmin = min(pointsInCluster[:, 0])
                ymin = min(pointsInCluster[:, 1])
                xave = np.average(pointsInCluster[:, 0])
                yave = np.average(pointsInCluster[:, 1])
                xmax = max(pointsInCluster[:, 0])
                ymax = max(pointsInCluster[:, 1])

                line_set += lineset_from_bounds(xmin, ymin, xmax, ymax)

            # o3d.visualization.draw_geometries([line_set, pcd])
            #
            # # o3d.visualization.draw_geometries([pcd])
