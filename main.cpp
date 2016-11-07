#include "stdafx.h"
#include "kinect2_grabber2.h"

#include <cmath>

#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>    
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/people/ground_based_people_detection_app.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/compression/octree_pointcloud_compression.h>

#include <pcl/filters/passthrough.h>

typedef pcl::PointXYZ PointType;

typedef struct Object {
	int id;
	pcl::PointXY coords;
};

// Tracking: maximum distance from object's position in the previous frame
const double RECOGNITION_DISTANCE_THRESHOLD = 0.15;
// Tracking: maximum number of objects on the screen at once
const int MAX_OBJECTS = 10;
// Tracking: allocating memory for objects
Object objects[MAX_OBJECTS];
// Counting: used to display the number of people
int objectsCount = 0; // number of objects
					  // Clustering: final point cloud pointer is assigned to this variable
pcl::PointCloud<PointType>::Ptr cloud_filtered(new pcl::PointCloud<PointType>);

/* Tracking: get a center point of a point cloud */
pcl::PointXY getCentroid(pcl::PointCloud<PointType>::Ptr cloud) {
	float x = 0, y = 0;
	int size = cloud->points.size();

	for (size_t i = 0; i < size; ++i) {
		x += cloud->points[i].x;
		y += cloud->points[i].y;
	}

	pcl::PointXY point;
	point.x = x / size;
	point.y = y / size;

	return point;
}

/* Tracking: retrieve an id of the object from a previous frame or generate a random id */
int getObjectId(pcl::PointXY centroid) {
	for (int i = 0; i < MAX_OBJECTS; i++) {
		if (objects[i].id > -1) {
			if (abs(objects[i].coords.x - centroid.x) < RECOGNITION_DISTANCE_THRESHOLD && abs(objects[i].coords.y - centroid.y) < RECOGNITION_DISTANCE_THRESHOLD) {
				return objects[i].id;
			}
		}
	}

	return (int)rand() & 10000;
}

/* Create an instance of Object from point cloud */
Object createObjectFromCloud(pcl::PointCloud<PointType>::Ptr cloud) {
	pcl::PointXY centroid = getCentroid(cloud);

	Object obj;
	obj.coords = centroid;
	obj.id = getObjectId(centroid);

	return obj;
}

/* Extract clusters and print objects count */
void extractClusters(pcl::PointCloud<PointType>::Ptr cloud) {

	/* THIS IS WHERE THE FUN STARTS */
	// std::cout << "PointCloud before clipping has: " << cloud->points.size() << " data points." << std::endl; //*

	//clipping
	pcl::PointCloud<PointType>::Ptr cloud_pass(new pcl::PointCloud<PointType>());
	pcl::PassThrough<PointType> pass1;
	pass1.setInputCloud(cloud);
	pass1.setFilterFieldName("z");
	pass1.setFilterLimits(0, 1.2); // reduces depth
	pass1.filter(*cloud_pass);
	// std::cout << "PointCloud after clipping Z has: " << cloud_pass->points.size() << " data points." << std::endl;

	// Create the filtering object: downsample the dataset using a leaf size of 1cm5*
	pcl::VoxelGrid<PointType> vg;

	vg.setInputCloud(cloud_pass);
	vg.setLeafSize(0.04f, 0.04f, 0.04f);
	vg.filter(*cloud_filtered);
	// std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size() << " data points." << std::endl;

	// Creating the KdTree object for the search method of the extraction
	pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
	tree->setInputCloud(cloud_filtered);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<PointType> ec;
	ec.setClusterTolerance(0.05);
	ec.setMinClusterSize(50);
	ec.setMaxClusterSize(25000);
	ec.setSearchMethod(tree);
	ec.setInputCloud(cloud_filtered);
	ec.extract(cluster_indices);

	objectsCount = 0;
	Object newObjects[MAX_OBJECTS];
	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
			cloud_cluster->points.push_back(cloud_filtered->points[*pit]); //*
		cloud_cluster->width = cloud_cluster->points.size();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		Object obj = createObjectFromCloud(cloud_cluster);

		newObjects[objectsCount] = obj;
		objectsCount++;
	}

	for (int i = 0; i < MAX_OBJECTS; i++) {
		objects[i].id = -1;

		if (newObjects[i].id > 0) {
			objects[i] = newObjects[i];
		}

		if (objects[i].id != -1) {
			cout << "new Object " << objects[i].id << "    pos: " << objects[i].coords << endl;
		}
	}

	cout << "-----------" << endl;
}

int main(int argc, char* argv[]) {
	// PCL Visualizer
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
		new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
	viewer->setCameraPosition(0.0, 0.0, -2.5, 0.0, 0.0, 0.0);
	viewer->setShowFPS(false);

	// Point Cloud
	pcl::PointCloud<PointType>::Ptr cloud;

	// Retrieved Point Cloud Callback Function
	boost::mutex mutex;
	boost::function<void(const pcl::PointCloud<PointType>::ConstPtr&)> function =
		[&cloud, &mutex](const pcl::PointCloud<PointType>::ConstPtr& ptr) {
		boost::mutex::scoped_lock lock(mutex);

		/* Point Cloud Processing */
		cout << "points received" << endl;

		cloud = ptr->makeShared();

		// Extracting the clusters and changing the cloud object
		extractClusters(cloud);
	};

	// Kinect2Grabber
	boost::shared_ptr<pcl::Grabber> grabber = boost::make_shared<pcl::Kinect2Grabber>();

	// Register Callback Function
	boost::signals2::connection connection = grabber->registerCallback(function);

	// Start Grabber
	grabber->start();

	while (!viewer->wasStopped()) {
		// Update Viewer
		viewer->spinOnce();

		boost::mutex::scoped_try_lock lock(mutex);
		if (lock.owns_lock() && cloud) {

			// Update Point Cloud
			if (!viewer->updatePointCloud(cloud_filtered, "cloud")) {
				viewer->addPointCloud(cloud_filtered, "cloud");
			}

			// Updating label with number of objects (lower left corner)
			if (!viewer->updateText("Objects on screen: " + std::to_string(objectsCount), 20, 20, 20, 1, 1, 1, "textId")) {
				viewer->addText("Objects on screen: " + std::to_string(objectsCount), 20, 20, 20, 1, 1, 1, "textId");
			}
		}
	}

	// Stop Grabber
	grabber->stop();

	// Disconnect Callback Function
	if (connection.connected()) {
		connection.disconnect();
	}

	return 0;
}
