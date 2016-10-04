#include "stdafx.h"
#include "kinect2_grabber2.h"

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

int objectsCount = 0; // number of objects
pcl::PointCloud<PointType>::Ptr cloud_filtered(new pcl::PointCloud<PointType>);

/* Extract clusters and print objects count */
void extractClusters(pcl::PointCloud<PointType>::Ptr cloud) {

	/* THIS IS WHERE THE FUN STARTS */
	std::cout << "PointCloud before clipping has: " << cloud->points.size() << " data points." << std::endl; //*

	//clipping

	pcl::PointCloud<PointType>::Ptr cloud_pass(new pcl::PointCloud<PointType>());
	pcl::PassThrough<PointType> pass1;
	pass1.setInputCloud(cloud);
	pass1.setFilterFieldName("z");
	pass1.setFilterLimits(0, 1); // reduces depth
	pass1.filter(*cloud_pass);
	std::cout << "PointCloud after clipping Z has: " << cloud_pass->points.size() << " data points." << std::endl;

	// Create the filtering object: downsample the dataset using a leaf size of 1cm
	pcl::VoxelGrid<PointType> vg;
	
	vg.setInputCloud(cloud_pass);
	vg.setLeafSize(0.04f, 0.04f, 0.04f);
	std::cout << "T1" << std::endl;
	vg.filter(*cloud_filtered);
	std::cout << "T2" << std::endl;
	std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size() << " data points." << std::endl;

	// Create the segmentation object for the planar model and set all the parameters
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointCloud<PointType>::Ptr cloud_plane(new pcl::PointCloud<PointType>());
	pcl::PCDWriter writer;
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_TORUS);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(10);
	seg.setDistanceThreshold(0.01);

	// Creating the KdTree object for the search method of the extraction
	pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
	tree->setInputCloud(cloud_filtered);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<PointType> ec;
	ec.setClusterTolerance(0.05);
	ec.setMinClusterSize(100);
	ec.setMaxClusterSize(25000);
	ec.setSearchMethod(tree);
	ec.setInputCloud(cloud_filtered);
	ec.extract(cluster_indices);

	int j = 0;
	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{
		j++;
	}

	objectsCount = j;
}

int main(int argc, char* argv[])
{

	// PCL Visualizer
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
		new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
	viewer->setCameraPosition(0.0, 0.0, -2.5, 0.0, 0.0, 0.0);

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
			if (!viewer->updateText("Objects on screen: " + std::to_string(objectsCount), 20, 20, "textId")) {
				viewer->addText("Objects on screen: " + std::to_string(objectsCount), 20, 20, "textId");
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
