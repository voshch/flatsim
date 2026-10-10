string LASERSCAN=laserscan
string POINTCLOUD=pointcloud
string IMU=imu
string CONTACT=contact

string name
string type       # one of the constants above
string topic      # absolute topic the sensor publishes on
string frame      # URDF link the sensor sits on
string sensor     # URDF <sensor> that backs it, empty = name
Fan[<=1] fan      # laserscan beams when the URDF has no such <sensor>
