# UnitreeProject2026
'Unitree Go2 Robot Dog (Quadruped Robotic)'
> This project is meant for educational purposes.
> Robot doggo find path using images and walks

## Getting the code
### Option 1: GitHub Desktop
When you click on the project <ins>UnitreeProject2026</ins> you will see a button in the top-right of the screen called '<> Code'. After clicking on the button you will see the options; HTTPS, SSH, GitHub CLI. Click on 'SSH'.

At the bottom of the popup you get an option to 'Open with GitHub Desktop', choose that option. If you don't have GitHub Desktop installed yet, here is the [link](https://docs.github.com/en/desktop/installing-and-authenticating-to-github-desktop/installing-github-desktop) for the instructions!

You will then get the option to **Clone a repository** (URL option selected), then choose where you want that code to be cloned to by clicking on 'Choose ...' (or leave it at the default location). Then click on 'Clone'. 

### Option 2: Terminal
Open a terminal of your choice, navigate to the folder where you want the code to be placed, and run the following command:

```
gh repo clone LiamStammeleerEhb/UnitreeProject2026
```
If you don't have the GitHub CLI installed yet, here is the [link](https://cli.github.com/manual/installation) to the installation instructions.

## After receiving the code
After having cloned the project you can open **Visual Studio Code**, if you don't have it installed yet, here is the [link](https://code.visualstudio.com/docs/setup/setup-overview) to the instructions on how to do so. After installing (or if you already have the app installed) you can open the app and in the top-left you see a button called 'File', then click on 'Open Folder ...' and select the folder where the project (the code for the dog) resides. Then press 'Open'.

### If only `Publisher.py` and `Subscriber.py` are present
If you only have the Python files and no ROS 2 package files yet, you must first place them inside a ROS 2 Python package.

A valid ROS 2 package should also contain:
- `package.xml`
- `setup.py`
- `resource/<package_name>`
- `__init__.py`

If these files are already in the repository, you do **not** need to create a new package again.  
You only need to build the workspace:

```bash
cd /home/jetson/ros
colcon build --symlink-install
source install/setup.bash
```

If the package files are missing, create a new package first and then add `Publisher.py` and `Subscriber.py` to it.

#### Create a new ROS 2 package (if needed)
If the repo does not already contain a ROS 2 package, create one and add the Python files:

1. Prepare workspace and create package:
```bash
mkdir -p ~/ros/src
cd ~/ros/src
ros2 pkg create --build-type ament_python walk_in_the_park --dependencies rclpy std_msgs
```

2. Copy files into the package and add __init__.py:
```bash
cp /path/to/downloaded/Publisher.py /home/jetson/ros/src/walk_in_the_park/walk_in_the_park/Publisher.py  # replace with the actual location of the file
cp /path/to/downloaded/Subscriber.py /home/jetson/ros/src/walk_in_the_park/walk_in_the_park/Subscriber.py  # replace with the actual location of the file
touch /home/jetson/ros/src/walk_in_the_park/walk_in_the_park/__init__.py
```

3. Verify or update setup.py entry points (example):
```python
entry_points={
  'console_scripts': [
    'publisher = walk_in_the_park.Publisher:main',
    'subscriber = walk_in_the_park.Subscriber:main',
  ],
},
```

4. Build and source the workspace:
```bash
cd /home/jetson/ros
colcon build --symlink-install
source install/setup.bash
```

5. Run nodes:
```bash
ros2 run walk_in_the_park subscriber
ros2 run walk_in_the_park publisher
```

### YOLO model location
Place your trained model file in:

`/home/jetson/Models`

Then open `Publisher.py` and change `MODEL_PATH` on line 24 to the full path of your model file, for example:

```python
MODEL_PATH = "/home/jetson/Models/KaaiGang.pt"
```

If your model has a different name or is stored in another folder, update `MODEL_PATH` accordingly.

## Starting up the Unitree
Before starting the Unitree you need to plug in the Jetson Nano using the power cable coming out of the Unitree. Attach the Jetson Nano safely to the back of the Unitree.

If you are using a hotspot or wifi network, make sure the network is running before starting the Unitree.

To start the Unitree, simply hold the power button located on the battery, then wait a minute until it stands up.

## Starting the code (path and ArUco detection)

To start the robot walking without the use of the ArUco codes you need to run 2 files in separate terminals while connected to the Jetson over SSH.

We recommend to first start the subscriber and then the publisher.
So go to your terminal and SSH into the Jetson:
```
ssh jetson@(IP of the jetson)
```
You can find the IP address by checking your router's connected devices or running `ip a` directly on the Jetson.

Then start the Subscriber:
```
ros2 run walk_in_the_park subscriber
```
After it starts running, start the Publisher:
```
ros2 run walk_in_the_park publisher
```
