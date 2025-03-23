from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
import dm_control.utils.transformations as tr
import numpy as np
import matplotlib.pyplot as plt

class CarObservables(composer.Observables):

    @property
    def body(self):
        return self._entity._mjcf_root.find('body', 'buddy')

    def get_sensor_mjcf(self, sensor_name):
        return self._entity._mjcf_root.find('sensor', sensor_name)

    @composer.observable
    def realsense_camera(self):
        return observable.MJCFCamera(self._entity._mjcf_root.find('camera', 'buddy_realsense_d435i'), height=64, width=128, buffer_size=1, depth=True)
    
    @composer.observable
    def compute_point_cloud(self):
        def get_point_cloud(physics):
            # Get the depth image from the camera
            depth_image = self.realsense_camera(physics)
            if depth_image is None:
                raise ValueError("Depth image is None. Check camera configuration.")
            
            # Process depth image into point cloud
            height, width, _ = depth_image.shape
            
            # Camera intrinsics
            fx = 77.25
            fy = 77.25
            cx = width / 2
            cy = height / 2

            # Generate the 2D grid of (x, y) pixel coordinates
            xs, ys = np.meshgrid(np.arange(width), np.arange(height))  # Shape: (H, W)

            # Normalize pixel coordinates and map to 3D coordinates
            xs = (xs - cx) / fx
            ys = (ys - cy) / fy
            zs = depth_image[:, :, 0]  # Extract depth values (H, W)
            # print(np.min(zs),np.max(zs))
            # plt.imshow(np.uint8(zs))
            # plt.show()
            

            # Mask out invalid depth points (e.g., zero or negative depth)
            valid_mask = np.logical_and(zs>0, zs<10) # Filter out points that are too far away
            # if not valid_mask.any():
            # valid_mask = zs>0
            
            xs, ys, zs = xs[valid_mask], ys[valid_mask], zs[valid_mask]
            # xs, ys, zs = xs.flatten(), ys.flatten(), zs.flatten()
            # max_z = np.where(zs==np.max(zs))
            # print(xs[max_z],ys[max_z],zs[max_z],'end')

            # Stack valid points into an N x 3 array
            point_cloud = np.stack((xs * zs, ys * zs, zs), axis=-1)  # Shape: (N, 3)

            # print(np.max(point_cloud[:,0]),np.max(point_cloud[:,1]),np.max(point_cloud[:,2]))

            target_size = int(height*width)
            if len(point_cloud) < target_size:
                padding = np.zeros((target_size - len(point_cloud),3))
                point_cloud = np.concatenate((point_cloud,padding),axis=0)

            return point_cloud

        return observable.Generic(get_point_cloud)
    
    @composer.observable
    def body_position(self):
        return observable.MJCFFeature('xpos', self._entity._mjcf_root.find('body', 'buddy'))

    @composer.observable
    def wheel_speeds(self):
        def get_wheel_speeds(physics):
            return np.concatenate([
                physics.bind(self.get_sensor_mjcf(f'buddy_wheel_{wheel}_vel')).sensordata
                for wheel in ['fl', 'fr', 'bl', 'br']
            ])

        return observable.Generic(get_wheel_speeds)

    @composer.observable
    def body_pose_2d(self):
        def get_pose_2d(physics):
            pos = physics.bind(self.body).xpos[:2]
            yaw = tr.quat_to_euler(physics.bind(self.body).xquat)[2]
            return np.append(pos, yaw)

        return observable.Generic(get_pose_2d)

    @composer.observable
    def body_vel_2d(self):
        def get_vel_2d(physics):
            quat = physics.bind(self.body).xquat
            velocity_local = physics.bind(self.get_sensor_mjcf('velocimeter')).sensordata
            return tr.quat_rotate(quat, velocity_local)[:2]

        return observable.Generic(get_vel_2d)

    @composer.observable
    def body_rotation(self):
        return observable.MJCFFeature('xquat', self.body)

    @composer.observable
    def body_rotation_matrix(self):
        def get_rotation_matrix(physics):
            quat = physics.bind(self.body).xquat
            return tr.quat_to_mat(quat).flatten()
        return observable.Generic(get_rotation_matrix)

    @composer.observable
    def sensors_vel(self):
        return observable.MJCFFeature('sensordata', self.get_sensor_mjcf('velocimeter'))

    @composer.observable
    def steering_pos(self):
        return observable.MJCFFeature('sensordata', self.get_sensor_mjcf('buddy_steering_pos'))

    @composer.observable
    def steering_vel(self):
        return observable.MJCFFeature('sensordata', self.get_sensor_mjcf('buddy_steering_vel'))

    @composer.observable
    def sensors_acc(self):
        return observable.MJCFFeature('sensordata', self.get_sensor_mjcf('buddy_accelerometer'))

    @composer.observable
    def sensors_gyro(self):
        return observable.MJCFFeature('sensordata', self.get_sensor_mjcf('buddy_gyro'))

    def _collect_from_attachments(self, attribute_name):
        out = []
        for entity in self._entity.iter_entities(exclude_self=True):
            out.extend(getattr(entity.observables, attribute_name, []))
        return out

    @property
    def kinematic_sensors(self):
        return ([self.sensors_vel] + [self.sensors_gyro] +
                self._collect_from_attachments('kinematic_sensors'))

    @property
    def all_observables(self):
        return [
            self.body_position,
            self.body_rotation,
            self.body_rotation_matrix,
            self.body_pose_2d,
            self.body_vel_2d,
            self.wheel_speeds,
            self.realsense_camera,
            self.steering_pos,
            self.steering_vel,
        ] + self.kinematic_sensors


class Car(composer.Robot):
    def _build(self, name='car'):
        model_path = "cars/pusher_car/buddy.xml"
        self._mjcf_root = mjcf.from_path(f'{model_path}')
        if name:
            self._mjcf_root.model = name

        self._actuators = self.mjcf_model.find_all('actuator')

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def actuators(self):
        return self._actuators

    def apply_action(self, physics, action, random_state=None):
        """Apply action to car's actuators. 
        `action` is expected to be a numpy array with [steering, throttle].
        """
        # steering, throttle = action
        # physics.bind(self.mjcf_model.find('actuator', 'buddy_steering_pos')).ctrl = steering
        # physics.bind(self.mjcf_model.find('actuator', 'buddy_throttle_velocity')).ctrl = throttle
        physics.bind(self.actuators).ctrl = action

    def _build_observables(self):
        return CarObservables(self)