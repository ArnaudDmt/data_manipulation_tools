import paper_colors
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from matplotlib.ticker import MaxNLocator
import plotly.express as px  # For color palette generation
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter,filtfilt



default_path = '.../Projects/HRP5_MultiContact_1'


contactNames = ["RightFootForceSensor"] #, "LeftFootForceSensor", "LeftHandForceSensor"] # ["RightFootForceSensor", "LeftFootForceSensor", "LeftHandForceSensor"]
contacts_area_when_set = [] # ["LeftHandForceSensor"]

contactNameToPlot = {"RightFootForceSensor": "Right foot", "LeftFootForceSensor": "Left foot", "LeftHandForceSensor": "Left hand"}

zeros_row = np.zeros((1, 3))

estimator_plot_args_default = {
    # Line 51 intersects the requested estimators with these keys, so anything missing here is
    # silently dropped from the figure rather than reported.
    'KO': {'name': 'KO', 'lineWidth': 1},
    'KO_ZPC': {'name': 'KO-ZPC', 'lineWidth': 1},
    'KO_WWS': {'name': 'KO-PC', 'lineWidth': 1},
    'Control': {'name': 'Control', 'lineWidth': 1},
    'WAIKO': {'name': 'WAIKO', 'lineWidth': 1},
    'Tilt': {'name': 'Valinor', 'lineWidth': 1},
    'WAIKO_NC': {'name': 'WAIKO_NC', 'lineWidth': 1},
    'Hartley': {'name': 'RI-EKF', 'lineWidth': 1},
    'Mocap': {'name': 'Ground truth', 'lineWidth': 1},
}

def continuous_euler(angles):
            continuous_angles = np.empty_like(angles)
            continuous_angles[0] = angles[0]
            for i in range(1, len(angles)):
                diff = angles[i] - angles[i-1]
                # Check each element of the diff array
                for j in range(len(diff)):
                    if diff[j] > np.pi:
                        diff[j] -= 2*np.pi
                    elif diff[j] < -np.pi:
                        diff[j] += 2*np.pi
                continuous_angles[i] = continuous_angles[i-1] + diff
            return continuous_angles

# Panels where the signal oscillates at step frequency, so every curve fills its own vertical band
# and whichever is drawn last hides the rest: the three velocities, and the roll and pitch. The KO
# curves are all drawn thinner there, so the bands stay distinguishable instead of one estimator
# painting over the others. Translation and yaw are smooth enough not to need it.
THIN_PANELS = {(1, 2), (2, 2), (1, 3), (2, 3), (3, 3)}
THIN_WIDTH = 0.4


def line_width(plot_args, observerName, row, col):
    # plot_args is passed in: inside plotPoseVel the parameter of that name shadows the module
    # dictionary, which is estimator_plot_args_default.
    if (row, col) in THIN_PANELS:
        return THIN_WIDTH
    return plot_args[observerName]["lineWidth"]


def plotPoseVel(estimators, path = default_path, colors = None, estimator_plot_args = estimator_plot_args_default):
        print(estimators)
        print(estimator_plot_args.keys())
        print(estimator_plot_args_default.keys())
        estimators = list(set(estimators).intersection(estimator_plot_args.keys()).intersection(estimator_plot_args_default.keys())) 
        
        for estimatorName in estimators:
                estimator_plot_args[estimatorName].update(estimator_plot_args_default[estimatorName])

        order = list(estimator_plot_args_default.keys())
        estimators = sorted(estimators, key=order.index)
        estimators.reverse()


        pos_x_inset = dict(cell=(1,1), l=0.20, w= 0.30, b= 0.20, h= 0.35)
        pos_y_inset = dict(cell=(2,1), l=0.10, w= 0.55, b= 0.65, h= 0.95)
        pos_z_inset = dict(cell=(3,1), l=0.15, w= 0.35, b= 0.55, h= 0.30)

        ori_roll_inset = dict(cell=(1,2), l=0.10, w= 0.55, b= 0.65, h= 0.95)
        ori_pitch_inset = dict(cell=(2,2), l=0.35, w= 0.55, b= 0.00, h= 0.30)
        ori_yaw_inset = dict(cell=(3,2), l=0.15, w= 0.30, b= 0.15, h= 0.30)

        vel_x_inset = dict(cell=(1,3), l=0.27, w= 0.25, b= 0.60, h= 0.40)
        vel_y_inset = dict(cell=(2,3), l=0.27, w= 0.25, b= 0.70, h= 0.40)
        vel_z_inset = dict(cell=(3,3), l=0.27, w= 0.25, b= 0.70, h= 0.45)

        # Which panels carry a zoom, and on what: (series, component, row, col). Position is left
        # out -- it drifts monotonically and a zoom adds nothing -- while orientation and velocity
        # oscillate at step frequency and are unreadable at full span.
        INSET_TARGETS = {
            'ori_pitch_inset': ('ori', 1, 2, 2),
            'vel_x_inset':     ('linVel', 0, 1, 3),
            'vel_y_inset':     ('linVel', 1, 2, 3),
            'vel_z_inset':     ('linVel', 2, 3, 3),
        }
        # Window each zoom covers, in seconds of the trial's own clock. Stated absolutely
        # rather than as a span around a computed centre: the interesting stretch was
        # chosen by eye on the data, and nothing in the code can rediscover it.
        INSET_RANGE = {'ori_pitch_inset': (226.4, 236.1),
                       # The velocity oscillates at step frequency: over the full 44 s the cycles
                       # merge, so its zoom stops at 142 s and keeps a handful of steps readable.
                       'vel_x_inset': (139.42, 141.6),
                       'vel_y_inset': (139.42, 141.6),
                       'vel_z_inset': (139.42, 141.6)}

        # Where the leader lines meet the inset box, when the automatic choice reads badly.
        INSET_LEADER_SIDE = {'vel_y_inset': 'left', 'vel_z_inset': 'left'}

        axis_idxs = dict()
        idx = 9
        insets = []

        def addInset(name, inset):
               nonlocal idx
               insets.append(inset)
               idx = idx + 1
               axis_idxs[name] = idx


        addInset('pos_x_inset', pos_x_inset)
        addInset('pos_y_inset', pos_y_inset)
        addInset('pos_z_inset', pos_z_inset)
        addInset('ori_roll_inset', ori_roll_inset)
        addInset('ori_pitch_inset', ori_pitch_inset)
        addInset('ori_yaw_inset', ori_yaw_inset)
        addInset('vel_x_inset', vel_x_inset)
        addInset('vel_y_inset', vel_y_inset)
        addInset('vel_z_inset', vel_z_inset)        
               

        figPoseVel = make_subplots(
        rows=3, cols=3, shared_xaxes=True, vertical_spacing=0.05, horizontal_spacing=0.09, insets=insets
        )

        figPoseVel.update_layout(
                template="plotly_white",
                legend=dict(
                        yanchor="bottom",
                        y=1.06,
                        xanchor="left",
                        x=0.01,
                        orientation="h",
                        bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Times New Roman"),
                ),
                legend_traceorder="reversed",
                font = dict(family = 'Times New Roman', size=10, color="black"),
                margin=dict(l=0.0,r=0.0,b=0.0,t=0.0)
                ,autosize=True  # Automatically adjusts the figure size
        )

        observer_data = pd.read_csv(f'{path}/output_data/finalDataCSV.csv',  delimiter=';')

        # observer_data["t"] = observer_data["t"] - 130

        # The robot stands still for the first two minutes; keeping that stretch squeezes the
        # walk into the right half of every panel. Cut at the first real displacement rather than
        # at a hard-coded index, which would not survive a change of trial.
        _mocap = observer_data[["Mocap_position_x", "Mocap_position_y"]].to_numpy()
        _moved = np.linalg.norm(_mocap - _mocap[0], axis=1) > 0.05
        _lead = int(round(5.0 / np.median(np.diff(observer_data["t"].to_numpy()[:1000]))))
        startIndex = max(int(np.argmax(_moved)) - _lead, 0) if _moved.any() else 0
        observer_data = observer_data.truncate(before=startIndex)

        # Reset the index to start from 0
        observer_data.reset_index(drop=True, inplace=True)

        # Pose and vels of the imu in the floating base
        posImuFb_overlap = observer_data[['HartleyIEKF_imuFbKine_position_x', 'HartleyIEKF_imuFbKine_position_y', 'HartleyIEKF_imuFbKine_position_z']].to_numpy()
        quaternions_rImuFb_overlap = observer_data[['HartleyIEKF_imuFbKine_ori_x', 'HartleyIEKF_imuFbKine_ori_y', 'HartleyIEKF_imuFbKine_ori_z', 'HartleyIEKF_imuFbKine_ori_w']].to_numpy()
        rImuFb_overlap = R.from_quat(quaternions_rImuFb_overlap)
        linVelImuFb_overlap = observer_data[['HartleyIEKF_imuFbKine_linVel_x', 'HartleyIEKF_imuFbKine_linVel_y', 'HartleyIEKF_imuFbKine_linVel_z']].to_numpy()
        angVelImuFb_overlap = observer_data[['HartleyIEKF_imuFbKine_angVel_x', 'HartleyIEKF_imuFbKine_angVel_y', 'HartleyIEKF_imuFbKine_angVel_z']].to_numpy()

        posFbImu_overlap = - rImuFb_overlap.apply(posImuFb_overlap, inverse=True)
        linVelFbImu_overlap = rImuFb_overlap.apply(np.cross(angVelImuFb_overlap, posImuFb_overlap), inverse=True) - rImuFb_overlap.apply(linVelImuFb_overlap, inverse=True)


        # Build only the estimators this run logged. The hardcoded dict this replaces was edited
        # by hand for whichever comparison was last made -- the Kinetics Observer was commented
        # out and VALINOR left active -- so it raised KeyError on any run not containing exactly
        # those observers.
        estimatorsPoses = {}
        available = set(observer_data.columns)
        for name in ('Mocap', 'KO', 'KO_ZPC', 'KO_WWS', 'Hartley', 'Tilt', 'WAIKO', 'WAIKO_NC', 'Control'):
            position = [f'{name}_position_{axis}' for axis in 'xyz']
            orientation = [f'{name}_orientation_{axis}' for axis in 'xyzw']
            if not set(position + orientation) <= available:
                continue
            entry = {'pos': observer_data[position].to_numpy(),
                     'ori': R.from_quat(observer_data[orientation].to_numpy()),
                     'linVel': None, 'angVel': None}
            for kind in ('linVel', 'angVel'):
                columns = [f'{name}_{kind}_{axis}' for axis in 'xyz']
                if set(columns) <= available:
                    entry[kind] = observer_data[columns].to_numpy()
            estimatorsPoses[name] = entry
        
        # Velocity of Hartley (different as we already have the velocity of the IMU)
        # estimated velocity
        if "Hartley" in estimators:
                linVelImu_Hartley_overlap = observer_data[['Hartley_IMU_linVel_x', 'Hartley_IMU_linVel_y', 'Hartley_IMU_linVel_z']].to_numpy()
                quaternionsHartley_fb_overlap = observer_data[['Hartley_orientation_x', 'Hartley_orientation_y', 'Hartley_orientation_z', 'Hartley_orientation_w']].to_numpy()
                rHartley_fb_overlap = R.from_quat(quaternionsHartley_fb_overlap)
                rWorldImuHartley_overlap = rHartley_fb_overlap * rImuFb_overlap.inv()
                locVelHartley_imu_estim = rWorldImuHartley_overlap.apply(linVelImu_Hartley_overlap, inverse=True)
                estimatorsPoses["Hartley"]["linVel"] = locVelHartley_imu_estim
        

        # Velocity of the mocap (different as we only have the position)
        posMocap_overlap = observer_data[['Mocap_position_x', 'Mocap_position_y', 'Mocap_position_z']].to_numpy()
        quaternionsMocap_overlap = observer_data[['Mocap_orientation_x', 'Mocap_orientation_y', 'Mocap_orientation_z', 'Mocap_orientation_w']].to_numpy()
        rMocap_overlap = R.from_quat(quaternionsMocap_overlap)
        posMocap_imu_overlap = posMocap_overlap + rMocap_overlap.apply(posFbImu_overlap)
        velMocap_imu_overlap = np.diff(posMocap_imu_overlap, axis=0)/0.005
        velMocap_imu_overlap = np.vstack((zeros_row,velMocap_imu_overlap))
        rWorldImuMocap_overlap = rMocap_overlap * rImuFb_overlap.inv()
        locVelMocap_imu_estim = rWorldImuMocap_overlap.apply(velMocap_imu_overlap, inverse=True)
        b, a = butter(2, 0.15, analog=False)
        locVelMocap_imu_estim = filtfilt(b, a, locVelMocap_imu_estim, axis=0)
        estimatorsPoses["Mocap"]["linVel"] = locVelMocap_imu_estim       
        

        print(estimators)
        if "Hartley" in estimators:
                estimatorsPoses["Hartley"]["ori2"] = estimatorsPoses["Hartley"]["ori"].apply(np.array([0.0, 0.0, 1.0]), inverse=True) 
                estimatorsPoses["Hartley"]["ori2"] = np.degrees(estimatorsPoses["Hartley"]["ori2"])
                estimatorsPoses["Hartley"]["ori"] = estimatorsPoses["Hartley"]["ori"].as_euler('xyz')
                estimatorsPoses["Hartley"]["ori"] = np.degrees(continuous_euler(estimatorsPoses["Hartley"]["ori"]))
                

        estimatorsPoses["Mocap"]["ori2"] = estimatorsPoses["Mocap"]["ori"].apply(np.array([0.0, 0.0, 1.0]), inverse=True) 
        estimatorsPoses["Mocap"]["ori2"] = np.degrees(estimatorsPoses["Mocap"]["ori2"])
        estimatorsPoses["Mocap"]["ori"] = estimatorsPoses["Mocap"]["ori"].as_euler('xyz')
        estimatorsPoses["Mocap"]["ori"] = np.degrees(continuous_euler(estimatorsPoses["Mocap"]["ori"]))
        
        print( estimatorsPoses["Mocap"]["ori2"] )

        # index_t_z_40 = 7990
        # index_t_z_50 = 10001

        # index_t_yaw_200 = 49999
        # index_t_yaw_240 = 60001

        # index_t_vel_139_5 = 27899
        # index_t_vel_141_5 = 28301

        #     positions = estimatorsPoses["Mocap"]["pos"][index_t_z_40:index_t_z_50 + 1]

        # Initialize cumulative distance
        #     cumulative_distance = 0.0

        #     # Iterate over consecutive pairs of points
        #     for i in range(len(positions) - 1):
        #         # Get current and next position (only x and y components)
        #         pos_current = positions[i][:2]  # Take x and y components
        #         pos_next = positions[i + 1][:2]  # Take x and y components
                
        #         # Compute the 2D distance between consecutive points
        #         distance = np.sqrt((pos_next[0] - pos_current[0])**2 + (pos_next[1] - pos_current[1])**2)
                
        #         # Add to cumulative distance
        #         cumulative_distance += distance

        #     print(f"Cumulative 2D Distance along x and y: {cumulative_distance}")


        rect_lims = {"pos_x": [None, None, None, None], "pos_y": [None, None, None, None], "pos_z": [None, None, None, None], "roll": [None, None, None, None], "pitch": [None, None, None, None], "yaw": [None, None, None, None], "vel_x": [None, None, None, None], "vel_y": [None, None, None, None], "vel_z": [None, None, None, None]}
        
        def computeObserverLocVel(observerName):
                linVelObserver_imu_overlap = estimatorsPoses[observerName]["linVel"] + np.cross(estimatorsPoses[observerName]["angVel"], estimatorsPoses[observerName]["ori"].apply(posFbImu_overlap)) + estimatorsPoses[observerName]["ori"].apply(linVelFbImu_overlap)

                rWorldImuObserver_overlap = estimatorsPoses[observerName]["ori"] * rImuFb_overlap.inv()
                locVelObserver_imu_estim = rWorldImuObserver_overlap.apply(linVelObserver_imu_overlap, inverse=True)
                estimatorsPoses[observerName]["linVel"] = locVelObserver_imu_estim

                estimatorsPoses[observerName]["ori2"] = estimatorsPoses[observerName]["ori"].apply(np.array([0.0, 0.0, 1.0]), inverse=True) 
                estimatorsPoses[observerName]["ori2"] = np.degrees(estimatorsPoses[observerName]["ori2"])
                estimatorsPoses[observerName]["ori"] = estimatorsPoses[observerName]["ori"].as_euler('xyz')
                estimatorsPoses[observerName]["ori"] = np.degrees(continuous_euler(estimatorsPoses[observerName]["ori"]))

        for estimator in estimators:
                if estimator in estimator_plot_args and estimator in estimatorsPoses.keys():
                
                        if estimator not in ["Mocap", "Hartley"]:
                                computeObserverLocVel(estimator)

        #         x_min_z = observer_data["t"][index_t_z_40]
        #         x_max_z = observer_data["t"][index_t_z_50]

        #         x_min_pitch = observer_data["t"][index_t_z_40]
        #         x_max_pitch = observer_data["t"][index_t_z_50]

        #         x_min_yaw = observer_data["t"][index_t_yaw_200]
        #         x_max_yaw = observer_data["t"][index_t_yaw_240]

        #         x_min_vel = observer_data["t"][index_t_vel_139_5]
        #         x_max_vel = observer_data["t"][index_t_vel_141_5]

        #         y_min_pos_x = np.min(estimatorsPoses[estimator]["pos"][index_t_yaw_200:index_t_yaw_240, 0])
        #         y_max_pos_x = np.max(estimatorsPoses[estimator]["pos"][index_t_yaw_200:index_t_yaw_240, 0])
        #         y_min_pos_y = np.min(estimatorsPoses[estimator]["pos"][index_t_yaw_200:index_t_yaw_240, 1])


        #         y_max_pos_y = np.max(estimatorsPoses[estimator]["pos"][index_t_yaw_200:index_t_yaw_240, 1])
        #         y_min_pos_z = np.min(estimatorsPoses[estimator]["pos"][index_t_yaw_200:index_t_yaw_240, 2])
        #         y_max_pos_z = np.max(estimatorsPoses[estimator]["pos"][index_t_yaw_200:index_t_yaw_240, 2])
                
        #         y_min_roll = np.min(estimatorsPoses[estimator]["ori2"][index_t_z_40:index_t_z_50, 0])
        #         y_max_roll = np.max(estimatorsPoses[estimator]["ori2"][index_t_z_40:index_t_z_50, 0])
        #         y_min_pitch = np.min(estimatorsPoses[estimator]["ori2"][index_t_z_40:index_t_z_50, 1])
        #         y_max_pitch = np.max(estimatorsPoses[estimator]["ori2"][index_t_z_40:index_t_z_50, 1])
        #         y_min_yaw = np.min(estimatorsPoses[estimator]["ori"][index_t_yaw_200:index_t_yaw_240, 2])
        #         y_max_yaw = np.max(estimatorsPoses[estimator]["ori"][index_t_yaw_200:index_t_yaw_240, 2])

        #         y_min_vel_x = np.min(estimatorsPoses[estimator]["linVel"][index_t_vel_139_5:index_t_vel_141_5, 0])
        #         y_max_vel_x = np.max(estimatorsPoses[estimator]["linVel"][index_t_vel_139_5:index_t_vel_141_5, 0])
        #         y_min_vel_y = np.min(estimatorsPoses[estimator]["linVel"][index_t_vel_139_5:index_t_vel_141_5, 1])
        #         y_max_vel_y = np.max(estimatorsPoses[estimator]["linVel"][index_t_vel_139_5:index_t_vel_141_5, 1])
        #         y_min_vel_z = np.min(estimatorsPoses[estimator]["linVel"][index_t_vel_139_5:index_t_vel_141_5, 2])
        #         y_max_vel_z = np.max(estimatorsPoses[estimator]["linVel"][index_t_vel_139_5:index_t_vel_141_5, 2])

        #         # Update global limits
        #         rect_lims["pos_x"][0] = x_min_yaw if rect_lims["pos_x"][0] is None else min(rect_lims["pos_x"][0], x_min_yaw)
        #         rect_lims["pos_x"][1] = x_max_yaw if rect_lims["pos_x"][1] is None else max(rect_lims["pos_x"][1], x_max_yaw)
        #         rect_lims["pos_x"][2] = y_min_pos_x if rect_lims["pos_x"][2] is None else min(rect_lims["pos_x"][2], y_min_pos_x)
        #         rect_lims["pos_x"][3] = y_max_pos_x if rect_lims["pos_x"][3] is None else max(rect_lims["pos_x"][3], y_max_pos_x)

        #         rect_lims["pos_y"][0] = x_min_yaw if rect_lims["pos_y"][0] is None else min(rect_lims["pos_y"][0], x_min_yaw)
        #         rect_lims["pos_y"][1] = x_max_yaw if rect_lims["pos_y"][1] is None else max(rect_lims["pos_y"][1], x_max_yaw)
        #         rect_lims["pos_y"][2] = y_min_pos_y if rect_lims["pos_y"][2] is None else min(rect_lims["pos_y"][2], y_min_pos_y)
        #         rect_lims["pos_y"][3] = y_max_pos_y if rect_lims["pos_y"][3] is None else max(rect_lims["pos_y"][3], y_max_pos_y)

        #         rect_lims["pos_z"][0] = x_min_z if rect_lims["pos_z"][0] is None else min(rect_lims["pos_z"][0], x_min_z)
        #         rect_lims["pos_z"][1] = x_max_z if rect_lims["pos_z"][1] is None else max(rect_lims["pos_z"][1], x_max_z)
        #         rect_lims["pos_z"][2] = y_min_pos_z if rect_lims["pos_z"][2] is None else min(rect_lims["pos_z"][2], y_min_pos_z)
        #         rect_lims["pos_z"][3] = y_max_pos_z if rect_lims["pos_z"][3] is None else max(rect_lims["pos_z"][3], y_max_pos_z)

        #         rect_lims["roll"][0] = x_min_z if rect_lims["roll"][0] is None else min(rect_lims["roll"][0], x_min_z)
        #         rect_lims["roll"][1] = x_max_z if rect_lims["roll"][1] is None else max(rect_lims["roll"][1], x_max_z)
        #         rect_lims["roll"][2] = y_min_roll if rect_lims["roll"][2] is None else min(rect_lims["roll"][2], y_min_roll)
        #         rect_lims["roll"][3] = y_max_roll if rect_lims["roll"][3] is None else max(rect_lims["roll"][3], y_max_roll) 

        #         rect_lims["pitch"][0] = x_min_pitch if rect_lims["pitch"][0] is None else min(rect_lims["pitch"][0], x_min_pitch)
        #         rect_lims["pitch"][1] = x_max_pitch if rect_lims["pitch"][1] is None else max(rect_lims["pitch"][1], x_max_pitch)
        #         rect_lims["pitch"][2] = y_min_pitch if rect_lims["pitch"][2] is None else min(rect_lims["pitch"][2], y_min_pitch)
        #         rect_lims["pitch"][3] = y_max_pitch if rect_lims["pitch"][3] is None else max(rect_lims["pitch"][3], y_max_pitch) 

        #         rect_lims["yaw"][0] = x_min_yaw if rect_lims["yaw"][0] is None else min(rect_lims["yaw"][0], x_min_yaw)
        #         rect_lims["yaw"][1] = x_max_yaw if rect_lims["yaw"][1] is None else max(rect_lims["yaw"][1], x_max_yaw)
        #         rect_lims["yaw"][2] = y_min_yaw if rect_lims["yaw"][2] is None else min(rect_lims["yaw"][2], y_min_yaw)
        #         rect_lims["yaw"][3] = y_max_yaw if rect_lims["yaw"][3] is None else max(rect_lims["yaw"][3], y_max_yaw) 

        #         rect_lims["vel_x"][0] = x_min_vel if rect_lims["vel_x"][0] is None else min(rect_lims["vel_x"][0], x_min_vel)
        #         rect_lims["vel_x"][1] = x_max_vel if rect_lims["vel_x"][1] is None else max(rect_lims["vel_x"][1], x_max_vel)
        #         rect_lims["vel_x"][2] = y_min_vel_x if rect_lims["vel_x"][2] is None else min(rect_lims["vel_x"][2], y_min_vel_x)
        #         rect_lims["vel_x"][3] = y_max_vel_x if rect_lims["vel_x"][3] is None else max(rect_lims["vel_x"][3], y_max_vel_x)

        #         rect_lims["vel_y"][0] = x_min_vel if rect_lims["vel_y"][0] is None else min(rect_lims["vel_y"][0], x_min_vel)
        #         rect_lims["vel_y"][1] = x_max_vel if rect_lims["vel_y"][1] is None else max(rect_lims["vel_y"][1], x_max_vel)
        #         rect_lims["vel_y"][2] = y_min_vel_y if rect_lims["vel_y"][2] is None else min(rect_lims["vel_y"][2], y_min_vel_y)
        #         rect_lims["vel_y"][3] = y_max_vel_y if rect_lims["vel_y"][3] is None else max(rect_lims["vel_y"][3], y_max_vel_y)

        #         rect_lims["vel_z"][0] = x_min_vel if rect_lims["vel_z"][0] is None else min(rect_lims["vel_z"][0], x_min_vel)
        #         rect_lims["vel_z"][1] = x_max_vel if rect_lims["vel_z"][1] is None else max(rect_lims["vel_z"][1], x_max_vel)
        #         rect_lims["vel_z"][2] = y_min_vel_z if rect_lims["vel_z"][2] is None else min(rect_lims["vel_z"][2], y_min_vel_z)
        #         rect_lims["vel_z"][3] = y_max_vel_z if rect_lims["vel_z"][3] is None else max(rect_lims["vel_z"][3], y_max_vel_z)


        # One window per zoom, taken well inside the walk so the robot is at steady pace.
        _time = observer_data["t"].to_numpy()
        INSET_WINDOW = {name: (int(np.searchsorted(_time, lo)), int(np.searchsorted(_time, hi)))
                        for name, (lo, hi) in INSET_RANGE.items()}

        for _name, (_series, _component, _row, _col) in INSET_TARGETS.items():
            _lo, _hi = INSET_WINDOW[_name]
            _n = (_row - 1) * 3 + _col
            _values = np.concatenate([estimatorsPoses[e][_series][_lo:_hi, _component]
                                      for e in estimators if estimatorsPoses[e].get(_series) is not None])
            _pad = 0.12 * (np.nanmax(_values) - np.nanmin(_values) or 1.0)
            figPoseVel.add_shape(
                type="rect", xref=f"x{_n}", yref=f"y{_n}",
                x0=_time[_lo], x1=_time[_hi - 1],
                y0=np.nanmin(_values) - _pad, y1=np.nanmax(_values) + _pad,
                line=dict(color="grey", width=1), layer="above")

        def plotPoseAndVel(observerName):
                color_Observer = paper_colors.rgba(colors, observerName)
                figPoseVel.add_trace(
                go.Scatter(
                        x=observer_data["t"],
                        y=estimatorsPoses[observerName]["pos"][:, 0],
                        mode="lines",showlegend= False,
                        line=dict(width=line_width(estimator_plot_args, observerName, 1, 1), color=color_Observer)
                ),
                row=1,
                col=1,
                )
                figPoseVel.add_trace(
                go.Scatter(
                        x=observer_data["t"],
                        y=estimatorsPoses[observerName]["pos"][:, 1],
                        mode="lines",showlegend= False,
                        line=dict(width=line_width(estimator_plot_args, observerName, 2, 1), color=color_Observer)
                ),
                row=2,
                col=1,
                )
                figPoseVel.add_trace(
                go.Scatter(
                        x=observer_data["t"],
                        y=estimatorsPoses[observerName]["pos"][:, 2],
                        mode="lines",showlegend= False,
                        line=dict(width=line_width(estimator_plot_args, observerName, 3, 1), color=color_Observer)
                ),
                row=3,
                col=1,
                )        
                
                
                figPoseVel.add_trace(
                go.Scatter(
                        x=observer_data["t"],
                        y=estimatorsPoses[observerName]["ori"][:, 0],
                        mode="lines",showlegend= False,
                        line=dict(width=line_width(estimator_plot_args, observerName, 1, 2), color=color_Observer)
                ),
                row=1,
                col=2,
                )

                figPoseVel.add_trace(
                go.Scatter(
                        x=observer_data["t"],
                        y=estimatorsPoses[observerName]["ori"][:, 1],
                        mode="lines",showlegend= False,
                        line=dict(width=line_width(estimator_plot_args, observerName, 2, 2), color=color_Observer)
                ),
                row=2,
                col=2,
                )

                figPoseVel.add_trace(
                go.Scatter(
                        x=observer_data["t"],
                        # ori2 is the gravity direction. Its x and y components approximate roll and
                        # pitch for small tilts, which is what the two panels above plot, but its z
                        # component is ~1 for an upright robot and np.degrees turned that into a flat
                        # 57.3 "yaw". The yaw is the third unwrapped Euler angle, already in degrees.
                        y=estimatorsPoses[observerName]["ori"][:, 2],
                        mode="lines",showlegend= False,
                        line=dict(width=line_width(estimator_plot_args, observerName, 3, 2), color=color_Observer)
                ),
                row=3,
                col=2,
                )


                figPoseVel.add_trace(
                go.Scatter(
                        x=observer_data["t"],
                        y=estimatorsPoses[observerName]["linVel"][:, 0],
                        mode="lines",showlegend= False,
                        line=dict(width=line_width(estimator_plot_args, observerName, 1, 3), color=color_Observer)
                ),
                row=1,
                col=3,
                )
                figPoseVel.add_trace(
                go.Scatter(
                        x=observer_data["t"],
                        y=estimatorsPoses[observerName]["linVel"][:, 1],
                        mode="lines",showlegend= False,
                        line=dict(width=line_width(estimator_plot_args, observerName, 2, 3), color=color_Observer)
                ),
                row=2,
                col=3,
                )
                figPoseVel.add_trace(
                go.Scatter(
                        x=observer_data["t"],
                        y=estimatorsPoses[observerName]["linVel"][:, 2],
                        mode="lines",showlegend= False,
                        line=dict(width=line_width(estimator_plot_args, observerName, 3, 3), color=color_Observer)
                ),
                row=3,
                col=3,
                )

                # # Add the inset plot as an additional trace
                # figPoseVel.add_trace(
                # go.Scatter(
                #         x=observer_data["t"][index_t_yaw_200:index_t_yaw_240],
                #         y=estimatorsPoses[observerName]["pos"][index_t_yaw_200:index_t_yaw_240, 0],
                #         mode='lines',
                #         showlegend= False,
                #         line=dict(width=estimator_plot_args[observerName]["lineWidth"]/2, color=color_Observer),
                #         xaxis=f"x{axis_idxs['pos_x_inset']}",
                #         yaxis=f"y{axis_idxs['pos_x_inset']}",
                # )
                # )

                

                # # Add a rectangle to subplot (1,1) surrounding the inset plot
                # figPoseVel.add_shape(
                # type="rect",
                # xref="x1",  # Absolute positioning on the x-axis of subplot (3,3)
                # yref="y1",  # Absolute positioning on the y-axis of subplot (3,3)
                # x0=rect_lims["pos_x"][0],  # Start of x-range
                # x1=rect_lims["pos_x"][1],  # End of x-range
                # y0=rect_lims["pos_x"][2],  # Start of y-range
                # y1=rect_lims["pos_x"][3],  # End of y-range
                # line=dict(color="grey", width=1),
                # layer="above"  # Ensures the rectangle appears above the plot
                # )



                # figPoseVel.add_trace(
                # go.Scatter(
                #         x=observer_data["t"][index_t_z_40:index_t_z_50],
                #         y=estimatorsPoses[observerName]["pos"][index_t_z_40:index_t_z_50, 2],
                #         mode='lines',
                #         showlegend= False,
                #         line=dict(width=estimator_plot_args[observerName]["lineWidth"]/2, color=color_Observer),
                #         xaxis='x12', 
                #         yaxis='y12'
                # )
                # )

                # # Add a rectangle to subplot (1,1) surrounding the inset plot
                # figPoseVel.add_shape(
                #     type="rect",
                #     xref="x4",  # Absolute positioning on the x-axis of subplot (3,3)
                #     yref="y4",  # Absolute positioning on the y-axis of subplot (3,3)
                #     x0=rect_lims["pos_y"][0],  # Start of x-range
                #     x1=rect_lims["pos_y"][1],  # End of x-range
                #     y0=rect_lims["pos_y"][2],  # Start of y-range
                #     y1=rect_lims["pos_y"][3],  # End of y-range
                #     line=dict(color="grey", width=1),
                #     layer="above"  # Ensures the rectangle appears above the plot
                # )

                # # Add a rectangle to subplot (1,1) surrounding the inset plot
                # figPoseVel.add_shape(
                # type="rect",
                # xref="x7",  # Absolute positioning on the x-axis of subplot (3,3)
                # yref="y7",  # Absolute positioning on the y-axis of subplot (3,3)
                # x0=rect_lims["pos_z"][0],  # Start of x-range
                # x1=rect_lims["pos_z"][1],  # End of x-range
                # y0=rect_lims["pos_z"][2],  # Start of y-range
                # y1=rect_lims["pos_z"][3],  # End of y-range
                # line=dict(color="grey", width=1),
                # layer="above"  # Ensures the rectangle appears above the plot
                # )
                

                # if estimator != "Mocap":
                #         # Add the inset plot a  s an additional trace
                #         figPoseVel.add_trace(
                #         go.Scatter(
                #                 x=observer_data["t"][index_t_yaw_200:index_t_yaw_240],
                #                 y=estimatorsPoses[observerName]["pos"][index_t_yaw_200:index_t_yaw_240, 2],
                #                 mode='lines',
                #                 showlegend= False,
                #                 line=dict(width=estimator_plot_args[observerName]["lineWidth"]/2, color=color_Observer),
                #                 xaxis=f"x{axis_idxs['pos_z_inset']}",
                #                 yaxis=f"y{axis_idxs['pos_z_inset']}",
                #         )
                #         )
                

                
                # figPoseVel.add_trace(
                # go.Scatter(
                #         x=observer_data["t"][index_t_z_40:index_t_z_50],
                #         y=estimatorsPoses[observerName]["ori2"][index_t_z_40:index_t_z_50, 0],
                #         mode='lines',
                #         showlegend= False,
                #         line=dict(width=estimator_plot_args[observerName]["lineWidth"]/2, color=color_Observer),
                #         xaxis=f"x{axis_idxs['ori_roll_inset']}",
                #         yaxis=f"y{axis_idxs['ori_roll_inset']}",
                # )
                # )

                # figPoseVel.add_shape(
                # type="rect",
                # xref="x2",  # Absolute positioning on the x-axis of subplot (3,3)
                # yref="y2",  # Absolute positioning on the y-axis of subplot (3,3)
                # x0=rect_lims["roll"][0],  # Start of x-range
                # x1=rect_lims["roll"][1],  # End of x-range
                # y0=rect_lims["roll"][2] * 1.1,  # Start of y-range
                # y1=rect_lims["roll"][3] * 1.1,  # End of y-range
                # line=dict(color="grey", width=1),
                # layer="above"  # Ensures the rectangle appears above the plot
                # )

 
                

                # figPoseVel.add_trace(
                # go.Scatter(
                #         x=observer_data["t"][index_t_z_40:index_t_z_50],
                #         y=estimatorsPoses[observerName]["ori2"][index_t_z_40:index_t_z_50, 1],
                #         mode='lines',
                #         showlegend= False,
                #         line=dict(width=estimator_plot_args[observerName]["lineWidth"]/2, color=color_Observer),
                #         xaxis=f"x{axis_idxs['ori_pitch_inset']}",
                #         yaxis=f"y{axis_idxs['ori_pitch_inset']}",
                # )
                # )

                # # Add a rectangle to subplot (1,1) surrounding the inset plot
                # figPoseVel.add_shape(
                # type="rect",
                # xref="x5",  # Absolute positioning on the x-axis of subplot (3,3)
                # yref="y5",  # Absolute positioning on the y-axis of subplot (3,3)
                # x0=rect_lims["pitch"][0],  # Start of x-range
                # x1=rect_lims["pitch"][1],  # End of x-range
                # y0=rect_lims["pitch"][2] * 1.1,  # Start of y-range
                # y1=rect_lims["pitch"][3] * 1.1,  # End of y-range
                # line=dict(color="grey", width=1),
                # layer="above"  # Ensures the rectangle appears above the plot
                # )


                # # # Add the inset plot as an additional trace
                # # figPoseVel.add_trace(
                # # go.Scatter(
                # #         x=observer_data["t"][index_t_yaw_200:index_t_yaw_240],
                # #         # y=estimatorsPoses[observerName]["ori"][index_t_yaw_200:index_t_yaw_240, 2],
                # #         y=estimatorsPoses[observerName]["ori"][index_t_yaw_200:index_t_yaw_240, 2],
                # #         mode='lines',
                # #         showlegend= False,
                # #         line=dict(width=estimator_plot_args[observerName]["lineWidth"]/2, color=color_Observer),
                # #         xaxis='x13', 
                # #         yaxis='y13'
                # # )
                # # )

                # # # Add a rectangle to subplot (1,1) surrounding the inset plot
                # # figPoseVel.add_shape(
                # # type="rect",
                # # xref="x8",  # Absolute positioning on the x-axis of subplot (3,3)
                # # yref="y8",  # Absolute positioning on the y-axis of subplot (3,3)
                # # x0=rect_lims["yaw"][0],  # Start of x-range
                # # x1=rect_lims["yaw"][1],  # End of x-range
                # # y0=rect_lims["yaw"][2],  # Start of y-range
                # # y1=rect_lims["yaw"][3],  # End of y-range
                # # line=dict(color="grey", width=1),
                # # layer="above"  # Ensures the rectangle appears above the plot
                # # )

                

                # # Add the inset plot as an additional trace
                # figPoseVel.add_trace(
                # go.Scatter(
                #         x=observer_data["t"][index_t_vel_139_5:index_t_vel_141_5],
                #         y=estimatorsPoses[observerName]["linVel"][index_t_vel_139_5:index_t_vel_141_5, 0],
                #         mode='lines',
                #         showlegend= False,
                #         line=dict(width=estimator_plot_args[observerName]["lineWidth"]/2, color=color_Observer),
                #         xaxis=f"x{axis_idxs['vel_x_inset']}",
                #         yaxis=f"y{axis_idxs['vel_x_inset']}",
                # )
                # )


                # # Add a rectangle to subplot (7,7) surrounding the inset plot
                # figPoseVel.add_shape(
                # type="rect",
                # xref="x3",  # Absolute positioning on the x-axis of subplot (3,3)
                # yref="y3",  # Absolute positioning on the y-axis of subplot (3,3)
                # x0=rect_lims["vel_x"][0],  # Start of x-range
                # x1=rect_lims["vel_x"][1],  # End of x-range
                # y0=rect_lims["vel_x"][2] * 1.1,  # Start of y-range
                # y1=rect_lims["vel_x"][3] * 1.1,  # End of y-range
                # line=dict(color="grey", width=1),
                # layer="above"  # Ensures the rectangle appears above the plot
                # )

                # # Add the inset plot as an additional trace
                # figPoseVel.add_trace(
                # go.Scatter(
                #         x=observer_data["t"][index_t_vel_139_5:index_t_vel_141_5],
                #         y=estimatorsPoses[observerName]["linVel"][index_t_vel_139_5:index_t_vel_141_5, 1],
                #         mode='lines',
                #         showlegend= False,
                #         line=dict(width=estimator_plot_args[observerName]["lineWidth"]/2, color=color_Observer),
                #         xaxis=f"x{axis_idxs['vel_y_inset']}",
                #         yaxis=f"y{axis_idxs['vel_y_inset']}",
                # )
                # )
                # # Add a rectangle to subplot (1,1) surrounding the inset plot
                # figPoseVel.add_shape(
                # type="rect",
                # xref="x6",  # Absolute positioning on the x-axis of subplot (3,3)
                # yref="y6",  # Absolute positioning on the y-axis of subplot (3,3)
                # x0=rect_lims["vel_y"][0],  # Start of x-range
                # x1=rect_lims["vel_y"][1],  # End of x-range
                # y0=rect_lims["vel_y"][2] * 1.1,  # Start of y-range
                # y1=rect_lims["vel_y"][3] * 1.1,  # End of y-range
                # line=dict(color="grey", width=1),
                # layer="above"  # Ensures the rectangle appears above the plot
                # )

                # # Add the inset plot as an additional trace
                # figPoseVel.add_trace(
                # go.Scatter(
                #         x=observer_data["t"][index_t_vel_139_5:index_t_vel_141_5],
                #         y=estimatorsPoses[observerName]["linVel"][index_t_vel_139_5:index_t_vel_141_5, 2],
                #         mode='lines',
                #         showlegend= False,
                #         line=dict(width=estimator_plot_args[observerName]["lineWidth"]/2, color=color_Observer),
                #         xaxis=f"x{axis_idxs['vel_z_inset']}",
                #         yaxis=f"y{axis_idxs['vel_z_inset']}",
                # )
                # )
                # # Add a rectangle to subplot (1,1) surrounding the inset plot
                # figPoseVel.add_shape(
                # type="rect",
                # xref="x9",  # Absolute positioning on the x-axis of subplot (3,3)
                # yref="y9",  # Absolute positioning on the y-axis of subplot (3,3)
                # x0=rect_lims["vel_z"][0],  # Start of x-range
                # x1=rect_lims["vel_z"][1],  # End of x-range
                # y0=rect_lims["vel_z"][2] * 1.1,  # Start of y-range
                # y1=rect_lims["vel_z"][3] * 1.1,  # End of y-range
                # line=dict(color="grey", width=1),
                # layer="above"  # Ensures the rectangle appears above the plot
                # )

                figPoseVel.update_layout(
                        xaxis=dict(
                                dtick=50, gridwidth=1, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=False, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman',  color="black")),
                        xaxis1=dict(
                                dtick=50, gridwidth=1, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=False, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman',  color="black")),
                        xaxis2=dict(
                                dtick=50, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=False, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=10, color="black")),
                        xaxis3=dict(
                                dtick=50, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=False, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=10, color="black")),
                        xaxis4=dict(
                                dtick=50, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=False, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=10, color="black")),
                        xaxis5=dict(
                                dtick=50, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=False, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=10, color="black")),
                        xaxis6=dict(
                                dtick=50, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=False, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=10, color="black")),
                        xaxis7=dict(
                                dtick=50, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=False, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=10, color="black")),
                        xaxis8=dict(
                                dtick=50, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=False, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=10, color="black")),
                        xaxis9=dict(
                                dtick=50, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=False, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=10, color="black")),
                        yaxis=dict(
                                gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', zerolinewidth = 1, linecolor= 'lightgrey', mirror=True, ticks='outside', showline=False, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=10, color="black")),
                        yaxis1=dict(
                                gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', zerolinewidth = 1, linecolor= 'lightgrey', mirror=True, ticks='outside', showline=False, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=10, color="black")),
                        yaxis2=dict(
                                gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', zerolinewidth = 1, linecolor= 'lightgrey', mirror=True, ticks='outside', showline=False, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=10, color="black"), dtick = 2),
                        yaxis3=dict(
                                gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', zerolinewidth = 1, linecolor= 'lightgrey', mirror=True, ticks='outside', showline=False, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=10, color="black"), ),
                        yaxis4=dict(
                                gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', zerolinewidth = 1, linecolor= 'lightgrey', mirror=True, ticks='outside', showline=False, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=10, color="black")),
                        yaxis5=dict(
                                gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', zerolinewidth = 1, linecolor= 'lightgrey', mirror=True, ticks='outside', showline=False, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=10, color="black")),
                        yaxis6=dict(
                                gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', zerolinewidth = 1, linecolor= 'lightgrey', mirror=True, ticks='outside', showline=False, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=10, color="black")),
                        yaxis7=dict(
                                gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', zerolinewidth = 1, linecolor= 'lightgrey', mirror=True, ticks='outside', showline=False, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=10, color="black"),dtick = 0.20),
                        yaxis8=dict(
                                gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', zerolinewidth = 1, linecolor= 'lightgrey', mirror=True, ticks='outside', showline=False, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=10, color="black")),
                        yaxis9=dict(
                                gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', zerolinewidth = 1, linecolor= 'lightgrey', mirror=True, ticks='outside', showline=False, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=10, color="black"))
                        )
                
                # figPoseVel.update_layout(
                #         xaxis10=dict(
                #                 dtick=20, gridcolor= 'lightgrey', zerolinecolor= 'darkgrey', linecolor= 'darkgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black")),
                #         yaxis10=dict(
                #                 gridcolor= 'lightgrey', zerolinecolor= 'darkgrey', linecolor= 'darkgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"))
                #         )
                
                # figPoseVel.update_layout(
                #         xaxis11=dict(
                #                 dtick=20, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'darkgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black")),
                #         yaxis11=dict(
                #                 gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'darkgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"))
                #         )
                
                # figPoseVel.update_layout(
                #         xaxis12=dict(
                #                 dtick=10, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'darkgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"),),
                #         yaxis12=dict(
                #                 gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'darkgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"),)
                #         )
                
                # figPoseVel.update_layout(
                #         xaxis13=dict(
                #                 dtick=5, linewidth=0.5, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'darkgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"),),
                #         yaxis13=dict(
                #                 gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'darkgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"), dtick=2)
                #         )
                
                # figPoseVel.update_layout(
                #         xaxis14=dict(
                #                 dtick=5, linewidth=0.5, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'darkgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"),),
                #         yaxis14=dict(
                #                 gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'darkgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"),)
                #         )
                
                # figPoseVel.update_layout(
                #         xaxis15=dict(
                #                 dtick=5, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'darkgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"),),
                #         yaxis15=dict(
                #                 gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'darkgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"),)
                #         )
                
                # figPoseVel.update_layout(
                #         xaxis16=dict(
                #                 dtick=1, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'darkgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"),),
                #         yaxis16=dict(
                #                 gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'darkgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"), dtick=0.05)
                #         )
                
                # figPoseVel.update_layout(
                #         xaxis17=dict(
                #                 dtick=1, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'darkgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"),),
                #         yaxis17=dict(
                #                 gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'darkgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"),)
                #         )
                
                # figPoseVel.update_layout(
                #         xaxis18=dict(
                #                 dtick=1, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'darkgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"),),
                #         yaxis18=dict(
                #                 gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'darkgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"),)
                #         )



                figPoseVel.add_trace(go.Scatter(
                x=[None],  # Dummy x value
                y=[None],  # Dummy y value
                mode="lines",
                marker=dict(size=10, color=color_Observer),
                name=f"{estimator_plot_args[observerName]['name']}"
                ))
        
        for estimator in estimators:
                if estimator in estimator_plot_args and estimator in estimatorsPoses.keys():
                        plotPoseAndVel(estimator)


        # The nine inset axes are declared above but nothing was ever drawn into them: the code
        # that did is commented out further down, pinned to sample indices of an older trial.
        # Fill the ones that earn their place here, after every main trace exists -- traces are
        # drawn in the order added, so a white patch laid now hides the main curves behind each
        # zoom, and the zoom's own curves go on top of it.
        INSET_TICKS = {}
        drawn = [e for e in estimators
                 if e in estimator_plot_args and e in estimatorsPoses.keys()]
        PANEL_SOURCES = {(1, 1): ('pos', 0), (2, 1): ('pos', 1), (3, 1): ('pos', 2),
                         (1, 2): ('ori', 0), (2, 2): ('ori', 1), (3, 2): ('ori', 2),
                         (1, 3): ('linVel', 0), (2, 3): ('linVel', 1), (3, 3): ('linVel', 2)}
        PANEL_YRANGE = {}
        for (prow, pcol), (pseries, pcomp) in PANEL_SOURCES.items():
                pspan = [estimatorsPoses[e][pseries][:, pcomp] for e in drawn
                         if estimatorsPoses[e].get(pseries) is not None]
                if not pspan:
                        continue
                pflat = np.concatenate(pspan)
                plow, phigh = float(np.nanmin(pflat)), float(np.nanmax(pflat))
                ppad = 0.04 * ((phigh - plow) or 1.0)
                figPoseVel.update_yaxes(range=[plow - ppad, phigh + ppad], row=prow, col=pcol)
                PANEL_YRANGE[(prow, pcol)] = (plow - ppad, phigh + ppad)

        def _rounded(x0, x1, y0, y1, rx, ry, n=10):
                """Polygon of a rounded rectangle, as a filled trace.

                Plotly shapes have no corner radius, and a shape drawn above the traces would
                also cover the inset itself: this is added as a trace, between the panel's curves
                and the inset's, so only the background is muted.
                """
                import math
                pts = []
                for cx, cy, a0 in ((x1 - rx, y0 + ry, -90), (x1 - rx, y1 - ry, 0),
                                   (x0 + rx, y1 - ry, 90), (x0 + rx, y0 + ry, 180)):
                        for k in range(n + 1):
                                a = math.radians(a0 + 90 * k / n)
                                pts.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
                return [q[0] for q in pts] + [pts[0][0]], [q[1] for q in pts] + [pts[0][1]]

        W_FIG, H_FIG = 1000, 400      # same canvas the figure is written at
        _panel_x_span = (float(observer_data["t"].iloc[0]), float(observer_data["t"].iloc[-1]))
        for inset_name, (series, component, row, col) in INSET_TARGETS.items():
                # A half-transparent white pad behind the velocity insets: the curves oscillate at
                # step frequency there and the box's frame and tick labels read poorly over them.
                if inset_name.startswith('vel') and (row, col) in PANEL_YRANGE:
                        n = (row - 1) * 3 + col
                        dx0, dx1 = figPoseVel.layout[f"xaxis{axis_idxs[inset_name]}"].domain
                        dy0, dy1 = figPoseVel.layout[f"yaxis{axis_idxs[inset_name]}"].domain
                        px0, px1 = figPoseVel.layout[f"xaxis{n}"].domain
                        py0, py1 = figPoseVel.layout[f"yaxis{n}"].domain
                        ylo, yhi = PANEL_YRANGE[(row, col)]
                        xlo, xhi = _panel_x_span
                        def _to_x(v): return xlo + (v - px0) / (px1 - px0) * (xhi - xlo)
                        def _to_y(v): return ylo + (v - py0) / (py1 - py0) * (yhi - ylo)
                        # The tick marks and their labels sit outside the box, and unevenly: the
                        # y labels hang to the left, the x labels below. Pad accordingly.
                        wx, wy = dx1 - dx0, dy1 - dy0
                        bx0, bx1 = _to_x(dx0 - 0.34 * wx), _to_x(dx1 + 0.05 * wx)
                        by0, by1 = _to_y(dy0 - 0.49 * wy), _to_y(dy1 + 0.07 * wy)
                        rx = 8.0 / W_FIG / (px1 - px0) * (xhi - xlo)
                        ry = 8.0 / H_FIG / (py1 - py0) * (yhi - ylo)
                        rxs, rys = _rounded(bx0, bx1, by0, by1, rx, ry)
                        figPoseVel.add_trace(go.Scatter(
                                x=rxs, y=rys, fill="toself",
                                fillcolor="rgba(255,255,255,0.7)", mode="lines",
                                line=dict(width=0), hoverinfo="skip", showlegend=False),
                                row=row, col=col)
                lo, hi = INSET_WINDOW[inset_name]
                span = [estimatorsPoses[e][series][lo:hi, component] for e in drawn
                        if estimatorsPoses[e].get(series) is not None]
                if not span:
                        continue
                flat = np.concatenate(span)
                low, high = float(np.nanmin(flat)), float(np.nanmax(flat))
                pad = 0.05 * ((high - low) or 1.0)
                x0, x1 = float(observer_data["t"].iloc[lo]), float(observer_data["t"].iloc[hi - 1])
                figPoseVel.add_trace(go.Scatter(
                        x=[x0, x1, x1, x0], y=[low - pad, low - pad, high + pad, high + pad],
                        fill="toself", fillcolor="white", mode="lines", line=dict(width=0),
                        hoverinfo="skip", showlegend=False,
                        xaxis=f"x{axis_idxs[inset_name]}", yaxis=f"y{axis_idxs[inset_name]}"))

                # The axis grid is painted under every trace, so the white backing above buries
                # it; layer="above traces" does not lift an inset's grid either. Draw it as
                # traces between the backing and the curves, and pin the ticks to the same values.
                ylo, yhi = low - pad, high + pad
                # Whole seconds only, and never rotated: fractional labels on a window a few
                # seconds wide did not fit and plotly tipped them on their side.
                xticks = MaxNLocator(4, integer=True).tick_values(x0, x1)
                yticks = MaxNLocator(4).tick_values(ylo, yhi)
                xticks = [v for v in xticks if x0 <= v <= x1]
                yticks = [v for v in yticks if ylo <= v <= yhi]
                for v in xticks:
                        figPoseVel.add_trace(go.Scatter(
                                x=[v, v], y=[ylo, yhi], mode="lines", hoverinfo="skip",
                                line=dict(color="lightgrey", width=1), showlegend=False,
                                xaxis=f"x{axis_idxs[inset_name]}", yaxis=f"y{axis_idxs[inset_name]}"))
                for v in yticks:
                        figPoseVel.add_trace(go.Scatter(
                                x=[x0, x1], y=[v, v], mode="lines", hoverinfo="skip",
                                line=dict(color="lightgrey", width=1), showlegend=False,
                                xaxis=f"x{axis_idxs[inset_name]}", yaxis=f"y{axis_idxs[inset_name]}"))
                INSET_TICKS[inset_name] = (xticks, yticks, (x0, x1), (ylo, yhi))
                for e in drawn:
                        if estimatorsPoses[e].get(series) is None:
                                continue
                        figPoseVel.add_trace(go.Scatter(
                                x=observer_data["t"][lo:hi],
                                y=estimatorsPoses[e][series][lo:hi, component],
                                mode="lines", showlegend=False,
                                line=dict(width=estimator_plot_args[e]["lineWidth"],
                                          color=paper_colors.rgba(colors, e)),
                                xaxis=f"x{axis_idxs[inset_name]}",
                                yaxis=f"y{axis_idxs[inset_name]}"))

        # The lines that pinned these ranges are commented out further down, so every panel was
        # left to plotly's autorange and its generous padding.


        for inset_name in INSET_TARGETS:
                if inset_name not in INSET_TICKS:
                        continue
                xticks, yticks, xrange, yrange = INSET_TICKS[inset_name]
                tick = dict(family="Times New Roman", size=9, color="black")
                figPoseVel.update_layout({
                        f"xaxis{axis_idxs[inset_name]}": dict(
                                gridcolor="lightgrey", zerolinecolor="lightgrey",
                                linecolor="dimgrey", mirror=True, showline=True,
                                ticks="outside", ticklen=2, tickcolor="lightgrey", tickfont=tick, showgrid=False, tickvals=xticks, tickangle=0,
                                range=list(xrange)),
                        f"yaxis{axis_idxs[inset_name]}": dict(
                                gridcolor="lightgrey", zerolinecolor="lightgrey",
                                linecolor="dimgrey", mirror=True, showline=True,
                                ticks="outside", ticklen=2, tickcolor="lightgrey", tickfont=tick, showgrid=False, tickvals=yticks, range=list(yrange))})

        # Calculate y-axis limits
        def calculate_limits(*datas):
                # Finding the axis limits linked to the max spike
                margin = 0.001

                max_signed_value = 0
                for data in datas:
                        # Find the index of the maximum absolute value
                        max_abs_index = np.argmax(np.abs(data))
                        # Get the value at this index (with the original sign)
                        max_sv = data[max_abs_index]
                        if(np.abs(max_sv) > np.abs(max_signed_value)):
                                max_signed_value = max_sv

                y_min_spike = max_signed_value * (1 - margin * np.sign(max_signed_value))
                y_max_spike = max_signed_value * (1 + margin * np.sign(max_signed_value))

                # Finding the axis limits linked to final values

                final_data = [data[-1] for data in datas]

                min_signed_value = min(final_data)
                max_signed_value = max(final_data)

                # Apply the margin to the min and max signed values
                y_min_end = min_signed_value * (1 - margin * np.sign(min_signed_value))
                y_max_end = max_signed_value * (1 + margin * np.sign(max_signed_value))

                return (min(y_min_spike, y_min_end), max(y_max_spike, y_max_end))

        # y_limits_x = calculate_limits(HartleyBias[:, 0], KineticsBias[:, 0], trueBias[:, 0])
        # y_limits_y = calculate_limits(HartleyBias[:, 1], KineticsBias[:, 1], trueBias[:, 1])
        # y_limits_z = calculate_limits(HartleyBias[:, 2], KineticsBias[:, 2], trueBias[:, 2])

        # Create the figure
        


        # Apply calculated y-axis limits
        # figPoseVel.update_yaxes(range=y_limits_x, row=1, col=1)
        # figPoseVel.update_yaxes(range=y_limits_y, row=2, col=1)
        # figPoseVel.update_yaxes(range=y_limits_z, row=3, col=1)

        # Update layout

        figPoseVel.update_yaxes(title=dict(text="Translation x [m]", standoff=10), row=1, col=1)
        figPoseVel.update_yaxes(title=dict(text="Translation y [m]", standoff=10), row=2, col=1)
        figPoseVel.update_yaxes(title=dict(text="Translation z [m]", standoff=10), row=3, col=1)
        # figPoseVel.update_yaxes(title=dict(text="Roll (°)", standoff=5), row=1, col=2)
        # figPoseVel.update_yaxes(title=dict(text="Pitch (°)", standoff=5), row=2, col=2)
        figPoseVel.update_yaxes(title=dict(text="Roll [deg]", standoff=5), row=1, col=2)
        figPoseVel.update_yaxes(title=dict(text="Pitch [deg]", standoff=5), row=2, col=2)
        figPoseVel.update_yaxes(title=dict(text="Yaw [deg]", standoff=5), row=3, col=2)
        figPoseVel.update_yaxes(title=dict(text="Velocity x [m/s]", standoff=5), row=1, col=3)
        figPoseVel.update_yaxes(title=dict(text="Velocity y [m/s]", standoff=5), row=2, col=3)
        figPoseVel.update_yaxes(title=dict(text="Velocity z [m/s]", standoff=5), row=3, col=3)


        figPoseVel.update_xaxes(title_text="Time [seconds]", row=3, col=1)
        figPoseVel.update_xaxes(title_text="Time [seconds]", row=3, col=2)
        figPoseVel.update_xaxes(title_text="Time [seconds]", row=3, col=3)

        # The source rectangle sits on a panel's axes and the box on the inset's own; a shape
        # cannot mix the two, so both are converted to paper coordinates, which plotly exposes
        # as each axis' domain once the ranges are fixed.
        _panel_x = (float(observer_data["t"].iloc[0]), float(observer_data["t"].iloc[-1]))

        def _paper(axis_name, value, span):
                d0, d1 = figPoseVel.layout[axis_name].domain
                lo, hi = span
                return d0 + (value - lo) / (hi - lo) * (d1 - d0)

        for inset_name, (series, component, row, col) in INSET_TARGETS.items():
                if inset_name not in INSET_TICKS:
                        continue
                _, _, xrange, yrange = INSET_TICKS[inset_name]
                n = (row - 1) * 3 + col
                panel_y = PANEL_YRANGE.get((row, col))
                if panel_y is None:
                        continue
                sx0 = _paper(f"xaxis{n}", xrange[0], _panel_x)
                sx1 = _paper(f"xaxis{n}", xrange[1], _panel_x)
                sy0 = _paper(f"yaxis{n}", yrange[0], panel_y)
                sy1 = _paper(f"yaxis{n}", yrange[1], panel_y)
                bx0, bx1 = figPoseVel.layout[f"xaxis{axis_idxs[inset_name]}"].domain
                by0, by1 = figPoseVel.layout[f"yaxis{axis_idxs[inset_name]}"].domain
                side = INSET_LEADER_SIDE.get(inset_name)
                dx = (bx0 + bx1) / 2 - (sx0 + sx1) / 2
                dy = (by0 + by1) / 2 - (sy0 + sy1) / 2
                if side == "left" or (side is None and abs(dx) >= abs(dy)):
                        # Attach to the box's left corners; the automatic rule picks the facing
                        # edge, which is the bottom one where the box sits mostly above.
                        if side == "left":
                                sx, bx = sx1, bx0
                        else:
                                sx, bx = (sx0, bx1) if dx < 0 else (sx1, bx0)
                        pairs = ((sx, sy1, bx, by1), (sx, sy0, bx, by0))
                else:
                        sy, by = (sy0, by1) if dy < 0 else (sy1, by0)
                        pairs = ((sx0, sy, bx0, by), (sx1, sy, bx1, by))
                for x0, y0, x1, y1 in pairs:
                        figPoseVel.add_shape(type="line", xref="paper", yref="paper",
                                             x0=x0, y0=y0, x1=x1, y1=y1, layer="above",
                                             line=dict(color="grey", width=0.8, dash="3px,2px"))

        W = 1000
        H = int(W/2.5) 
        figPoseVel.update_layout(width=W, height=H)
        
        # Show the plot
        figPoseVel.show()
        

        figPoseVel.write_image(f'/tmp/poseAndVel.svg', width=W, height=H)
        figPoseVel.write_image(f'/tmp/poseAndVel.pdf', width=W, height=H)