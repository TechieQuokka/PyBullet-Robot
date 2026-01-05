# Environment Design

## Overview
Custom Gymnasium environment defining Kuka iiwa robot arm pick-and-place tasks

## Gymnasium Interface

### Class Structure
```python
class PickPlaceEnv(gymnasium.Env):
    """
    Cube pick-and-place environment using Kuka iiwa robot arm
    """
    metadata = {'render_modes': ['human', 'rgb_array']}
```

## Observation Space

### Definition
```python
observation_space = spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(17,),  # 7 (joint angles) + 7 (joint velocities) + 3 (cube position)
    dtype=np.float32
)
```

### Components

| Index | Component | Description | Range |
|-------|-----------|-------------|-------|
| 0-6 | Joint Positions | 7 joint angles (radians) | [-2π, 2π] |
| 7-13 | Joint Velocities | 7 joint velocities (rad/s) | [-10, 10] |
| 14-16 | Cube Position | Cube (x, y, z) position | Within workspace |

### Observation Collection
```python
def _get_obs(self):
    # Robot state
    joint_states = p.getJointStates(self.robot_id, self.joint_indices)
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]

    # Cube state
    cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)

    observation = np.concatenate([
        joint_positions,
        joint_velocities,
        cube_pos
    ], dtype=np.float32)

    return observation
```

## Action Space

### Definition
```python
action_space = spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(7,),  # 7 DOF joint velocity commands
    dtype=np.float32
)
```

### Action Scaling
- **Input**: [-1, 1] normalized values
- **Output**: Actual joint velocity commands
- **Max Velocity**: 0.5 rad/s (safe and smooth control)

```python
def _apply_action(self, action):
    # Scale action from [-1, 1] to actual velocity
    max_velocity = 0.5  # rad/s
    velocities = action * max_velocity

    # Apply to each joint
    for i, joint_index in enumerate(self.joint_indices):
        p.setJointMotorControl2(
            bodyUniqueId=self.robot_id,
            jointIndex=joint_index,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=velocities[i],
            force=150  # Maximum force
        )
```

## Environment Configuration

### Workspace Definition
```python
WORKSPACE = {
    'x': [0.3, 0.7],   # Workspace above table
    'y': [-0.3, 0.3],
    'z': [0.0, 0.5]    # Relative to table height
}

CUBE_START_POS = [0.5, 0.0, 0.05]   # Fixed starting position
TARGET_POS = [0.5, 0.2, 0.05]        # Fixed target position
```

### Physical Parameters
```python
CUBE_SIZE = 0.05  # 5cm cube
CUBE_MASS = 0.1   # 100g

TABLE_HEIGHT = 0.0
TABLE_SIZE = [1.0, 1.0, 0.02]

GRIPPER_ATTACH_DISTANCE = 0.03  # Auto-attach within 3cm
SUCCESS_DISTANCE = 0.05          # Success within 5cm
```

## PyBullet Setup

### Initialization
```python
def _setup_simulation(self):
    # Connect to PyBullet
    if self.render_mode == 'human':
        self.physics_client = p.connect(p.GUI)
    else:
        self.physics_client = p.connect(p.DIRECT)

    # Physics settings
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1/240)  # 240 Hz

    # Load plane and table
    self.plane_id = p.loadURDF("plane.urdf")
    self.table_id = self._create_table()

    # Load robot
    self.robot_id = p.loadURDF(
        "kuka_iiwa/model.urdf",
        basePosition=[0, 0, 0],
        useFixedBase=True
    )

    # Create cube
    self.cube_id = self._create_cube()
```

### Robot Configuration
```python
def _configure_robot(self):
    # Get controllable joint indices
    self.joint_indices = []
    for i in range(p.getNumJoints(self.robot_id)):
        info = p.getJointInfo(self.robot_id, i)
        if info[2] == p.JOINT_REVOLUTE:  # Revolute joints only
            self.joint_indices.append(i)

    # Reset to home position
    home_positions = [0, 0, 0, -1.57, 0, 1.57, 0]  # Safe home pose
    for i, joint_idx in enumerate(self.joint_indices):
        p.resetJointState(self.robot_id, joint_idx, home_positions[i])
```

## Episode Management

### Reset Logic
```python
def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    # Reset robot to home position
    self._reset_robot()

    # Reset cube to start position
    p.resetBasePositionAndOrientation(
        self.cube_id,
        self.cube_start_pos,
        [0, 0, 0, 1]  # No rotation
    )

    # Reset internal state
    self.steps = 0
    self.cube_grasped = False

    observation = self._get_obs()
    info = {}

    return observation, info
```

### Step Logic
```python
def step(self, action):
    # Apply action
    self._apply_action(action)

    # Step simulation
    p.stepSimulation()

    # Get new observation
    observation = self._get_obs()

    # Check gripper attachment
    self._check_gripper_attachment()

    # Calculate reward
    reward = self._compute_reward()

    # Check termination
    terminated = self._check_success()
    truncated = self.steps >= self.max_episode_steps

    self.steps += 1

    info = {
        'is_success': terminated,
        'cube_grasped': self.cube_grasped,
        'distance_to_target': self._get_distance_to_target()
    }

    return observation, reward, terminated, truncated, info
```

### Gripper Attachment (Simplified)
```python
def _check_gripper_attachment(self):
    # Get end-effector position
    ee_state = p.getLinkState(self.robot_id, self.ee_link_index)
    ee_pos = ee_state[0]

    # Get cube position
    cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)

    # Calculate distance
    distance = np.linalg.norm(np.array(ee_pos) - np.array(cube_pos))

    # Auto-attach if close enough and not already grasped
    if distance < self.gripper_attach_distance and not self.cube_grasped:
        # Create constraint (virtual attachment)
        self.grasp_constraint = p.createConstraint(
            self.robot_id,
            self.ee_link_index,
            self.cube_id,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            cube_pos
        )
        self.cube_grasped = True
```

## Termination Conditions

### Success Condition
```python
def _check_success(self):
    if not self.cube_grasped:
        return False

    cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
    distance = np.linalg.norm(np.array(cube_pos) - np.array(self.target_pos))

    return distance < self.success_distance  # 5cm
```

### Failure Conditions
- **Timeout**: Exceeding 200 steps
- **Robot Collision**: Self-collision or table collision (optional)
- **Workspace Exit**: Cube leaves workspace bounds

## Rendering

### Visualization Modes
```python
def render(self):
    if self.render_mode == 'human':
        # Already rendering in GUI mode
        time.sleep(1/240)  # Real-time playback

    elif self.render_mode == 'rgb_array':
        # Get camera image
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[1.0, 0.0, 1.0],
            cameraTargetPosition=[0.5, 0.0, 0.0],
            cameraUpVector=[0, 0, 1]
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=100
        )
        (_, _, px, _, _) = p.getCameraImage(
            width=640, height=480,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        return px
```

## Helper Functions

### Distance Calculations
```python
def _get_distance_to_cube(self):
    """End-effector to cube distance"""
    ee_state = p.getLinkState(self.robot_id, self.ee_link_index)
    ee_pos = np.array(ee_state[0])
    cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
    return np.linalg.norm(ee_pos - np.array(cube_pos))

def _get_distance_to_target(self):
    """Cube to target distance"""
    cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
    return np.linalg.norm(np.array(cube_pos) - np.array(self.target_pos))
```

### Collision Detection (Optional)
```python
def _check_collisions(self):
    """Check for self-collision or table collision"""
    # Get all contact points
    contact_points = p.getContactPoints(self.robot_id)

    for contact in contact_points:
        # Check if robot colliding with itself or table (not cube)
        if contact[2] != self.cube_id:
            return True
    return False
```

## Environment Variants (Future)

### Difficulty Levels
```python
# Easy: Fixed positions
CUBE_START_POS = [0.5, 0.0, 0.05]
TARGET_POS = [0.5, 0.2, 0.05]

# Medium: Random start, fixed target
CUBE_START_POS = random_position(workspace)
TARGET_POS = [0.5, 0.2, 0.05]

# Hard: Both random
CUBE_START_POS = random_position(workspace)
TARGET_POS = random_position(workspace)
```

### Curriculum Learning
```python
def increase_difficulty(self):
    """Gradually increase difficulty"""
    # Start with close positions
    # Gradually increase distance
    # Add random rotations
    # Add multiple cubes
```

## Performance Considerations

### Simulation Speed
- **DIRECT mode**: Fast execution during training (~240 Hz)
- **GUI mode**: Visualization during evaluation (~60 Hz)

### Memory Optimization
- Disable unnecessary visual data
- Release constraints at episode end

### Stability
- Set appropriate force limits
- Smooth velocity control
- Safe home position

## Testing Checklist
- [ ] Verify observation space bounds
- [ ] Action space correctly applied
- [ ] Gripper attachment logic works
- [ ] Success condition accurately determined
- [ ] Episode reset fully initializes
- [ ] Rendering modes work properly
- [ ] No memory leaks
