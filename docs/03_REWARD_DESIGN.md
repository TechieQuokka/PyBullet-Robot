# Reward Function Design

## Reward Philosophy

### Shaped Reward (Option C - Selected)
Balance between learning speed and robustness by providing incremental feedback while maintaining clear task objectives.

### Design Principles
1. **Guide exploration**: Provide signals for approaching correct behavior
2. **Avoid over-shaping**: Don't make task trivial or create local optima
3. **Encourage efficiency**: Penalize time waste, reward speed
4. **Clear objectives**: Large rewards for actual task completion

## Reward Components

### 1. Distance-Based Approach Reward

#### Phase 1: Reaching the Cube
```python
def _compute_reach_reward(self):
    """Reward for moving gripper towards cube"""
    ee_pos = self._get_ee_position()
    cube_pos = self._get_cube_position()

    distance = np.linalg.norm(ee_pos - cube_pos)

    # Negative distance (closer is better)
    reach_reward = -0.1 * distance

    return reach_reward
```

**Characteristics:**
- Range: -0.1 to 0 (max distance ~1m → min reward -0.1)
- Smooth gradient guides exploration
- Encourages gripper to approach cube

#### Phase 2: Moving Cube to Target
```python
def _compute_place_reward(self):
    """Reward for moving cube towards target (only if grasped)"""
    if not self.cube_grasped:
        return 0.0

    cube_pos = self._get_cube_position()
    target_pos = self.target_pos

    distance = np.linalg.norm(cube_pos - target_pos)

    # Negative distance (closer is better)
    place_reward = -0.1 * distance

    return place_reward
```

**Characteristics:**
- Only active when cube is grasped
- Prevents reward for moving cube unintentionally
- Smooth gradient towards target

### 2. Grasp Success Bonus
```python
def _compute_grasp_bonus(self):
    """Large bonus for successfully grasping the cube"""
    if self.cube_grasped and not self.grasp_bonus_given:
        self.grasp_bonus_given = True
        return 50.0
    return 0.0
```

**Characteristics:**
- One-time bonus: +50
- Significant milestone reward
- Prevents repeated bonuses for same grasp

### 3. Task Completion Bonus
```python
def _compute_success_bonus(self):
    """Large bonus for successfully placing cube at target"""
    if self._check_success():
        return 100.0
    return 0.0
```

**Characteristics:**
- One-time bonus: +100
- Ultimate objective achievement
- Episode terminates after this

### 4. Time Penalty
```python
def _compute_time_penalty(self):
    """Small penalty for each timestep to encourage efficiency"""
    return -0.01
```

**Characteristics:**
- Per-step penalty: -0.01
- Encourages faster solutions
- Prevents dawdling behavior

### 5. Collision Penalty (Optional)
```python
def _compute_collision_penalty(self):
    """Penalty for unwanted collisions (self-collision or dropping cube)"""
    penalty = 0.0

    # Self-collision
    if self._check_self_collision():
        penalty -= 1.0

    # Cube dropped (fell off table)
    if self._check_cube_dropped():
        penalty -= 5.0
        self.terminated = True

    return penalty
```

**Characteristics:**
- Discourages dangerous behaviors
- Can be disabled for simpler learning

## Complete Reward Function

### Implementation
```python
def _compute_reward(self):
    """
    Complete reward function combining all components

    Returns:
        float: Total reward for current step
    """
    reward = 0.0

    # Distance-based rewards
    if not self.cube_grasped:
        # Phase 1: Approach cube
        reward += self._compute_reach_reward()
    else:
        # Phase 2: Move to target
        reward += self._compute_place_reward()

    # Milestone bonuses
    reward += self._compute_grasp_bonus()
    reward += self._compute_success_bonus()

    # Per-step penalty
    reward += self._compute_time_penalty()

    # Safety penalties (optional)
    # reward += self._compute_collision_penalty()

    return reward
```

### Reward Ranges

| Component | Min | Max | Typical |
|-----------|-----|-----|---------|
| Reach reward | -0.1 | 0 | -0.05 |
| Place reward | -0.1 | 0 | -0.05 |
| Grasp bonus | 0 | 50 | 0 or 50 |
| Success bonus | 0 | 100 | 0 or 100 |
| Time penalty | -0.01 | -0.01 | -0.01 |
| **Total (before grasp)** | -0.11 | -0.01 | -0.06 |
| **Total (grasping)** | 49.89 | 49.99 | 49.94 |
| **Total (success)** | 99.89 | 99.99 | 99.94 |

## Expected Reward Trajectory

### Episode Reward Over Time

```
Episode:      1    10    50   100   200   500  1000
Avg Reward: -20   -15    -5    +5   +15   +40   +60

Phase 1 (0-200 episodes): Learning to approach cube
  - Mostly negative rewards
  - Gradual improvement in reach behavior

Phase 2 (200-500 episodes): Learning to grasp
  - Occasional +50 bonuses
  - Average reward becomes positive

Phase 3 (500-1000 episodes): Learning complete task
  - Frequent grasp bonuses
  - Occasional +100 success bonuses
  - Average reward > +50
```

### Within-Episode Reward

```
Successful Episode (~100 steps):
Step   1-30:  -0.06 per step (approaching cube)
Step     31:  +49.94 (grasp bonus + approach reward)
Step  32-80:  -0.06 per step (moving to target)
Step     81:  +99.94 (success bonus + place reward)
Total: 30*(-0.06) + 49.94 + 49*(-0.06) + 99.94 = +145

Failed Episode (~200 steps, timeout):
Step 1-200:   -0.06 per step (random exploration)
Total: 200*(-0.06) = -12
```

## Reward Shaping Best Practices

### Do's ✅
- **Smooth gradients**: Continuous rewards for continuous progress
- **Milestone rewards**: Significant bonuses for important achievements
- **Normalize scale**: Keep reward components on similar scales
- **Phase-based**: Different rewards for different task phases

### Don'ts ❌
- **Over-dense rewards**: Too many reward components confuse learning
- **Conflicting signals**: Rewards that contradict each other
- **Sparse-only**: Pure sparse rewards learn very slowly
- **Unbounded rewards**: Distance rewards should have reasonable bounds

## Debugging Rewards

### Verification Checklist
```python
def test_reward_function():
    """Test reward function sanity"""
    env = PickPlaceEnv()

    # Test 1: Initial state
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    assert -1 < reward < 1, "Initial reward out of bounds"

    # Test 2: Grasp bonus
    env.cube_grasped = True
    env.grasp_bonus_given = False
    reward = env._compute_reward()
    assert reward > 40, "Grasp bonus not working"

    # Test 3: Success bonus
    env._set_cube_at_target()
    reward = env._compute_reward()
    assert reward > 90, "Success bonus not working"

    print("All reward tests passed!")
```

### Common Issues

#### Reward Hacking
**Problem**: Agent finds unintended way to maximize reward
**Example**: Pushing cube instead of grasping
**Solution**: Add constraint (only reward placement if grasped)

#### Sparse Learning
**Problem**: Agent never discovers grasp
**Example**: Random exploration never triggers +50 bonus
**Solution**: Increase reach reward coefficient or add intermediate milestones

#### Reward Saturation
**Problem**: Agent stops improving after reaching local optimum
**Example**: Learns to grasp but not place
**Solution**: Balance bonus magnitudes, ensure place reward is significant

## Reward Tuning Guide

### If Agent Not Approaching Cube
```python
# Increase reach reward coefficient
reach_reward = -0.2 * distance  # instead of -0.1
```

### If Agent Not Grasping
```python
# Increase grasp bonus
grasp_bonus = 100.0  # instead of 50.0

# Add proximity bonus
if distance < 0.05 and not cube_grasped:
    reward += 10.0  # Close to grasping
```

### If Agent Grasps But Doesn't Place
```python
# Increase place reward coefficient
place_reward = -0.2 * distance  # instead of -0.1

# Reduce time penalty to allow more exploration
time_penalty = -0.005  # instead of -0.01
```

### If Episodes Too Long
```python
# Increase time penalty
time_penalty = -0.02  # instead of -0.01

# Add step limit penalty
if self.steps > 150:
    reward -= 0.1  # Strong encouragement to finish
```

## Alternative Reward Functions

### Sparse Reward (Option A)
```python
def _compute_reward_sparse(self):
    """Pure sparse reward - only success matters"""
    if self._check_success():
        return 1.0
    return 0.0
```
**Pros**: Simplest, no reward hacking
**Cons**: Very slow learning, may never discover solution

### Dense Reward (Option B)
```python
def _compute_reward_dense(self):
    """Very dense reward - continuous feedback"""
    reward = 0.0

    # Always penalize both distances
    reward -= self._get_distance_to_cube()
    reward -= self._get_distance_to_target()

    # Grasp bonus
    if self.cube_grasped:
        reward += 10.0

    # Success bonus
    if self._check_success():
        reward += 100.0

    return reward
```
**Pros**: Fastest learning initially
**Cons**: May overfit, potential reward hacking

### Curriculum Reward
```python
def _compute_reward_curriculum(self):
    """Gradually increase difficulty"""
    if self.curriculum_stage == 1:
        # Stage 1: Just approach
        return -0.1 * self._get_distance_to_cube()

    elif self.curriculum_stage == 2:
        # Stage 2: Approach and grasp
        reward = -0.1 * self._get_distance_to_cube()
        reward += self._compute_grasp_bonus()
        return reward

    else:
        # Stage 3: Full task
        return self._compute_reward()  # Full shaped reward
```
**Pros**: Progressive learning, robust
**Cons**: Requires stage switching logic

## Monitoring Reward Distribution

### TensorBoard Logging
```python
# In training loop
self.writer.add_scalar('reward/total', episode_reward, episode)
self.writer.add_scalar('reward/reach', reach_reward_sum, episode)
self.writer.add_scalar('reward/place', place_reward_sum, episode)
self.writer.add_scalar('reward/grasp_bonus', grasp_bonus_count, episode)
self.writer.add_scalar('reward/success_bonus', success_count, episode)
```

### Expected Distributions
- **Early training**: Mostly reach rewards, few bonuses
- **Mid training**: Balanced reach/place rewards, some grasp bonuses
- **Late training**: Efficient episodes, frequent success bonuses

## Reward Function Validation

### Unit Tests
```python
def test_reward_components():
    env = PickPlaceEnv()

    # Test monotonicity
    env.reset()
    for _ in range(10):
        action = [0.1] * 7  # Move towards cube
        _, reward1, _, _, _ = env.step(action)
        _, reward2, _, _, _ = env.step(action)
        # Reward should not decrease if getting closer

    # Test bonus trigger
    env.cube_grasped = True
    reward = env._compute_reward()
    assert 40 < reward < 60, "Grasp bonus not in expected range"
```

### Integration Tests
```python
def test_full_episode():
    env = PickPlaceEnv()
    total_reward = 0

    obs, _ = env.reset()
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        if done or truncated:
            break

    # Total reward should be reasonable
    assert -50 < total_reward < 200
```
