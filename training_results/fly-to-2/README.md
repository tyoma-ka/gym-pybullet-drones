# Training Result - fly-to

## Experiment Summary  
- **Date:** `26.03.2025`  
- **Purpose:** `fly to the point [1,0,1] and hover there`  
- **Algorithm Used:** `PPO`  

---

## Training Parameters  

| Parameter              | Value       |
|------------------------|-------------|
| **Episode Duration**   | `4 seconds` |
| **Number of Steps**    | `1.000.000` |
---

## Reward Function Used  

```python
state = self._getDroneStateVector(0)
ret = max(0, 2 - np.linalg.norm(self.TARGET_POS-state[0:3]))
return ret
```

## Terminated Function Used  
```python
state = self._getDroneStateVector(0)
if np.linalg.norm(self.TARGET_POS-state[0:3]) < .0001:
    return True
else:
    return False
```

## Truncated Function Used  
```python
state = self._getDroneStateVector(0)
if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0 # Truncate when the drone is too far away
     or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
):
    return True
if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
    return True
else:
    return False
```