# Flight_System
 A simple flight system.

## Installation

Since the scene of this system is built in Panda3D, you need to download `Panda3D` engine to run this program.

## Example

Here are some scenes from the flight system.

### Single flight

https://user-images.githubusercontent.com/67897233/182586083-14e2c8cf-f415-4ec1-be4e-7d1269a177ae.mp4

### Fight another aircraft



https://user-images.githubusercontent.com/67897233/182591967-61c5ad3e-9374-4f6a-8a92-76e74a348b4c.mp4



## Run

For single flight, 

```python
cd ./single_flight
python flight.py
```

You can change actions of this aircraft by keyboard input.

* `w` and `s` for speed up and speed down.

* `d` and `a` for yaw, `↑` and `↓` for pitch,  `→` and `←` for roll.

You can change the model of the aircraft in folder `/models`. A different model than the one shown in the video was uploaded since the original one is too large. 

For fighting with another aircraft, which is trained by REINFORCE algorithm, 

```python
cd ./render_env
python main.py
python show.py
```

You can press `1` or `2` or `x` or `c` to change camera view.

If you want to train the aircraft by yourself, run

```python
cd ./render_env
python train_reinforce.py
```
