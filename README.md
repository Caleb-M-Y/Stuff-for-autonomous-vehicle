# BearCar

![bearcar_portrait](/images/bearcar_portrait.png)

BearCar is an autonomous driving platform based on an 1:16 RC car and a Raspberry Pi SBC.
Visit [documentations](https://ucaengineeringphysics.github.io/bearcar_docs/) for more details.

This project is strongly inspired by the
[DonkeyCar](https://github.com/autorope/donkeycar) project.

## Quick Start

Fire up the terminal on your Raspberry Pi, and run following commands in it.

### Download BearCar software

```bash
cd ~
git clone https://github.com/UCAEngineeringPhysics/BearCar.git
```

### Setup Environment

```bash
cd ~/BearCar
./setup_pi_env.sh
```

### Hook Up the Gamepad

### Have Fun Racing

```bash
cd ~/BearCar
cp models/example_pilot models/pilot.pth
uv run scripts/autopilot.py
```

## Demo Videos

- [Initial BearCar](https://youtube.com/shorts/Kcm6qQqev3s)
- [Another Autopilot](https://youtu.be/8GX6HnfgrJQ)
