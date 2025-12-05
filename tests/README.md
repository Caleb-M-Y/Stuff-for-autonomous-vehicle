# Unit Tests
Follow the order to test out BearCart's functionality unit by unit.

## 1. Blink Headlight
```console
python led.py
```

## 2. Find Buttons and Axes on Gamepad
```console
python joystick.py
```

## 3. Camera Preview
```console
python camera.py
```

## 4. Serial Communication
### 4.1 Transmit Throttle Dutycycle
```console
python serial_steering.py
```

### 4.2 Transmit Steering Dutycycle
```console
python serial_throttle.py
```

## 5. Control Steering and Throttle using Gamepad
```console
python joystick_drivetrain.py
```

## 6. Integrated Test with All Components
```console
python camera_joystick_drivetrain.py
```
