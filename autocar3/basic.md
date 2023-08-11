# 시작하기
본 실습은 앞서 구축한 원격 개발환경에서 Pop 라이브러리로 AutoCar3를 제어하는 방법을 소개합니다.  
Pop 라이브러리는 모터를 움직이고, 센서값을 읽으며, 인공지능 예측을 실행하는 등 AutoCar3를 제어하는데 필요한 전반적인 기능을 한백전자에서 파이썬으로 구현한 응용 라이브러리입니다.
실습 준비가 완료되었으면 본문 내용을 하나씩 천천히 따라가길 바랍니다.


# 기본 제어
AutoCar3 온라인 설명서를 참고하며 기본 제어 방법을 실습합니다.

> https://pop-docs.readthedocs.io/en/latest/pop.pilot/#class-autocar

## AutoCar3
### 객체 생성
```python
from pop.Pilot import AutoCar
from time import sleep

car = AutoCar()
```

### IMU 센서
**캘리브레이션**
```python
car.setCalibration()
```

**오일러값 읽기**
```python
car.setSensorStatus(euler=1)
eu = car.getEuler()
print(eu[0], eu[1], eu[2])
car.setSensorStatus(euler=0)
```
### 서보 모터 (스티어링: -1.0 ~ 1.0)
```python
car.steering = 1.0
sleep(1)
car.steering = -1.0
sleep(1)
car.steering = 0
```

### DC 모터 (주행: 20 ~ 100)
```python
car.forward(50)
sleep(1)
car.backward(100)
sleep(1)
car.stop()
```

### LED
```python
car.setLamp(1,0)
sleep(1)
car.setLamp(0,1)
sleep(1)
car.setLamp(0,0)
```

### Ultrasonic (10 ~ 180cm)
```python
for _ in range(10):
    us = car.getUltrasonic()
    print(us[0], us[1])
    sleep(0.1)
``` 
### Buzzer
```python
car.alarm(scale=4,pitch=8,duration=0.5)
sleep(0.5)
car.alarm(scale=4,pitch=8,duration=0.01)
```

### 
```python
from pop.Pilot import AutoCar
from time import sleep

car = AutoCar()
car.setSensorStatus(euler=1)
car.forward(80)
car.steering = 0.0

for i in range(100):
    eu = car.getEuler()
    sleep(0.1)
    print(eu)

car.setSensorStatus(euler=0)
car.stop()
```
